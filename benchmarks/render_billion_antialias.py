"""Render deterministic antialiasing comparisons for the billion-point image.

The saved ShuffleSnap assignment is reused directly. The deterministic source
point stream is cached once as float32 coordinates so subsequent rendering
runs never regenerate it. All rasterizers use integer GPU accumulation, which
makes their 8-bit results independent of atomic scheduling order.

Default outputs (all 4800 x 4800 RGB PNGs) are written together under:

  results/renders/billion_shufflesnap_31623_seed0_interp75_aa_4800/

Methods:

* nearest_native: current nearest-pixel averaging, as a reference.
* bilinear_native: subpixel samples distributed over four pixel centers.
* smoothstep_native: subpixel circular footprints with smoothstep edges.
* disk_2x_lanczos: 9600-square 2 px smoothstep disks, Lanczos to 4800.
* disk_4x_lanczos: 19200-square 4 px smoothstep disks, Lanczos to 4800.
* disk_2x/4x_gaussian_area: Gaussian prefilter, then positive-lobe area reduction.
* disk_2x/4x_box_area: scale-wide box prefilter, then positive-lobe area reduction.

The supersampled disk methods first accumulate point impulses, then convolve
both color sums and weights with a radial smoothstep disk. This is the linear
equivalent of splatting the same disk per point, while avoiding billions of
redundant random atomic writes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_billion import CHUNK as GENERATION_CHUNK, generate_meander
from bench_common import MARGIN, results_dir


DEFAULT_SIDE = 31_623
DEFAULT_SEED = 0
DEFAULT_INTERP = 0.75
DEFAULT_OUT_SIZE = 4_800
DEFAULT_STREAM_CHUNK = 20_000_000
THREADS = 256
WEIGHT_SCALE = 256
ICC_CANDIDATES = (
    Path("/usr/share/color/icc/colord/sRGB.icc"),
    Path("/usr/share/color/icc/ghostscript/srgb.icc"),
)


@dataclass(frozen=True)
class Method:
    key: str
    filename: str
    description: str
    scale: int = 1
    disk_diameter: float | None = None
    downsample_mode: str | None = None


METHODS = (
    Method(
        "nearest_native",
        "01_nearest_native_4800.png",
        "Native 4800-square nearest-pixel accumulation and per-pixel normalization",
    ),
    Method(
        "bilinear_native",
        "02_bilinear_native_4800.png",
        "Native 4800-square four-bin bilinear accumulation and normalization",
    ),
    Method(
        "smoothstep_native",
        "03_smoothstep_native_4800.png",
        "Native 4800-square subpixel circular splats (1.5 px diameter, smoothstep edge)",
    ),
    Method(
        "disk_2x_lanczos",
        "04_disk_2x_9600_lanczos_4800.png",
        "9600-square 2 px smoothstep disks, normalized then Lanczos-downsampled",
        scale=2,
        disk_diameter=2.0,
        downsample_mode="lanczos",
    ),
    Method(
        "disk_4x_lanczos",
        "05_disk_4x_19200_lanczos_4800.png",
        "19200-square 4 px smoothstep disks, normalized then Lanczos-downsampled",
        scale=4,
        disk_diameter=4.0,
        downsample_mode="lanczos",
    ),
    Method(
        "disk_2x_gaussian_area",
        "06_disk_2x_9600_gaussian_area_4800.png",
        "9600-square 2 px smoothstep disks, Gaussian sigma 1 px, then area-downsampled",
        scale=2,
        disk_diameter=2.0,
        downsample_mode="gaussian_area",
    ),
    Method(
        "disk_4x_gaussian_area",
        "07_disk_4x_19200_gaussian_area_4800.png",
        "19200-square 4 px smoothstep disks, Gaussian sigma 2 px, then area-downsampled",
        scale=4,
        disk_diameter=4.0,
        downsample_mode="gaussian_area",
    ),
    Method(
        "disk_2x_box_area",
        "08_disk_2x_9600_box_area_4800.png",
        "9600-square 2 px smoothstep disks, 2 px box prefilter, then area-downsampled",
        scale=2,
        disk_diameter=2.0,
        downsample_mode="box_area",
    ),
    Method(
        "disk_4x_box_area",
        "09_disk_4x_19200_box_area_4800.png",
        "19200-square 4 px smoothstep disks, 4 px box prefilter, then area-downsampled",
        scale=4,
        disk_diameter=4.0,
        downsample_mode="box_area",
    ),
)
METHOD_BY_KEY = {method.key: method for method in METHODS}


def log(message: str) -> None:
    print(message, flush=True)


def elapsed(start: float) -> float:
    return time.perf_counter() - start


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_icc() -> tuple[bytes, str]:
    for path in ICC_CANDIDATES:
        if path.is_file():
            return path.read_bytes(), str(path)
    raise RuntimeError("No standard sRGB ICC profile was found")


def assignment_path(side: int, seed: int) -> Path:
    return results_dir() / "assignments" / f"billion_meander_{side}_seed{seed}.npy"


def position_cache_path(side: int, seed: int) -> Path:
    return results_dir() / "positions" / f"billion_meander_{side}_seed{seed}_float32.npy"


def validate_assignment(path: Path, n: int) -> np.memmap:
    if not path.is_file():
        raise RuntimeError(f"Saved billion-point assignment is missing: {path}")
    assignment = np.load(path, mmap_mode="r")
    if assignment.shape != (n,) or assignment.dtype != np.int32:
        raise RuntimeError(
            f"Assignment must have shape {(n,)} and dtype int32; "
            f"found {assignment.shape} {assignment.dtype}"
        )
    return assignment


def prepare_position_cache(side: int, seed: int) -> tuple[np.memmap, dict[str, Any]]:
    n = side * side
    path = position_cache_path(side, seed)
    if path.is_file():
        points = np.load(path, mmap_mode="r")
        if points.shape != (n, 2) or points.dtype != np.float32:
            raise RuntimeError(
                f"Position cache must have shape {(n, 2)} and dtype float32; "
                f"found {points.shape} {points.dtype}"
            )
        log(f"POSITION CACHE reused {path} ({points.nbytes / 2**30:.2f} GiB)")
        metadata = {
            "created": False,
            "seconds": 0.0,
            "path": str(path),
            "bytes": int(points.nbytes),
            "dtype": str(points.dtype),
            "shape": list(points.shape),
        }
        creation_record = path.with_suffix(".json")
        if creation_record.is_file():
            metadata["initialCreation"] = json.loads(creation_record.read_text(encoding="utf-8"))
        return points, metadata

    required = n * 2 * np.dtype(np.float32).itemsize
    free = shutil.disk_usage(path.parent.parent).free
    if free < required + 8 * 2**30:
        raise RuntimeError(
            f"Position cache needs {required / 2**30:.1f} GiB plus an 8 GiB safety reserve, "
            f"but only {free / 2**30:.1f} GiB is free"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npy")
    if temporary.exists():
        raise RuntimeError(f"Incomplete position-cache temporary already exists: {temporary}")

    start = time.perf_counter()
    log("POSITION CACHE missing; reconstructing the deterministic source stream once")
    points64 = generate_meander(n, seed=seed, chunk=GENERATION_CHUNK)
    generation_seconds = elapsed(start)
    log(f"POSITION CACHE source reconstruction {generation_seconds:.3f}s")

    conversion_start = time.perf_counter()
    cached = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(n, 2)
    )
    for i0 in range(0, n, GENERATION_CHUNK):
        i1 = min(i0 + GENERATION_CHUNK, n)
        cached[i0:i1] = points64[i0:i1]
    cached.flush()
    del cached, points64
    gc.collect()
    temporary.replace(path)
    conversion_seconds = elapsed(conversion_start)
    total_seconds = elapsed(start)
    points = np.load(path, mmap_mode="r")
    log(
        f"POSITION CACHE wrote {path} ({points.nbytes / 2**30:.2f} GiB) "
        f"in {total_seconds:.3f}s"
    )
    metadata = {
        "created": True,
        "seconds": total_seconds,
        "generationSeconds": generation_seconds,
        "float32WriteSeconds": conversion_seconds,
        "path": str(path),
        "bytes": int(points.nbytes),
        "dtype": str(points.dtype),
        "shape": list(points.shape),
    }
    atomic_write_json(path.with_suffix(".json"), metadata)
    return points, metadata


@cuda.jit(device=True, inline=True)
def _inv_lab_f(value: float) -> float:
    delta = 6.0 / 29.0
    if value > delta:
        return value * value * value
    return 3.0 * delta * delta * (value - 4.0 / 29.0)


@cuda.jit(device=True, inline=True)
def _srgb_gamma(value: float) -> float:
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


@cuda.jit(device=True, inline=True)
def _point_rgb_float(px: float, py: float) -> tuple[float, float, float]:
    """Return the continuous sRGB channel values on the 0..255 scale."""
    fy = 90.0 / 116.0
    fx = fy + (px - 0.5) * 160.0 / 500.0
    fz = fy - ((py - 0.5) * -160.0) / 200.0
    xx = 0.95047 * _inv_lab_f(fx)
    yy = _inv_lab_f(fy)
    zz = 1.08883 * _inv_lab_f(fz)
    rr = _srgb_gamma(3.2406 * xx - 1.5372 * yy - 0.4986 * zz)
    gg = _srgb_gamma(-0.9689 * xx + 1.8758 * yy + 0.0415 * zz)
    bb = _srgb_gamma(0.0557 * xx - 0.2040 * yy + 1.0570 * zz)
    return rr * 255.0, gg * 255.0, bb * 255.0


@cuda.jit(device=True, inline=True)
def _point_rgb(px: float, py: float) -> tuple[int, int, int]:
    red, green, blue = _point_rgb_float(px, py)
    return int(red + 0.5), int(green + 0.5), int(blue + 0.5)


@cuda.jit(device=True, inline=True)
def _render_xy(px: float, py: float, cell: int, side: int,
               interpolation: float) -> tuple[float, float]:
    row = cell // side
    col = cell - row * side
    grid_scale = (1.0 - 2.0 * MARGIN) / (side - 1)
    target_x = MARGIN + col * grid_scale
    target_y = MARGIN + row * grid_scale
    source_weight = 1.0 - interpolation
    return (
        source_weight * px + interpolation * target_x,
        source_weight * py + interpolation * target_y,
    )


@cuda.jit
def scatter_nearest(points: np.ndarray, cells: np.ndarray, count: int,
                    accumulation: np.ndarray, render_size: int, side: int,
                    interpolation: float) -> None:
    index = cuda.grid(1)
    if index >= count:
        return
    px = points[index, 0]
    py = points[index, 1]
    x, y = _render_xy(px, py, cells[index], side, interpolation)
    ix = int(x * render_size)
    iy = int(y * render_size)
    if ix < 0:
        ix = 0
    elif ix >= render_size:
        ix = render_size - 1
    if iy < 0:
        iy = 0
    elif iy >= render_size:
        iy = render_size - 1
    flat = iy * render_size + ix
    red, green, blue = _point_rgb(px, py)
    cuda.atomic.add(accumulation, (flat, 0), red)
    cuda.atomic.add(accumulation, (flat, 1), green)
    cuda.atomic.add(accumulation, (flat, 2), blue)
    cuda.atomic.add(accumulation, (flat, 3), 1)


@cuda.jit
def scatter_bilinear(points: np.ndarray, cells: np.ndarray, count: int,
                     accumulation: np.ndarray, render_size: int, side: int,
                     interpolation: float) -> None:
    index = cuda.grid(1)
    if index >= count:
        return
    px = points[index, 0]
    py = points[index, 1]
    x, y = _render_xy(px, py, cells[index], side, interpolation)
    u = x * render_size - 0.5
    v = y * render_size - 0.5
    ix = math.floor(u)
    iy = math.floor(v)
    fx = u - ix
    fy = v - iy
    red, green, blue = _point_rgb(px, py)
    for dy in range(2):
        yy = iy + dy
        if yy < 0 or yy >= render_size:
            continue
        wy = 1.0 - fy if dy == 0 else fy
        for dx in range(2):
            xx = ix + dx
            if xx < 0 or xx >= render_size:
                continue
            wx = 1.0 - fx if dx == 0 else fx
            weight = int(wx * wy * WEIGHT_SCALE + 0.5)
            if weight == 0:
                continue
            flat = yy * render_size + xx
            cuda.atomic.add(accumulation, (flat, 0), red * weight)
            cuda.atomic.add(accumulation, (flat, 1), green * weight)
            cuda.atomic.add(accumulation, (flat, 2), blue * weight)
            cuda.atomic.add(accumulation, (flat, 3), weight)


@cuda.jit(device=True, inline=True)
def _smoothstep_disk_weight(distance: float, radius: float,
                            antialias_half_width: float) -> float:
    inner = radius - antialias_half_width
    outer = radius + antialias_half_width
    if distance <= inner:
        return 1.0
    if distance >= outer:
        return 0.0
    position = (distance - inner) / (outer - inner)
    smooth = position * position * (3.0 - 2.0 * position)
    return 1.0 - smooth


@cuda.jit
def scatter_smoothstep(points: np.ndarray, cells: np.ndarray, count: int,
                       accumulation: np.ndarray, render_size: int, side: int,
                       interpolation: float, radius: float,
                       antialias_half_width: float) -> None:
    index = cuda.grid(1)
    if index >= count:
        return
    px = points[index, 0]
    py = points[index, 1]
    x, y = _render_xy(px, py, cells[index], side, interpolation)
    u = x * render_size - 0.5
    v = y * render_size - 0.5
    base_x = math.floor(u)
    base_y = math.floor(v)
    red, green, blue = _point_rgb(px, py)

    # radius=.75 and AA half-width=.5 have support 1.25 px. Relative to
    # floor(u/v), [-1, 2] covers every possibly nonzero pixel center.
    for dy in range(-1, 3):
        yy = base_y + dy
        if yy < 0 or yy >= render_size:
            continue
        delta_y = yy - v
        for dx in range(-1, 3):
            xx = base_x + dx
            if xx < 0 or xx >= render_size:
                continue
            delta_x = xx - u
            distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
            coverage = _smoothstep_disk_weight(distance, radius, antialias_half_width)
            weight = int(coverage * WEIGHT_SCALE + 0.5)
            if weight == 0:
                continue
            flat = yy * render_size + xx
            cuda.atomic.add(accumulation, (flat, 0), red * weight)
            cuda.atomic.add(accumulation, (flat, 1), green * weight)
            cuda.atomic.add(accumulation, (flat, 2), blue * weight)
            cuda.atomic.add(accumulation, (flat, 3), weight)


@cuda.jit
def normalize_accumulation(accumulation: np.ndarray, image: np.ndarray) -> None:
    index = cuda.grid(1)
    if index >= accumulation.shape[0]:
        return
    weight = accumulation[index, 3]
    if weight == 0:
        image[index, 0] = 255
        image[index, 1] = 255
        image[index, 2] = 255
        return
    half = weight // 2
    image[index, 0] = (accumulation[index, 0] + half) // weight
    image[index, 1] = (accumulation[index, 1] + half) // weight
    image[index, 2] = (accumulation[index, 2] + half) // weight


@cuda.jit
def resolve_disk(accumulation: np.ndarray, image: np.ndarray, render_size: int,
                 kernel: np.ndarray, kernel_width: int) -> None:
    index = cuda.grid(1)
    if index >= accumulation.shape[0]:
        return
    y = index // render_size
    x = index - y * render_size
    radius = kernel_width // 2
    sum_red = np.uint64(0)
    sum_green = np.uint64(0)
    sum_blue = np.uint64(0)
    sum_weight = np.uint64(0)
    for ky in range(kernel_width):
        yy = y + ky - radius
        if yy < 0 or yy >= render_size:
            continue
        for kx in range(kernel_width):
            xx = x + kx - radius
            if xx < 0 or xx >= render_size:
                continue
            kernel_weight = np.uint64(kernel[ky * kernel_width + kx])
            if kernel_weight == 0:
                continue
            source = yy * render_size + xx
            point_count = np.uint64(accumulation[source, 3])
            weighted_count = point_count * kernel_weight
            sum_weight += weighted_count
            sum_red += np.uint64(accumulation[source, 0]) * kernel_weight
            sum_green += np.uint64(accumulation[source, 1]) * kernel_weight
            sum_blue += np.uint64(accumulation[source, 2]) * kernel_weight
    if sum_weight == 0:
        image[index, 0] = 255
        image[index, 1] = 255
        image[index, 2] = 255
        return
    half = sum_weight // 2
    image[index, 0] = (sum_red + half) // sum_weight
    image[index, 1] = (sum_green + half) // sum_weight
    image[index, 2] = (sum_blue + half) // sum_weight


def smoothstep_kernel(diameter: float) -> np.ndarray:
    radius = diameter / 2.0
    antialias_half_width = 0.5
    support = radius + antialias_half_width
    half = math.ceil(support)
    axis = np.arange(-half, half + 1, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    distance = np.sqrt(xx * xx + yy * yy)
    inner = radius - antialias_half_width
    position = np.clip((distance - inner) / (2.0 * antialias_half_width), 0.0, 1.0)
    weights = 1.0 - position * position * (3.0 - 2.0 * position)
    weights[distance >= support] = 0.0
    fixed = np.rint(weights * 65_535.0).astype(np.uint32)
    nonzero_rows = np.flatnonzero(np.any(fixed, axis=1))
    nonzero_cols = np.flatnonzero(np.any(fixed, axis=0))
    fixed = fixed[
        nonzero_rows[0]:nonzero_rows[-1] + 1,
        nonzero_cols[0]:nonzero_cols[-1] + 1,
    ]
    if fixed.shape[0] != fixed.shape[1] or fixed.shape[0] % 2 != 1:
        raise RuntimeError("Smoothstep disk kernel must be odd and square")
    return fixed


def warm_cuda_kernels() -> None:
    points = np.array([[0.25, 0.75], [0.75, 0.25]], dtype=np.float32)
    cells = np.array([0, 3], dtype=np.int32)
    d_points = cuda.to_device(points)
    d_cells = cuda.to_device(cells)
    d_accumulation = cuda.to_device(np.zeros((16, 4), dtype=np.uint32))
    for kernel in (scatter_nearest, scatter_bilinear):
        kernel[1, 32](d_points, d_cells, 2, d_accumulation, 4, 2, np.float32(0.75))
    scatter_smoothstep[1, 32](
        d_points, d_cells, 2, d_accumulation, 4, 2, np.float32(0.75),
        np.float32(0.75), np.float32(0.5),
    )
    d_image = cuda.device_array((16, 3), dtype=np.uint8)
    normalize_accumulation[1, 32](d_accumulation, d_image)
    disk = smoothstep_kernel(2.0)
    d_disk = cuda.to_device(disk.ravel())
    resolve_disk[1, 32](d_accumulation, d_image, 4, d_disk, disk.shape[0])
    cuda.synchronize()
    del d_points, d_cells, d_accumulation, d_image, d_disk
    gc.collect()
    cuda.current_context().deallocations.clear()


def check_gpu_memory(pixel_count: int, needs_image: bool = True) -> None:
    accumulation_bytes = pixel_count * 4 * np.dtype(np.uint32).itemsize
    image_bytes = pixel_count * 3 if needs_image else 0
    reserve = 2 * 2**30
    required = accumulation_bytes + image_bytes + reserve
    free, _ = cuda.current_context().get_memory_info()
    if free < required:
        raise RuntimeError(
            f"Render requires about {required / 2**30:.2f} GiB of free GPU memory "
            f"including reserve, but only {free / 2**30:.2f} GiB is free"
        )


def stream_points(points: np.memmap, assignment: np.memmap,
                  accumulation: Any, method: Method, side: int,
                  interpolation: float, out_size: int, stream_chunk: int) -> float:
    n = points.shape[0]
    render_size = out_size * method.scale
    device_points = cuda.device_array((stream_chunk, 2), dtype=np.float32)
    device_cells = cuda.device_array(stream_chunk, dtype=np.int32)
    start = time.perf_counter()
    progress_interval = max(1, math.ceil(n / stream_chunk / 10))
    chunk_number = 0
    for i0 in range(0, n, stream_chunk):
        i1 = min(i0 + stream_chunk, n)
        count = i1 - i0
        device_points[:count].copy_to_device(points[i0:i1])
        device_cells[:count].copy_to_device(assignment[i0:i1])
        blocks = (count + THREADS - 1) // THREADS
        if method.key == "nearest_native" or method.disk_diameter is not None:
            scatter_nearest[blocks, THREADS](
                device_points, device_cells, count, accumulation, render_size, side,
                np.float32(interpolation),
            )
        elif method.key == "bilinear_native":
            scatter_bilinear[blocks, THREADS](
                device_points, device_cells, count, accumulation, render_size, side,
                np.float32(interpolation),
            )
        elif method.key == "smoothstep_native":
            scatter_smoothstep[blocks, THREADS](
                device_points, device_cells, count, accumulation, render_size, side,
                np.float32(interpolation), np.float32(0.75), np.float32(0.5),
            )
        else:
            raise RuntimeError(f"Unhandled method: {method.key}")
        chunk_number += 1
        if chunk_number % progress_interval == 0 or i1 == n:
            cuda.synchronize()
            seconds = elapsed(start)
            log(
                f"  {method.key}: {i1:,}/{n:,} points "
                f"({100.0 * i1 / n:5.1f}%) in {seconds:.2f}s "
                f"[{i1 / max(seconds, 1e-9) / 1e6:.1f} Mpoints/s]"
            )
    cuda.synchronize()
    seconds = elapsed(start)
    del device_points, device_cells
    return seconds


def resolve_to_host(accumulation: Any, method: Method,
                    render_size: int) -> tuple[np.ndarray, float, dict[str, Any]]:
    pixel_count = render_size * render_size
    device_image = cuda.device_array((pixel_count, 3), dtype=np.uint8)
    blocks = (pixel_count + THREADS - 1) // THREADS
    start = time.perf_counter()
    details: dict[str, Any] = {}
    if method.disk_diameter is None:
        normalize_accumulation[blocks, THREADS](accumulation, device_image)
    else:
        disk = smoothstep_kernel(method.disk_diameter)
        device_disk = cuda.to_device(disk.ravel())
        resolve_disk[blocks, THREADS](
            accumulation, device_image, render_size, device_disk, disk.shape[0]
        )
        details = {
            "diskDiameterHighResolutionPixels": method.disk_diameter,
            "diskKernel": disk.tolist(),
            "diskKernelShape": list(disk.shape),
            "diskKernelWeightSum": int(disk.sum()),
            "diskAntialiasHalfWidthPixels": 0.5,
        }
    cuda.synchronize()
    host = device_image.copy_to_host().reshape(render_size, render_size, 3)
    seconds = elapsed(start)
    del device_image
    if method.disk_diameter is not None:
        del device_disk
    return host, seconds, details


def downsample_lanczos(image: np.ndarray, out_size: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    source = Image.fromarray(image)
    resized = source.resize((out_size, out_size), resample=Image.Resampling.LANCZOS)
    result = np.asarray(resized, dtype=np.uint8).copy()
    source.close()
    resized.close()
    return result, elapsed(start)


def downsample_filtered_area(image: np.ndarray, out_size: int, scale: int,
                             mode: str) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Apply a positive prefilter, then BOX-resize as integer area reduction."""
    start = time.perf_counter()
    source = Image.fromarray(image)
    if mode == "gaussian_area":
        radius = scale / 2.0
        filtered = source.filter(ImageFilter.GaussianBlur(radius=radius))
        details = {
            "downsampleFilter": "Gaussian prefilter followed by BOX area reduction",
            "gaussianSigmaHighResolutionPixels": radius,
            "areaReductionScale": scale,
        }
    elif mode == "box_area":
        # Pillow defines box width as 2*radius+1; this yields widths 2 and 4.
        radius = (scale - 1.0) / 2.0
        filtered = source.filter(ImageFilter.BoxBlur(radius=radius))
        details = {
            "downsampleFilter": "Box prefilter followed by BOX area reduction",
            "boxWidthHighResolutionPixels": scale,
            "boxRadiusHighResolutionPixels": radius,
            "areaReductionScale": scale,
        }
    else:
        source.close()
        raise RuntimeError(f"Unknown positive-kernel downsample mode: {mode}")
    resized = filtered.resize((out_size, out_size), resample=Image.Resampling.BOX)
    result = np.asarray(resized, dtype=np.uint8).copy()
    source.close()
    filtered.close()
    resized.close()
    return result, elapsed(start), details


def save_png(image: np.ndarray, path: Path, icc: bytes) -> float:
    start = time.perf_counter()
    pil_image = Image.fromarray(image)
    pil_image.save(
        path,
        format="PNG",
        icc_profile=icc,
        dpi=(300.0, 300.0),
        compress_level=6,
        optimize=False,
    )
    pil_image.close()
    return elapsed(start)


def render_method(points: np.memmap, assignment: np.memmap, method: Method,
                  output_dir: Path, side: int, interpolation: float,
                  out_size: int, stream_chunk: int, icc: bytes,
                  overwrite: bool) -> dict[str, Any]:
    path = output_dir / method.filename
    if path.exists() and not overwrite:
        raise RuntimeError(f"Output already exists (use --overwrite): {path}")
    render_size = out_size * method.scale
    pixel_count = render_size * render_size
    check_gpu_memory(pixel_count)
    log(
        f"METHOD {method.key} start: accumulation {render_size}x{render_size}, "
        f"output {out_size}x{out_size}"
    )
    total_start = time.perf_counter()
    allocation_start = time.perf_counter()
    accumulation = cuda.device_array((pixel_count, 4), dtype=np.uint32)
    accumulation[:] = 0
    cuda.synchronize()
    allocation_seconds = elapsed(allocation_start)

    accumulation_seconds = stream_points(
        points, assignment, accumulation, method, side, interpolation, out_size, stream_chunk
    )
    high_resolution, resolve_seconds, resolve_details = resolve_to_host(
        accumulation, method, render_size
    )
    del accumulation
    cuda.current_context().deallocations.clear()

    downsample_details: dict[str, Any] = {}
    if method.scale > 1 and method.downsample_mode == "lanczos":
        final_image, downsample_seconds = downsample_lanczos(high_resolution, out_size)
        downsample_details = {"downsampleFilter": "Lanczos"}
        del high_resolution
        gc.collect()
    elif method.scale > 1 and method.downsample_mode in {"gaussian_area", "box_area"}:
        final_image, downsample_seconds, downsample_details = downsample_filtered_area(
            high_resolution, out_size, method.scale, method.downsample_mode
        )
        del high_resolution
        gc.collect()
    elif method.scale > 1:
        raise RuntimeError(f"Supersampled method lacks a downsample mode: {method.key}")
    else:
        final_image = high_resolution
        downsample_seconds = 0.0

    png_seconds = save_png(final_image, path, icc)
    total_seconds = elapsed(total_start)
    channel_min = final_image.reshape(-1, 3).min(axis=0).astype(int).tolist()
    channel_max = final_image.reshape(-1, 3).max(axis=0).astype(int).tolist()
    channel_mean = final_image.reshape(-1, 3).mean(axis=0).tolist()
    del final_image
    gc.collect()
    cuda.current_context().deallocations.clear()

    result = {
        "method": method.key,
        "description": method.description,
        "output": str(path),
        "outputFilename": path.name,
        "outputWidth": out_size,
        "outputHeight": out_size,
        "accumulationWidth": render_size,
        "accumulationHeight": render_size,
        "supersampleScale": method.scale,
        "allocationAndClearSeconds": allocation_seconds,
        "accumulationSeconds": accumulation_seconds,
        "resolveAndTransferSeconds": resolve_seconds,
        "downsampleSeconds": downsample_seconds,
        "pngEncodingSeconds": png_seconds,
        "totalSeconds": total_seconds,
        "fileBytes": path.stat().st_size,
        "sha256": sha256(path),
        "channelMin": channel_min,
        "channelMax": channel_max,
        "channelMean": channel_mean,
        **downsample_details,
        **resolve_details,
    }
    log(
        f"METHOD {method.key} done in {total_seconds:.3f}s -> {path.name} "
        f"({path.stat().st_size / 2**20:.2f} MiB)"
    )
    return result


def validate_output(path: Path, out_size: int, expected_icc: bytes) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        if image.size != (out_size, out_size):
            raise RuntimeError(f"{path.name} has incorrect dimensions: {image.size}")
        if image.mode != "RGB":
            raise RuntimeError(f"{path.name} has incorrect mode: {image.mode}")
        embedded_icc = image.info.get("icc_profile")
        if embedded_icc != expected_icc:
            raise RuntimeError(f"{path.name} does not contain the expected sRGB ICC profile")
        dpi = image.info.get("dpi")
        if not dpi or any(abs(float(value) - 300.0) > 0.1 for value in dpi):
            raise RuntimeError(f"{path.name} does not report 300 ppi: {dpi}")
    return {"filename": path.name, "size": [out_size, out_size], "mode": "RGB", "dpi": dpi}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=DEFAULT_SIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--interp", type=float, default=DEFAULT_INTERP)
    parser.add_argument("--out-size", type=int, default=DEFAULT_OUT_SIZE)
    parser.add_argument("--stream-chunk", type=int, default=DEFAULT_STREAM_CHUNK)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_BY_KEY),
        default=[method.key for method in METHODS],
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.interp <= 1.0:
        raise RuntimeError("--interp must be in [0, 1]")
    if args.out_size <= 0 or args.stream_chunk <= 0 or args.side <= 1:
        raise RuntimeError("side, out size, and stream chunk must be positive")
    if not cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    n = args.side * args.side
    assignment_file = assignment_path(args.side, args.seed)
    assignment = validate_assignment(assignment_file, n)
    points, cache_metadata = prepare_position_cache(args.side, args.seed)
    output_dir = args.output_dir or (
        results_dir()
        / "renders"
        / (
            f"billion_shufflesnap_{args.side}_seed{args.seed}_"
            f"interp{int(round(args.interp * 100))}_aa_{args.out_size}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    icc, icc_path = load_icc()
    device = cuda.get_current_device()
    warm_start = time.perf_counter()
    warm_cuda_kernels()
    warm_seconds = elapsed(warm_start)
    log(f"CUDA ready: {device.name.decode()} compute capability {device.compute_capability}")
    log(f"CUDA JIT warmup {warm_seconds:.3f}s (excluded from per-image timings)")

    batch_start = time.perf_counter()
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "createdAtUnix": time.time(),
        "source": {
            "assignment": str(assignment_file),
            "assignmentBytes": assignment_file.stat().st_size,
            "positions": cache_metadata,
            "side": args.side,
            "pointCount": n,
            "seed": args.seed,
            "interpolation": args.interp,
            "margin": MARGIN,
        },
        "output": {
            "directory": str(output_dir),
            "width": args.out_size,
            "height": args.out_size,
            "mode": "RGB",
            "ppi": 300,
            "iccProfile": icc_path,
        },
        "implementation": {
            "gpu": device.name.decode(),
            "computeCapability": list(device.compute_capability),
            "numba": __import__("numba").__version__,
            "numpy": np.__version__,
            "pillow": __import__("PIL").__version__,
            "integerAccumulation": "uint32",
            "weightedAccumulationScale": WEIGHT_SCALE,
            "streamChunkPoints": args.stream_chunk,
            "cudaJitWarmupSecondsExcluded": warm_seconds,
        },
        "variants": [],
    }
    manifest_path = output_dir / "timings.json"
    atomic_write_json(manifest_path, manifest)

    for key in args.methods:
        result = render_method(
            points,
            assignment,
            METHOD_BY_KEY[key],
            output_dir,
            args.side,
            args.interp,
            args.out_size,
            args.stream_chunk,
            icc,
            args.overwrite,
        )
        manifest["variants"].append(result)
        manifest["batchElapsedSeconds"] = elapsed(batch_start)
        atomic_write_json(manifest_path, manifest)

    validations = [
        validate_output(output_dir / item["outputFilename"], args.out_size, icc)
        for item in manifest["variants"]
    ]
    manifest["validation"] = validations
    manifest["batchElapsedSeconds"] = elapsed(batch_start)
    atomic_write_json(manifest_path, manifest)
    log(f"VALIDATED {len(validations)} RGB PNGs at {args.out_size}x{args.out_size}, 300 ppi, sRGB")
    log(f"TIMINGS {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
