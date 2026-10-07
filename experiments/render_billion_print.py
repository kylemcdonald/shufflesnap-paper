"""Print-quality render of the billion-point assignment (supersampled, tiled on the GPU).

Port of the earlier work's print renderer (../../shufflesnap-paper/benchmarks/
render_billion_4x_box_area_tiled.py, with its helpers from render_billion_antialias.py).
The picture is the one of render_billion.py: every point drawn at
(1 - t) * source + t * assigned cell center in unit-square coordinates, colored by its
source position on the Lab (a, b) plane at L = 74, colors averaged where points overlap,
empty space white.  The technique:

1. Conceptual canvas ``scale`` times the output size (4x: 76800^2 for a 19200^2 print).
   Each point adds its 24-bit fixed-point color and a count of 1 to its nearest pixel
   (uint64 atomics, so the result does not depend on thread order).
2. Color sums and counts are convolved with a smoothstep disk ``scale`` pixels wide,
   then divided (float32): the linear equivalent of splatting one disk per point.
3. A ``scale``-wide box prefilter (half-weight end taps, as Pillow's fractional box) and
   an integer ``scale`` x ``scale`` area reduction give the output pixels, rounded once
   to 8 or 16 bits.  Both are one separable 2*scale-tap kernel, applied on the GPU as a
   horizontal and a vertical pass, so only output pixels return to the host.

The canvas is processed as full-width horizontal strips with enough halo for the disk
and the box, so it never has to fit in GPU memory; strips match a full-canvas render
exactly.  The earlier renderer's 8-bit path accumulated rounded 8-bit colors and
filtered in uint8; here 8-bit output takes the high-precision path and rounds only at
the end.  The PNG carries the standard sRGB ICC profile and the requested ppi.

``--color adobe-slice`` colors each point instead by the Coons patch of the CIELAB
lightness slice L* = ``--slice-lightness`` of the Adobe RGB (1998) gamut, evaluated at
its source position.  ``--slice-method lab`` (default) is gamut_slice.coons_luts
(hue-pinned corners, Coons blend in Lab; the patch of gamut_slice.py's square PNG) with
the system Adobe RGB (1998) profile; ``color-cube`` builds the patch exactly as
color-cube's scripts/lightness_slice.py (copied unchanged to experiments/) and embeds
that script's ICC profile.  Those colors are linear, so
averaging, the disk and the box/area reduction all happen in linear light; the Adobe RGB
gamma (563/256) is applied once at the end.

Source points are regenerated once (billion.generate_meander, bit-identical to the
earlier instance) and cached as float32 next to the assignments.

    PYTHONPATH=../shufflesnap/src python experiments/render_billion_print.py \\
        --out-size 19200 --dpi 1200 --bit-depth 8
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda
from PIL import Image

from billion import CHUNK, OUT, generate_meander
import lightness_slice
from gamut_slice import ADOBE_GAMMA, color_cube_luts, coons_luts, lab_to_rgb_matrix

DEFAULT_SIDE = 31_623
DEFAULT_SEED = 0
DEFAULT_INTERP = 0.75
DEFAULT_ASSIGNMENT = OUT / "billion_assignment_31623_seed0_w16r1.npy"
DEFAULT_OUT_SIZE = 19_200
DEFAULT_SCALE = 4
DEFAULT_DPI = 1200
DEFAULT_TILE_OUTPUT_ROWS = 1_440
DEFAULT_STREAM_CHUNK = 20_000_000
THREADS = 256
# Eight resolved high-resolution rows cover the support of both the 4 px and
# 8 px box prefilters exposed by this renderer, with extra seam safety.
FILTER_HALO = 8
POINT_COLOR_MAX = 16_777_215  # 24-bit fixed point; sums fit uint64 for a billion points.
ICC_CANDIDATES = {
    "srgb": (Path("/usr/share/color/icc/colord/sRGB.icc"),
             Path("/usr/share/color/icc/ghostscript/srgb.icc")),
    "adobe": (Path("/usr/share/color/icc/colord/AdobeRGB1998.icc"),),
}
ICC_NAMES = {"srgb": b"sRGB", "adobe": b"Adobe RGB (1998)"}
COLOR_LAB74 = 0
COLOR_ADOBE_SLICE = 1  # Lab Coons patch (gamut_slice.coons_luts)
COLOR_ADOBE_SLICE_RGB = 2  # linear-RGB Coons patch (color-cube lightness_slice.py)


def log(message: str) -> None:
    print(message, flush=True)


def elapsed(start: float) -> float:
    return time.perf_counter() - start


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_icc(space: str) -> tuple[bytes, str]:
    for path in ICC_CANDIDATES[space]:
        if path.is_file():
            return path.read_bytes(), str(path)
    raise RuntimeError(f"No standard {ICC_NAMES[space].decode()} ICC profile was found")


def validate_assignment(path: Path, n: int) -> np.memmap:
    if not path.is_file():
        raise RuntimeError(f"Assignment is missing: {path}")
    assignment = np.load(path, mmap_mode="r")
    if assignment.shape != (n,) or assignment.dtype != np.int32:
        raise RuntimeError(
            f"Assignment must have shape {(n,)} and dtype int32; "
            f"found {assignment.shape} {assignment.dtype}"
        )
    return assignment


def position_cache_path(side: int, seed: int) -> Path:
    return OUT / f"billion_positions_{side}_seed{seed}_float32.npy"


def prepare_position_cache(side: int, seed: int) -> tuple[np.memmap, dict[str, Any]]:
    n = side * side
    path = position_cache_path(side, seed)
    if path.is_file():
        points = np.load(path, mmap_mode="r")
        if points.shape != (n, 2) or points.dtype != np.float32:
            raise RuntimeError(f"Position cache has shape {points.shape} {points.dtype}")
        log(f"POSITION CACHE reused {path} ({points.nbytes / 2**30:.2f} GiB)")
        return points, {"created": False, "path": str(path)}

    required = n * 2 * np.dtype(np.float32).itemsize
    if shutil.disk_usage(path.parent).free < required + 8 * 2**30:
        raise RuntimeError(f"Position cache needs {required / 2**30:.1f} GiB plus 8 GiB reserve")
    temporary = path.with_name(path.stem + ".tmp.npy")
    start = time.perf_counter()
    log("POSITION CACHE missing; regenerating the source points once")
    points64 = generate_meander(n, seed=seed)
    cached = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(n, 2))
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        cached[i0:i1] = points64[i0:i1]
    cached.flush()
    del cached, points64
    gc.collect()
    temporary.replace(path)
    seconds = elapsed(start)
    log(f"POSITION CACHE wrote {path} in {seconds:.1f}s")
    return np.load(path, mmap_mode="r"), {"created": True, "seconds": seconds, "path": str(path)}


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
    """Continuous sRGB channel values on the 0..255 scale (render_billion.lab_to_srgb)."""
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
def _lut_at(luts, curve: int, t: float, channel: int) -> float:
    samples = luts.shape[1] - 1
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    f = t * samples
    i = int(f)
    if i > samples - 1:
        i = samples - 1
    w = f - i
    return (1.0 - w) * luts[curve, i, channel] + w * luts[curve, i + 1, channel]


@cuda.jit(device=True, inline=True)
def _patch(px: float, py: float, luts, corners, c: int) -> float:
    return ((1.0 - py) * _lut_at(luts, 0, px, c) + py * _lut_at(luts, 1, px, c)
            + (1.0 - px) * _lut_at(luts, 2, py, c) + px * _lut_at(luts, 3, py, c)
            - ((1.0 - px) * (1.0 - py) * corners[0, c] + px * (1.0 - py) * corners[1, c]
               + px * py * corners[2, c] + (1.0 - px) * py * corners[3, c]))


@cuda.jit(device=True, inline=True)
def _unit(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@cuda.jit(device=True, inline=True)
def _slice_rgb(px: float, py: float, luts, corners) -> tuple[float, float, float]:
    """Linear Adobe RGB of the Lab Coons patch at (px, py) (gamut_slice.evaluate).
    ``corners`` rows 0-3 are the corner Lab values, rows 4-6 gamut_slice.lab_to_rgb_matrix."""
    px = _unit(px)
    py = _unit(py)
    fy = (_patch(px, py, luts, corners, 0) + 16.0) / 116.0
    fx = _inv_lab_f(fy + _patch(px, py, luts, corners, 1) / 500.0)
    fz = _inv_lab_f(fy - _patch(px, py, luts, corners, 2) / 200.0)
    fy = _inv_lab_f(fy)
    return (_unit(corners[4, 0] * fx + corners[4, 1] * fy + corners[4, 2] * fz),
            _unit(corners[5, 0] * fx + corners[5, 1] * fy + corners[5, 2] * fz),
            _unit(corners[6, 0] * fx + corners[6, 1] * fy + corners[6, 2] * fz))


@cuda.jit(device=True, inline=True)
def _render_xy(px: float, py: float, cell: int, side: int,
               interpolation: float) -> tuple[float, float]:
    row = cell // side
    col = cell - row * side
    grid_scale = 1.0 / (side - 1)
    target_x = col * grid_scale
    target_y = row * grid_scale
    source_weight = 1.0 - interpolation
    return (
        source_weight * px + interpolation * target_x,
        source_weight * py + interpolation * target_y,
    )


@cuda.jit
def scatter_nearest_strip(points, cells, count, accumulation, global_render_size,
                          accumulation_x_start, accumulation_width,
                          accumulation_y_start, accumulation_height, side, interpolation,
                          color_mode, luts, corners):
    index = cuda.grid(1)
    if index >= count:
        return
    px = points[index, 0]
    py = points[index, 1]
    x, y = _render_xy(px, py, cells[index], side, interpolation)
    ix = int(x * global_render_size)
    iy = int(y * global_render_size)
    if ix < 0:
        ix = 0
    elif ix >= global_render_size:
        ix = global_render_size - 1
    if iy < 0:
        iy = 0
    elif iy >= global_render_size:
        iy = global_render_size - 1
    local_y = iy - accumulation_y_start
    if local_y < 0 or local_y >= accumulation_height:
        return
    local_x = ix - accumulation_x_start
    if local_x < 0 or local_x >= accumulation_width:
        return
    flat = local_y * accumulation_width + local_x
    if color_mode == COLOR_ADOBE_SLICE_RGB:
        sx = _unit(px)
        sy = _unit(py)
        rr = _unit(_patch(sx, sy, luts, corners, 0))
        gg = _unit(_patch(sx, sy, luts, corners, 1))
        bb = _unit(_patch(sx, sy, luts, corners, 2))
        rr *= 255.0
        gg *= 255.0
        bb *= 255.0
    elif color_mode == COLOR_ADOBE_SLICE:
        rr, gg, bb = _slice_rgb(px, py, luts, corners)
        rr *= 255.0
        gg *= 255.0
        bb *= 255.0
    else:
        rr, gg, bb = _point_rgb_float(px, py)
    # Wide integer atomics keep accumulation independent of thread order.
    cuda.atomic.add(accumulation, (flat, 0), int(rr * (POINT_COLOR_MAX / 255.0) + 0.5))
    cuda.atomic.add(accumulation, (flat, 1), int(gg * (POINT_COLOR_MAX / 255.0) + 0.5))
    cuda.atomic.add(accumulation, (flat, 2), int(bb * (POINT_COLOR_MAX / 255.0) + 0.5))
    cuda.atomic.add(accumulation, (flat, 3), 1)


@cuda.jit
def resolve_disk_float(accumulation, image, width, accumulation_height, resolved_width,
                       resolved_x_offset, resolved_offset, kernel, kernel_width):
    """Disk-filter the accumulation; ``image`` is resolved_height x resolved_width, offset
    by (resolved_offset, resolved_x_offset) inside the accumulation (width ``width``)."""
    index = cuda.grid(1)
    if index >= image.shape[0]:
        return
    source_y = index // resolved_width + resolved_offset
    x = index % resolved_width + resolved_x_offset
    radius = kernel_width // 2
    red, green, blue, weight = 0.0, 0.0, 0.0, 0.0
    for ky in range(kernel_width):
        yy = source_y + ky - radius
        if yy < 0 or yy >= accumulation_height:
            continue
        for kx in range(kernel_width):
            xx = x + kx - radius
            if xx < 0 or xx >= width:
                continue
            w = float(kernel[ky * kernel_width + kx])
            source = yy * width + xx
            weight += float(accumulation[source, 3]) * w
            red += float(accumulation[source, 0]) * w
            green += float(accumulation[source, 1]) * w
            blue += float(accumulation[source, 2]) * w
    if weight == 0.0:
        image[index, 0] = 1.0
        image[index, 1] = 1.0
        image[index, 2] = 1.0
    else:
        divisor = weight * POINT_COLOR_MAX
        image[index, 0] = red / divisor
        image[index, 1] = green / divisor
        image[index, 2] = blue / divisor


def reduce_kernel(scale: int) -> np.ndarray:
    """Box prefilter followed by a ``scale``-wide area mean, as one 1D kernel.

    The box matches Pillow's fractional-radius support: ``scale + 1`` taps with
    half-weight ends.  Tap ``d`` of the result reads source offset ``d - scale // 2``
    from the first source pixel of an output pixel; edges replicate.
    """
    box = np.ones(scale + 1, dtype=np.float64) / scale
    box[0] *= 0.5
    box[-1] *= 0.5
    return np.convolve(np.ones(scale) / scale, box).astype(np.float32)


@cuda.jit
def reduce_rows(image, width, height, out_width, crop_left, kernel, scale, partial):
    """Horizontal pass: partial[y, ox] = sum_d kernel[d] * image[y, crop_left + ox*scale + d - scale//2]."""
    index = cuda.grid(1)
    if index >= height * out_width:
        return
    y = index // out_width
    ox = index - y * out_width
    x0 = crop_left + ox * scale - scale // 2
    red, green, blue = 0.0, 0.0, 0.0
    for d in range(kernel.shape[0]):
        x = x0 + d
        if x < 0:
            x = 0
        elif x >= width:
            x = width - 1
        w = kernel[d]
        source = y * width + x
        red += w * image[source, 0]
        green += w * image[source, 1]
        blue += w * image[source, 2]
    partial[index, 0] = red
    partial[index, 1] = green
    partial[index, 2] = blue


@cuda.jit
def reduce_columns(partial, height, out_width, out_rows, crop_top, kernel, scale, top,
                   inverse_gamma, output):
    """Vertical pass and quantization: rows ``crop_top + oy*scale + d - scale//2``, clamped.
    A positive ``inverse_gamma`` encodes linear values as value ** inverse_gamma first."""
    index = cuda.grid(1)
    if index >= out_rows * out_width:
        return
    oy = index // out_width
    ox = index - oy * out_width
    y0 = crop_top + oy * scale - scale // 2
    for c in range(3):
        value = 0.0
        for d in range(kernel.shape[0]):
            y = y0 + d
            if y < 0:
                y = 0
            elif y >= height:
                y = height - 1
            value += kernel[d] * partial[y * out_width + ox, c]
        if value < 0.0:
            value = 0.0
        elif value > 1.0:
            value = 1.0
        if inverse_gamma > 0.0:
            value = value ** inverse_gamma
        output[oy, ox, c] = math.floor(value * top + 0.5)


def write_rgb_png(path, pixels, icc, dpi, icc_name=b"sRGB"):
    """Stream RGB8/RGB16 PNG rows with Sub filtering, ICC profile and the given ppi.

    PNG uses big-endian samples and filters bytes per pixel (3 or 6 bytes per RGB pixel).
    Specification: https://www.w3.org/TR/png-3/
    """
    if pixels.dtype not in (np.uint8, np.uint16) or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Expected H x W x 3 uint8 or uint16 pixels")
    depth = 8 * pixels.dtype.itemsize
    bpp = 3 * pixels.dtype.itemsize
    with Path(path).open("wb") as handle:
        def chunk(kind, data):
            handle.write(struct.pack(">I", len(data)))
            handle.write(kind)
            handle.write(data)
            handle.write(struct.pack(">I", zlib.crc32(data, zlib.crc32(kind))))

        handle.write(b"\x89PNG\r\n\x1a\n")
        height, width = pixels.shape[:2]
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, 2, 0, 0, 0))
        chunk(b"iCCP", icc_name + b"\0\0" + zlib.compress(icc))
        ppm = round(dpi / 0.0254)
        chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))
        compressor = zlib.compressobj(6)
        big_endian = pixels.dtype.newbyteorder(">")
        for row in pixels:
            raw = row.astype(big_endian).view(np.uint8).reshape(-1)
            filtered = raw.copy()
            filtered[bpp:] = raw[bpp:] - raw[:-bpp]
            data = compressor.compress(b"\x01" + filtered.tobytes())
            if data:
                chunk(b"IDAT", data)
        chunk(b"IDAT", compressor.flush())
        chunk(b"IEND", b"")


def warm_kernels(disk: np.ndarray, color) -> float:
    start = time.perf_counter()
    accumulation = cuda.to_device(np.zeros((32, 4), dtype=np.uint64))
    scatter_nearest_strip[1, 32](
        cuda.to_device(np.array([[0.25, 0.75], [0.75, 0.25]], dtype=np.float32)),
        cuda.to_device(np.array([0, 3], dtype=np.int32)),
        2, accumulation, 4, 0, 4, 0, 8, 2, np.float32(DEFAULT_INTERP), *color,
    )
    image = cuda.device_array((16, 3), dtype=np.float32)
    resolve_disk_float[1, 32](accumulation, image, 4, 8, 4, 0, 2,
                              cuda.to_device(disk.ravel()), disk.shape[0])
    cuda.synchronize()
    del accumulation, image
    gc.collect()
    cuda.current_context().deallocations.clear()
    return elapsed(start)


def stream_strip(points, assignment, accumulation, global_render_size, accumulation_x_start,
                 accumulation_width, accumulation_y_start, accumulation_height, side,
                 interpolation, stream_chunk, strip_number, strip_count, color) -> float:
    n = points.shape[0]
    device_points = cuda.device_array((stream_chunk, 2), dtype=np.float32)
    device_cells = cuda.device_array(stream_chunk, dtype=np.int32)
    start = time.perf_counter()
    progress_interval = max(1, math.ceil(n / stream_chunk / 5))
    chunk_number = 0
    for i0 in range(0, n, stream_chunk):
        i1 = min(i0 + stream_chunk, n)
        count = i1 - i0
        device_points[:count].copy_to_device(points[i0:i1])
        device_cells[:count].copy_to_device(assignment[i0:i1])
        blocks = (count + THREADS - 1) // THREADS
        scatter_nearest_strip[blocks, THREADS](
            device_points, device_cells, count, accumulation, global_render_size,
            accumulation_x_start, accumulation_width, accumulation_y_start,
            accumulation_height, side, np.float32(interpolation), *color,
        )
        chunk_number += 1
        if chunk_number % progress_interval == 0 or i1 == n:
            cuda.synchronize()
            log(f"  strip {strip_number}/{strip_count}: {i1:,}/{n:,} points "
                f"({100.0 * i1 / n:5.1f}%) in {elapsed(start):.2f}s")
    cuda.synchronize()
    del device_points, device_cells
    return elapsed(start)


def render_strip(points, assignment, final_image, output_y_start, output_y_end, out_size,
                 scale, side, interpolation, stream_chunk, disk, strip_number,
                 strip_count, output_x_start=0, output_x_end=None,
                 image_row0=0, color=None, inverse_gamma=0.0) -> dict[str, Any]:
    """Render rows [output_y_start, output_y_end) and columns [output_x_start, output_x_end)
    of the out_size^2 image.  ``final_image`` holds just those columns, and its first row
    is output row ``image_row0``."""
    if output_x_end is None:
        output_x_end = out_size
    global_render_size = out_size * scale
    out_width = output_x_end - output_x_start
    core_start = output_y_start * scale
    core_end = output_y_end * scale
    resolved_start = max(0, core_start - FILTER_HALO)
    resolved_end = min(global_render_size, core_end + FILTER_HALO)
    disk_radius = disk.shape[0] // 2
    accumulation_start = max(0, resolved_start - disk_radius)
    accumulation_end = min(global_render_size, resolved_end + disk_radius)
    accumulation_height = accumulation_end - accumulation_start
    resolved_height = resolved_end - resolved_start
    # the same halos in x, so a crop matches the corresponding part of the full image
    core_left = output_x_start * scale
    resolved_left = max(0, core_left - FILTER_HALO)
    resolved_right = min(global_render_size, output_x_end * scale + FILTER_HALO)
    accumulation_left = max(0, resolved_left - disk_radius)
    accumulation_right = min(global_render_size, resolved_right + disk_radius)
    accumulation_width = accumulation_right - accumulation_left
    resolved_width = resolved_right - resolved_left
    accumulation_pixels = accumulation_width * accumulation_height
    resolved_pixels = resolved_width * resolved_height

    free, total = cuda.current_context().get_memory_info()
    output_rows = output_y_end - output_y_start
    # the accumulation is freed before the reduction buffers are allocated
    estimated_gpu_bytes = max(accumulation_pixels * 32 + stream_chunk * 12,
                              accumulation_pixels * 32 + resolved_pixels * 12,
                              resolved_pixels * 12 + resolved_height * out_width * 12
                              + output_rows * out_width * 3 * final_image.dtype.itemsize)
    if estimated_gpu_bytes + 2 * 2**30 > free:
        raise RuntimeError(
            f"Strip {strip_number} needs approximately {estimated_gpu_bytes / 2**30:.2f} GiB "
            f"plus a 2 GiB reserve, but only {free / 2**30:.2f} GiB is free"
        )
    log(f"STRIP {strip_number}/{strip_count}: output rows {output_y_start}:{output_y_end}, "
        f"GPU estimate {estimated_gpu_bytes / 2**30:.2f} GiB")
    strip_start = time.perf_counter()
    accumulation = cuda.device_array((accumulation_pixels, 4), dtype=np.uint64)
    accumulation[:] = 0
    accumulation_seconds = stream_strip(
        points, assignment, accumulation, global_render_size, accumulation_left,
        accumulation_width, accumulation_start, accumulation_height, side, interpolation,
        stream_chunk, strip_number, strip_count, color,
    )

    resolve_start = time.perf_counter()
    device_disk = cuda.to_device(disk.ravel())
    device_image = cuda.device_array((resolved_pixels, 3), dtype=np.float32)
    blocks = (resolved_pixels + THREADS - 1) // THREADS
    resolve_disk_float[blocks, THREADS](
        accumulation, device_image, accumulation_width, accumulation_height, resolved_width,
        resolved_left - accumulation_left, resolved_start - accumulation_start, device_disk,
        disk.shape[0],
    )
    cuda.synchronize()
    resolve_seconds = elapsed(resolve_start)
    del accumulation, device_disk
    cuda.current_context().deallocations.clear()

    filter_start = time.perf_counter()
    kernel = cuda.to_device(reduce_kernel(scale))
    partial = cuda.device_array((resolved_height * out_width, 3), dtype=np.float32)
    count = resolved_height * out_width
    reduce_rows[(count + THREADS - 1) // THREADS, THREADS](
        device_image, resolved_width, resolved_height, out_width, core_left - resolved_left,
        kernel, scale, partial)
    del device_image
    device_output = cuda.device_array((output_rows, out_width, 3), dtype=final_image.dtype)
    count = output_rows * out_width
    reduce_columns[(count + THREADS - 1) // THREADS, THREADS](
        partial, resolved_height, out_width, output_rows, core_start - resolved_start, kernel,
        scale, np.float32(np.iinfo(final_image.dtype).max), np.float32(inverse_gamma),
        device_output)
    reduced = device_output.copy_to_host()
    del partial, device_output, kernel
    cuda.current_context().deallocations.clear()
    final_image[output_y_start - image_row0:output_y_end - image_row0] = reduced
    final_image.flush()
    flat = reduced.reshape(-1, 3)
    result = {
        "strip": strip_number,
        "outputRows": [output_y_start, output_y_end],
        "outputColumns": [output_x_start, output_x_end],
        "highResolutionAccumulationRows": [accumulation_start, accumulation_end],
        "estimatedGpuBytes": estimated_gpu_bytes,
        "accumulationSeconds": accumulation_seconds,
        "resolveSeconds": resolve_seconds,
        "filterDownsampleTransferSeconds": elapsed(filter_start),
        "totalSeconds": elapsed(strip_start),
        "channelMin": flat.min(axis=0).astype(int).tolist(),
        "channelMax": flat.max(axis=0).astype(int).tolist(),
        "channelSum": flat.sum(axis=0, dtype=np.uint64).astype(int).tolist(),
    }
    del reduced, flat
    gc.collect()
    log(f"STRIP {strip_number}/{strip_count} done in {result['totalSeconds']:.1f}s")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--side", type=int, default=DEFAULT_SIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--assignment", type=Path, default=DEFAULT_ASSIGNMENT)
    parser.add_argument("--interp", type=float, default=DEFAULT_INTERP)
    parser.add_argument("--color", choices=("lab74", "adobe-slice"), default="adobe-slice",
                        help="lab74: Lab (a, b) plane at L = 74 in sRGB; adobe-slice: Coons "
                             "patch of an Adobe RGB (1998) lightness slice")
    parser.add_argument("--slice-lightness", type=float, default=67.5)
    parser.add_argument("--slice-method", choices=("lab", "color-cube"), default="lab",
                        help="lab: gamut_slice's Lab patch; color-cube: lightness_slice.py's linear-RGB patch")
    parser.add_argument("--bit-depth", type=int, choices=(8, 16), default=8)
    parser.add_argument("--out-size", type=int, default=DEFAULT_OUT_SIZE)
    parser.add_argument("--dpi", type=float, default=DEFAULT_DPI)
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, choices=(4, 8))
    parser.add_argument("--disk-diameter", type=float,
                        help="Smoothstep disk diameter in high-resolution pixels (default: --scale)")
    parser.add_argument("--tile-output-rows", type=int, default=DEFAULT_TILE_OUTPUT_ROWS)
    parser.add_argument("--stream-chunk", type=int, default=DEFAULT_STREAM_CHUNK)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="render only output pixels [X0, X1) x [Y0, Y1) of the out-size image")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.interp <= 1.0:
        raise RuntimeError("--interp must be in [0, 1]")
    disk_diameter = float(args.scale) if args.disk_diameter is None else args.disk_diameter
    if not cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    Image.MAX_IMAGE_PIXELS = None  # our own output, larger than Pillow's bomb threshold
    dtype = np.uint8 if args.bit_depth == 8 else np.uint16

    n = args.side * args.side
    assignment = validate_assignment(args.assignment, n)
    points, cache_metadata = prepare_position_cache(args.side, args.seed)
    inches = args.out_size / args.dpi
    x0, y0, x1, y1 = args.crop or (0, 0, args.out_size, args.out_size)
    if not (0 <= x0 < x1 <= args.out_size and 0 <= y0 < y1 <= args.out_size):
        raise RuntimeError("--crop must lie inside the image")
    width, height = x1 - x0, y1 - y0
    crop_tag = f"_crop{x0}-{y0}-{x1}-{y1}" if args.crop else ""
    adobe = args.color == "adobe-slice"
    color_cube = args.slice_method == "color-cube"
    color_tag = (f"_adobeL{args.slice_lightness:g}{'cc' if color_cube else ''}" if adobe else "")
    output_path = (args.output or OUT / "renders" / (
        f"billion_{args.assignment.stem.removeprefix('billion_assignment_')}"
        f"_interp{round(100 * args.interp)}{color_tag}_{args.out_size}{crop_tag}"
        f"_disk{disk_diameter:g}"
        f"_{args.dpi:g}dpi_{args.bit_depth}bit.png"
    )).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists (use --overwrite): {output_path}")

    icc_space = "adobe" if adobe else "srgb"
    icc, icc_path = load_icc(icc_space)
    icc_name = ICC_NAMES[icc_space]
    if adobe and color_cube:
        luts, corners, polygon, sides = color_cube_luts(args.slice_lightness)
        color = (COLOR_ADOBE_SLICE_RGB, cuda.to_device(luts),
                 cuda.to_device(np.concatenate([corners, np.zeros((3, 3), np.float32)])))
        icc = lightness_slice.icc_profile("adobe")
        icc_name = lightness_slice.NAMES["adobe"].encode("latin-1")
        icc_path = "color-cube lightness_slice.icc_profile('adobe')"
        inverse_gamma = 1.0 / ADOBE_GAMMA
        log(f"COLOR Adobe RGB (1998) L* = {args.slice_lightness:g} slice: {sides}-gon, "
            f"color-cube linear-RGB Coons patch, linear averaging, gamma {ADOBE_GAMMA:g}")
    elif adobe:
        luts, corners, polygon = coons_luts(args.slice_lightness)
        corners = np.concatenate([corners, lab_to_rgb_matrix()]).astype(np.float32)
        color = (COLOR_ADOBE_SLICE, cuda.to_device(luts), cuda.to_device(corners))
        inverse_gamma = 1.0 / ADOBE_GAMMA
        log(f"COLOR Adobe RGB (1998) L* = {args.slice_lightness:g} slice: {len(polygon)}-gon, "
            f"Lab Coons patch, linear averaging, gamma {ADOBE_GAMMA:g}")
    else:
        color = (COLOR_LAB74, cuda.to_device(np.zeros((4, 2, 3), np.float32)),
                 cuda.to_device(np.zeros((7, 3), np.float32)))
        inverse_gamma = 0.0
    disk = smoothstep_kernel(disk_diameter)
    warm_seconds = warm_kernels(disk, color)
    device = cuda.get_current_device()
    log(f"CUDA ready: {device.name.decode()}; JIT warmup {warm_seconds:.2f}s (not timed)")
    log(f"OUTPUT {args.out_size}^2 px = {inches:g} in at {args.dpi:g} ppi, {args.bit_depth}-bit, "
        f"from a {args.out_size * args.scale}^2 canvas; disk {disk_diameter:g} canvas px = "
        f"{disk_diameter / args.scale:g} output px"
        + (f"; crop x {x0}:{x1}, y {y0}:{y1} ({width}x{height} px = "
           f"{width / args.dpi:g}x{height / args.dpi:g} in)" if args.crop else ""))

    temporary_pixels = output_path.with_suffix(output_path.suffix + ".pixels.tmp")
    temporary_png = output_path.with_suffix(output_path.suffix + ".tmp")
    timing_path = output_path.with_suffix(".json")
    generation_start = time.perf_counter()
    final_image = np.memmap(temporary_pixels, mode="w+", dtype=dtype,
                            shape=(height, width, 3))
    strip_count = math.ceil(height / args.tile_output_rows)
    strips = []
    for strip_index, ys in enumerate(range(y0, y1, args.tile_output_rows), start=1):
        strips.append(render_strip(
            points, assignment, final_image, ys, min(y1, ys + args.tile_output_rows),
            args.out_size, args.scale, args.side, args.interp, args.stream_chunk, disk,
            strip_index, strip_count, x0, x1, y0, color, inverse_gamma,
        ))

    encode_start = time.perf_counter()
    final_image.flush()
    write_rgb_png(temporary_png, final_image, icc, args.dpi, icc_name)
    png_encoding_seconds = elapsed(encode_start)
    temporary_png.replace(output_path)
    generation_seconds = elapsed(generation_start)
    del final_image
    gc.collect()
    temporary_pixels.unlink()

    with Image.open(output_path) as image:
        image.load()
        if image.size != (width, height):
            raise RuntimeError(f"Unexpected PNG size {image.size}")
        if image.info.get("icc_profile") != icc:
            raise RuntimeError(f"PNG does not carry the {icc_space} ICC profile")
        dpi = image.info.get("dpi")
        if not dpi or any(abs(float(v) - args.dpi) > 0.1 for v in dpi):
            raise RuntimeError(f"PNG reports {dpi} ppi")

    pixel_count = width * height
    result = {
        "source": {"assignment": str(args.assignment.resolve()), "positions": cache_metadata,
                   "side": args.side, "pointCount": n, "seed": args.seed,
                   "interpolation": args.interp},
        "color": ({"mode": "adobe-slice", "method": args.slice_method,
                   "lightness": args.slice_lightness,
                   "space": "Adobe RGB (1998)", "gamma": ADOBE_GAMMA,
                   "averaging": "linear"} if adobe else
                  {"mode": "lab74", "space": "sRGB", "averaging": "gamma-encoded"}),
        "output": {
            "path": str(output_path), "width": width, "height": height,
            "fullImageSize": args.out_size, "crop": [x0, y0, x1, y1],
            "bitsPerChannel": args.bit_depth, "ppi": args.dpi, "fullImageInches": inches,
            "iccProfile": icc_path, "fileBytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
            "channelMin": np.min([s["channelMin"] for s in strips], axis=0).tolist(),
            "channelMax": np.max([s["channelMax"] for s in strips], axis=0).tolist(),
            "channelMean": (np.sum([s["channelSum"] for s in strips], axis=0) / pixel_count).tolist(),
        },
        "method": {
            "name": f"disk_{args.scale}x_box_area_tiled",
            "conceptualHighResolutionSize": args.out_size * args.scale,
            "diskDiameterHighResolutionPixels": disk_diameter,
            "boxWidthHighResolutionPixels": args.scale,
            "tileOutputRows": args.tile_output_rows,
            "stripCount": strip_count,
            "pointColor": (f"linear Adobe RGB from the {args.slice_method} Coons patch of the slice"
                           if adobe else
                           "continuous Lab-derived sRGB") + ", 24-bit fixed point, uint64 sums",
            "filterPrecision": "float32 resolve and box/area reduction on the GPU, rounded once",
        },
        "implementation": {"gpu": device.name.decode(),
                           "numba": __import__("numba").__version__, "numpy": np.__version__},
        "timing": {"totalGenerationSeconds": generation_seconds,
                   "pngEncodingSeconds": png_encoding_seconds, "strips": strips},
    }
    atomic_write_json(timing_path, result)
    log(f"DONE {output_path} in {generation_seconds:.1f}s "
        f"({output_path.stat().st_size / 2**20:.1f} MiB); record {timing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
