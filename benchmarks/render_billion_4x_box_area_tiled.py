"""Render the billion-point image supersampled using tiled GPU accumulation.

The conceptual high-resolution canvas is four times the requested output size.
It is processed as overlapping, full-width horizontal strips so an oversized
square accumulator never needs to fit in GPU memory. Each strip includes enough
halo for the smoothstep disk and Pillow box prefilter. Strip boundaries are
aligned to output rows before the final integer-ratio BOX area reduction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda
from PIL import Image, ImageFilter

from render_billion_antialias import (
    DEFAULT_INTERP,
    DEFAULT_SEED,
    DEFAULT_SIDE,
    THREADS,
    _point_rgb,
    _point_rgb_float,
    _render_xy,
    assignment_path,
    load_icc,
    prepare_position_cache,
    results_dir,
    smoothstep_kernel,
    validate_assignment,
)


DEFAULT_OUT_SIZE = 14_400
DEFAULT_SCALE = 4
DEFAULT_TILE_OUTPUT_ROWS = 2_880
DEFAULT_STREAM_CHUNK = 20_000_000
# Eight resolved high-resolution rows cover the support of both the 4 px and
# 8 px box prefilters exposed by this renderer, with extra seam safety.
FILTER_HALO = 8


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


@cuda.jit(device=True, inline=True)
def hash_u32(value: np.uint32) -> np.uint32:
    """Deterministic integer mixer suitable for counter-based GPU noise."""
    value ^= value >> 16
    value = np.uint32(value * np.uint32(0x7FEB352D))
    value ^= value >> 15
    value = np.uint32(value * np.uint32(0x846CA68B))
    value ^= value >> 16
    return value


@cuda.jit(device=True, inline=True)
def channel_noise(global_index: int, channel_salt: int, seed: int) -> float:
    counter = np.uint32(global_index) ^ np.uint32(channel_salt) ^ np.uint32(seed)
    # Convert all 32 hash bits to a uniform value in [-0.5, 0.5).
    return float(hash_u32(counter)) * (1.0 / 4294967296.0) - 0.5


@cuda.jit(device=True, inline=True)
def dithered_u8(value: float, global_index: int, channel_salt: int,
                 seed: int, half_range: float) -> int:
    noise = channel_noise(global_index, channel_salt, seed) * (2.0 * half_range)
    quantized = int(value + noise + 0.5)
    if quantized < 0:
        return 0
    if quantized > 255:
        return 255
    return quantized


@cuda.jit
def scatter_nearest_strip(
    points: np.ndarray,
    cells: np.ndarray,
    count: int,
    accumulation: np.ndarray,
    global_render_size: int,
    accumulation_y_start: int,
    accumulation_height: int,
    side: int,
    interpolation: float,
    point_index_offset: int,
    color_dither_half_range: float,
    color_dither_seed: int,
) -> None:
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
    flat = local_y * global_render_size + ix
    if color_dither_half_range > 0.0:
        red_float, green_float, blue_float = _point_rgb_float(px, py)
        global_index = point_index_offset + index
        red = dithered_u8(
            red_float, global_index, 0xA511E9B3, color_dither_seed,
            color_dither_half_range,
        )
        green = dithered_u8(
            green_float, global_index, 0x63D83595, color_dither_seed,
            color_dither_half_range,
        )
        blue = dithered_u8(
            blue_float, global_index, 0xC6BC2796, color_dither_seed,
            color_dither_half_range,
        )
    else:
        red, green, blue = _point_rgb(px, py)
    cuda.atomic.add(accumulation, (flat, 0), red)
    cuda.atomic.add(accumulation, (flat, 1), green)
    cuda.atomic.add(accumulation, (flat, 2), blue)
    cuda.atomic.add(accumulation, (flat, 3), 1)


@cuda.jit
def resolve_disk_window(
    accumulation: np.ndarray,
    image: np.ndarray,
    width: int,
    accumulation_height: int,
    resolved_height: int,
    resolved_offset: int,
    kernel: np.ndarray,
    kernel_width: int,
) -> None:
    index = cuda.grid(1)
    if index >= image.shape[0]:
        return
    resolved_y = index // width
    x = index - resolved_y * width
    source_y = resolved_y + resolved_offset
    radius = kernel_width // 2
    sum_red = np.uint64(0)
    sum_green = np.uint64(0)
    sum_blue = np.uint64(0)
    sum_weight = np.uint64(0)
    for ky in range(kernel_width):
        yy = source_y + ky - radius
        if yy < 0 or yy >= accumulation_height:
            continue
        for kx in range(kernel_width):
            xx = x + kx - radius
            if xx < 0 or xx >= width:
                continue
            kernel_weight = np.uint64(kernel[ky * kernel_width + kx])
            if kernel_weight == 0:
                continue
            source = yy * width + xx
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


def warm_kernels(disk: np.ndarray) -> float:
    start = time.perf_counter()
    points = np.array([[0.25, 0.75], [0.75, 0.25]], dtype=np.float32)
    cells = np.array([0, 3], dtype=np.int32)
    accumulation = cuda.to_device(np.zeros((32, 4), dtype=np.uint32))
    device_points = cuda.to_device(points)
    device_cells = cuda.to_device(cells)
    scatter_nearest_strip[1, 32](
        device_points,
        device_cells,
        2,
        accumulation,
        4,
        0,
        8,
        2,
        np.float32(DEFAULT_INTERP),
        0,
        np.float32(0.0),
        0,
    )
    device_disk = cuda.to_device(disk.ravel())
    image = cuda.device_array((16, 3), dtype=np.uint8)
    resolve_disk_window[1, 32](
        accumulation,
        image,
        4,
        8,
        4,
        2,
        device_disk,
        disk.shape[0],
    )
    cuda.synchronize()
    del accumulation, device_points, device_cells, device_disk, image
    gc.collect()
    cuda.current_context().deallocations.clear()
    return elapsed(start)


def stream_strip(
    points: np.memmap,
    assignment: np.memmap,
    accumulation: Any,
    global_render_size: int,
    accumulation_y_start: int,
    accumulation_height: int,
    side: int,
    interpolation: float,
    stream_chunk: int,
    color_dither_half_range: float,
    color_dither_seed: int,
    strip_number: int,
    strip_count: int,
) -> float:
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
            device_points,
            device_cells,
            count,
            accumulation,
            global_render_size,
            accumulation_y_start,
            accumulation_height,
            side,
            np.float32(interpolation),
            i0,
            np.float32(color_dither_half_range),
            color_dither_seed,
        )
        chunk_number += 1
        if chunk_number % progress_interval == 0 or i1 == n:
            cuda.synchronize()
            seconds = elapsed(start)
            log(
                f"  strip {strip_number}/{strip_count}: {i1:,}/{n:,} points "
                f"({100.0 * i1 / n:5.1f}%) in {seconds:.2f}s"
            )
    cuda.synchronize()
    seconds = elapsed(start)
    del device_points, device_cells
    return seconds


def render_strip(
    points: np.memmap,
    assignment: np.memmap,
    final_image: np.memmap,
    output_y_start: int,
    output_y_end: int,
    out_size: int,
    scale: int,
    side: int,
    interpolation: float,
    stream_chunk: int,
    disk: np.ndarray,
    color_dither_half_range: float,
    color_dither_seed: int,
    strip_number: int,
    strip_count: int,
) -> dict[str, Any]:
    global_render_size = out_size * scale
    core_start = output_y_start * scale
    core_end = output_y_end * scale
    resolved_start = max(0, core_start - FILTER_HALO)
    resolved_end = min(global_render_size, core_end + FILTER_HALO)
    disk_radius = disk.shape[0] // 2
    accumulation_start = max(0, resolved_start - disk_radius)
    accumulation_end = min(global_render_size, resolved_end + disk_radius)
    accumulation_height = accumulation_end - accumulation_start
    resolved_height = resolved_end - resolved_start
    accumulation_pixels = global_render_size * accumulation_height
    resolved_pixels = global_render_size * resolved_height

    free, total = cuda.current_context().get_memory_info()
    estimated_gpu_bytes = accumulation_pixels * 16 + resolved_pixels * 3 + stream_chunk * 12
    if estimated_gpu_bytes + 2 * 2**30 > free:
        raise RuntimeError(
            f"Strip {strip_number} needs approximately {estimated_gpu_bytes / 2**30:.2f} GiB "
            f"plus a 2 GiB reserve, but only {free / 2**30:.2f} GiB is free"
        )

    log(
        f"STRIP {strip_number}/{strip_count}: output rows {output_y_start}:{output_y_end}, "
        f"high-resolution core {core_start}:{core_end}, GPU estimate "
        f"{estimated_gpu_bytes / 2**30:.2f} GiB"
    )
    strip_start = time.perf_counter()
    allocation_start = time.perf_counter()
    accumulation = cuda.device_array((accumulation_pixels, 4), dtype=np.uint32)
    accumulation[:] = 0
    cuda.synchronize()
    allocation_seconds = elapsed(allocation_start)

    accumulation_seconds = stream_strip(
        points,
        assignment,
        accumulation,
        global_render_size,
        accumulation_start,
        accumulation_height,
        side,
        interpolation,
        stream_chunk,
        color_dither_half_range,
        color_dither_seed,
        strip_number,
        strip_count,
    )

    resolve_start = time.perf_counter()
    device_disk = cuda.to_device(disk.ravel())
    device_image = cuda.device_array((resolved_pixels, 3), dtype=np.uint8)
    blocks = (resolved_pixels + THREADS - 1) // THREADS
    resolve_disk_window[blocks, THREADS](
        accumulation,
        device_image,
        global_render_size,
        accumulation_height,
        resolved_height,
        resolved_start - accumulation_start,
        device_disk,
        disk.shape[0],
    )
    cuda.synchronize()
    high_resolution = device_image.copy_to_host().reshape(
        resolved_height, global_render_size, 3
    )
    resolve_seconds = elapsed(resolve_start)
    del accumulation, device_disk, device_image
    cuda.current_context().deallocations.clear()

    filter_start = time.perf_counter()
    source = Image.fromarray(high_resolution)
    filtered = source.filter(ImageFilter.BoxBlur(radius=(scale - 1.0) / 2.0))
    crop_top = core_start - resolved_start
    core = filtered.crop((0, crop_top, global_render_size, crop_top + core_end - core_start))
    reduced = core.resize(
        (out_size, output_y_end - output_y_start),
        resample=Image.Resampling.BOX,
    )
    reduced_array = np.asarray(reduced, dtype=np.uint8)
    final_image[output_y_start:output_y_end] = reduced_array
    final_image.flush()
    channel_min = reduced_array.reshape(-1, 3).min(axis=0).astype(int).tolist()
    channel_max = reduced_array.reshape(-1, 3).max(axis=0).astype(int).tolist()
    channel_sum = reduced_array.reshape(-1, 3).sum(axis=0, dtype=np.uint64).astype(int).tolist()
    filter_seconds = elapsed(filter_start)
    source.close()
    filtered.close()
    core.close()
    reduced.close()
    del high_resolution, reduced_array
    gc.collect()
    cuda.current_context().deallocations.clear()

    result = {
        "strip": strip_number,
        "outputRows": [output_y_start, output_y_end],
        "highResolutionCoreRows": [core_start, core_end],
        "highResolutionResolvedRows": [resolved_start, resolved_end],
        "highResolutionAccumulationRows": [accumulation_start, accumulation_end],
        "estimatedGpuBytes": estimated_gpu_bytes,
        "gpuFreeBytesAtStart": int(free),
        "gpuTotalBytes": int(total),
        "allocationAndClearSeconds": allocation_seconds,
        "accumulationSeconds": accumulation_seconds,
        "resolveAndTransferSeconds": resolve_seconds,
        "filterDownsampleAndStitchSeconds": filter_seconds,
        "totalSeconds": elapsed(strip_start),
        "channelMin": channel_min,
        "channelMax": channel_max,
        "channelSum": channel_sum,
    }
    log(f"STRIP {strip_number}/{strip_count} done in {result['totalSeconds']:.3f}s")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=DEFAULT_SIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--interp", type=float, default=DEFAULT_INTERP)
    parser.add_argument("--out-size", type=int, default=DEFAULT_OUT_SIZE)
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, choices=(4, 8))
    parser.add_argument(
        "--disk-diameter",
        type=float,
        help="Smoothstep disk diameter in high-resolution pixels (defaults to --scale)",
    )
    parser.add_argument("--tile-output-rows", type=int, default=DEFAULT_TILE_OUTPUT_ROWS)
    parser.add_argument("--stream-chunk", type=int, default=DEFAULT_STREAM_CHUNK)
    parser.add_argument(
        "--color-dither-half-range",
        type=float,
        default=0.0,
        help="Uniform per-channel noise half-range on the 0..255 scale before quantization",
    )
    parser.add_argument(
        "--color-dither-seed",
        type=int,
        default=0,
        help="Counter-based deterministic color-dither seed",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.interp <= 1.0:
        raise RuntimeError("--interp must be in [0, 1]")
    if min(args.side, args.out_size, args.tile_output_rows, args.stream_chunk) <= 0:
        raise RuntimeError("side, output size, tile rows, and stream chunk must be positive")
    disk_diameter = float(args.scale) if args.disk_diameter is None else args.disk_diameter
    if disk_diameter <= 0.0:
        raise RuntimeError("--disk-diameter must be positive")
    if not 0.0 <= args.color_dither_half_range <= 0.5:
        raise RuntimeError("--color-dither-half-range must be in [0, 0.5]")
    if not cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    # These are trusted arrays generated by this process. Pillow's default
    # decompression-bomb threshold is lower than both a high-resolution strip
    # and the requested 14,400-square output, even though no untrusted input is
    # being decoded here.
    Image.MAX_IMAGE_PIXELS = None

    n = args.side * args.side
    assignment_file = assignment_path(args.side, args.seed)
    assignment = validate_assignment(assignment_file, n)
    points, cache_metadata = prepare_position_cache(args.side, args.seed)
    default_directory = (
        results_dir()
        / "renders"
        / f"billion_shufflesnap_{args.side}_seed{args.seed}_interp75_aa_4800"
    )
    output_path = args.output or (
        default_directory
        / f"10_disk_4x_{args.out_size * args.scale}_box_area_{args.out_size}.png"
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists (use --overwrite): {output_path}")

    icc, icc_path = load_icc()
    disk = smoothstep_kernel(disk_diameter)
    warm_seconds = warm_kernels(disk)
    device = cuda.get_current_device()
    log(f"CUDA ready: {device.name.decode()} compute capability {device.compute_capability}")
    log(f"CUDA JIT warmup {warm_seconds:.3f}s (excluded from generation timing)")
    if args.color_dither_half_range > 0.0:
        log(
            "COLOR DITHER continuous Lab-derived sRGB + independent uniform "
            f"[-{args.color_dither_half_range}, +{args.color_dither_half_range}) "
            f"before uint8 quantization; deterministic seed {args.color_dither_seed}"
        )

    temporary_pixels = output_path.with_suffix(output_path.suffix + ".pixels.tmp")
    temporary_png = output_path.with_suffix(output_path.suffix + ".tmp")
    timing_path = output_path.with_suffix(".timing.json")
    for temporary in (temporary_pixels, temporary_png):
        if temporary.exists():
            raise RuntimeError(f"Incomplete temporary file already exists: {temporary}")

    generation_start = time.perf_counter()
    final_image = np.memmap(
        temporary_pixels,
        mode="w+",
        dtype=np.uint8,
        shape=(args.out_size, args.out_size, 3),
    )
    strip_count = math.ceil(args.out_size / args.tile_output_rows)
    strips: list[dict[str, Any]] = []
    for strip_index, output_y_start in enumerate(
        range(0, args.out_size, args.tile_output_rows), start=1
    ):
        output_y_end = min(args.out_size, output_y_start + args.tile_output_rows)
        strips.append(
            render_strip(
                points,
                assignment,
                final_image,
                output_y_start,
                output_y_end,
                args.out_size,
                args.scale,
                args.side,
                args.interp,
                args.stream_chunk,
                disk,
                args.color_dither_half_range,
                args.color_dither_seed,
                strip_index,
                strip_count,
            )
        )

    encode_start = time.perf_counter()
    final_image.flush()
    pil_image = Image.fromarray(final_image)
    pil_image.save(
        temporary_png,
        format="PNG",
        icc_profile=icc,
        dpi=(300.0, 300.0),
        compress_level=6,
        optimize=False,
    )
    pil_image.close()
    png_encoding_seconds = elapsed(encode_start)
    temporary_png.replace(output_path)
    generation_seconds = elapsed(generation_start)

    pixel_count = args.out_size * args.out_size
    channel_min = np.min([item["channelMin"] for item in strips], axis=0).astype(int).tolist()
    channel_max = np.max([item["channelMax"] for item in strips], axis=0).astype(int).tolist()
    channel_sum = np.sum(
        np.asarray([item["channelSum"] for item in strips], dtype=np.uint64), axis=0
    )
    channel_mean = (channel_sum / pixel_count).tolist()
    del final_image
    gc.collect()
    temporary_pixels.unlink()

    validation_start = time.perf_counter()
    old_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(output_path) as image:
            image.load()
            if image.size != (args.out_size, args.out_size):
                raise RuntimeError(f"Incorrect dimensions: {image.size}")
            if image.mode != "RGB":
                raise RuntimeError(f"Incorrect mode: {image.mode}")
            if image.info.get("icc_profile") != icc:
                raise RuntimeError("PNG does not contain the expected sRGB ICC profile")
            dpi = image.info.get("dpi")
            if not dpi or any(abs(float(value) - 300.0) > 0.1 for value in dpi):
                raise RuntimeError(f"PNG does not report 300 ppi: {dpi}")
    finally:
        Image.MAX_IMAGE_PIXELS = old_max_pixels
    validation_seconds = elapsed(validation_start)

    result = {
        "schemaVersion": 2,
        "source": {
            "assignment": str(assignment_file),
            "positions": cache_metadata,
            "side": args.side,
            "pointCount": n,
            "seed": args.seed,
            "interpolation": args.interp,
        },
        "output": {
            "path": str(output_path),
            "width": args.out_size,
            "height": args.out_size,
            "mode": "RGB",
            "ppi": 300,
            "iccProfile": icc_path,
            "fileBytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
            "channelMin": channel_min,
            "channelMax": channel_max,
            "channelMean": channel_mean,
        },
        "method": {
            "name": f"disk_{args.scale}x_box_area_tiled",
            "conceptualHighResolutionSize": [args.out_size * args.scale] * 2,
            "supersampleScale": args.scale,
            "diskDiameterHighResolutionPixels": disk_diameter,
            "diskKernel": disk.tolist(),
            "diskKernelWeightSum": int(disk.sum()),
            "boxWidthHighResolutionPixels": args.scale,
            "boxRadiusHighResolutionPixels": (args.scale - 1.0) / 2.0,
            "areaReductionScale": args.scale,
            "tileOutputRows": args.tile_output_rows,
            "resolvedHaloHighResolutionRows": FILTER_HALO,
            "stripCount": strip_count,
            "integerAccumulation": "uint32",
            "pointColor": {
                "source": "continuous Lab-derived sRGB channel values",
                "quantization": "round to uint8 after optional additive dither",
                "ditherDistribution": "independent uniform per point and RGB channel",
                "ditherHalfRange8BitValues": args.color_dither_half_range,
                "ditherInterval": [
                    -args.color_dither_half_range,
                    args.color_dither_half_range,
                ],
                "ditherUpperBoundExclusive": True,
                "ditherSeed": args.color_dither_seed,
                "deterministicCounter": "global point index plus channel salt",
            },
        },
        "implementation": {
            "gpu": device.name.decode(),
            "computeCapability": list(device.compute_capability),
            "numba": __import__("numba").__version__,
            "numpy": np.__version__,
            "pillow": __import__("PIL").__version__,
            "streamChunkPoints": args.stream_chunk,
            "cudaJitWarmupSecondsExcluded": warm_seconds,
        },
        "timing": {
            "totalGenerationSeconds": generation_seconds,
            "pngEncodingSeconds": png_encoding_seconds,
            "validationSecondsExcluded": validation_seconds,
            "strips": strips,
        },
        "validation": {
            "dimensions": [args.out_size, args.out_size],
            "mode": "RGB",
            "dpi": list(dpi),
            "iccProfileMatchesStandardSrgb": True,
        },
    }
    atomic_write_json(timing_path, result)
    log(
        f"DONE {output_path.name}: {args.out_size}x{args.out_size} from conceptual "
        f"{args.out_size * args.scale}x{args.out_size * args.scale} in "
        f"{generation_seconds:.3f}s ({output_path.stat().st_size / 2**20:.2f} MiB)"
    )
    log(f"SHA256 {result['output']['sha256']}")
    log(f"TIMING {timing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
