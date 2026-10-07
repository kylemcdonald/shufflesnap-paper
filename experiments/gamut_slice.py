"""Coons-patch parameterization of one CIELAB lightness slice of the Adobe RGB (1998) gamut.

Builds on the geometry of ~/Documents/GitHub/color-cube/scripts/lightness_slice.py.  The
plane L* = const is flat in linear RGB (Y is linear in RGB), so its intersection with the
unit cube is a convex polygon, here a pentagon; its boundary is the edge of the gamut at
that lightness.

The patch is laid out to match the hue layout of the L = 74 Lab (a, b) coloring of
render_billion.py (green-yellow top left, orange top right, magenta bottom right,
teal-blue bottom left):

* For a quadrilateral slice (e.g. L* = 67.5) the corners are its four vertices.
  Otherwise the corners are the boundary points with the Lab hues of that coloring's corners,
  45, 315 and 225 degrees for top right, bottom right and bottom left.  The top-left corner
  is the green vertex (hue ~147, target 135): a vertex just inside a side would send a
  crease across the patch.
* Each side runs along the gamut boundary at constant speed in Lab.
* The Coons patch blends the four sides in Lab (a, b) at the slice lightness.  Blending in
  linear RGB, as lightness_slice.py does, keeps green in one corner and lets magenta fill
  the middle (mean hue error against the L = 74 coloring 37 degrees, against 8 here).
  The Lab slice is not convex, but this patch stays inside the gamut (checked by
  ``coons_luts``).

``coons_luts`` returns the four sides sampled finely enough for a GPU kernel to evaluate
the patch by linear interpolation.
"""

from __future__ import annotations

import numpy as np

ADOBE_PRIMARIES = [(0.64, 0.33), (0.21, 0.71), (0.15, 0.06)]
ADOBE_GAMMA = 563 / 256
D65 = (0.3127, 0.329)
# corner hues of the L = 74 coloring (degrees): top left, top right, bottom right, bottom left
CORNER_HUES = (135.0, 45.0, 315.0, 225.0)
SNAP_TO_VERTEX = (True, False, False, False)
DELTA = 6 / 29


def _xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1 - x - y) / y])


def adobe_rgb_to_xyz() -> np.ndarray:
    primaries = np.array([_xy_to_xyz(*p) for p in ADOBE_PRIMARIES]).T
    return primaries * np.linalg.solve(primaries, _xy_to_xyz(*D65))


def lab_to_rgb_matrix() -> np.ndarray:
    """Linear Adobe RGB = this matrix @ (finv(fx), finv(fy), finv(fz)), D65 Lab white."""
    return np.linalg.inv(adobe_rgb_to_xyz()) @ np.diag(_xy_to_xyz(*D65))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = rgb @ adobe_rgb_to_xyz().T / _xy_to_xyz(*D65)
    f = np.where(xyz > DELTA ** 3, np.cbrt(xyz), xyz / (3 * DELTA ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    return np.where(f > DELTA, f ** 3, 3 * DELTA ** 2 * (f - 4 / 29)) @ lab_to_rgb_matrix().T


def slice_polygon(normal: np.ndarray, level: float) -> np.ndarray:
    """Vertices of {x in [0,1]^3 : normal . x = level}, in order around the polygon."""
    points = []
    for i in range(8):
        a = np.array([(i >> k) & 1 for k in range(3)], float)
        for k in range(3):
            if a[k]:
                continue
            b = a.copy()
            b[k] = 1
            fa, fb = normal @ a - level, normal @ b - level
            if fa == 0:
                points.append(a)
            elif fa * fb < 0:
                points.append(a + (b - a) * fa / (fa - fb))
    points = np.unique(np.round(points, 12), axis=0)
    center = points.mean(0)
    e1 = points[0] - center
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal / np.linalg.norm(normal), e1)
    angle = np.arctan2((points - center) @ e2, (points - center) @ e1)
    return points[np.argsort(angle)]


def _hue(lab: np.ndarray) -> np.ndarray:
    return np.arctan2(lab[..., 2], lab[..., 1])


def coons_luts(lightness: float, samples: int = 1 << 16, per_edge: int = 20_000):
    """Sides of the patch for the slice at ``lightness``.

    Returns (luts, corners, polygon): ``luts`` is (4, samples + 1, 3) float32 Lab for
    top(x), bottom(x), left(y), right(y) at parameter k / samples, with x to the right and
    y down; ``corners`` is (4, 3) Lab for top left, top right, bottom right, bottom left.
    The patch is (1-y) top(x) + y bottom(x) + (1-x) left(y) + x right(y) minus the
    bilinear blend of the corners.
    """
    normal = adobe_rgb_to_xyz()[1]
    fy = (lightness + 16) / 116
    level = fy ** 3 if fy > DELTA else 3 * DELTA ** 2 * (fy - 4 / 29)
    polygon = slice_polygon(normal, level)
    m = len(polygon)
    t = np.linspace(0, 1, per_edge, endpoint=False)[:, None]
    boundary = np.concatenate([polygon[i] + (polygon[(i + 1) % m] - polygon[i]) * t
                               for i in range(m)])
    vertex_index = np.arange(m) * per_edge
    hue = _hue(rgb_to_lab(boundary))

    # a quadrilateral slice maps onto the square with its vertices as the corners, so the
    # most saturated colors land at the corners and the patch covers the whole slice
    snaps = (True,) * 4 if m == 4 else SNAP_TO_VERTEX
    corners = []
    for target, snap in zip(CORNER_HUES, snaps):
        candidates = vertex_index if snap else np.arange(len(boundary))
        error = np.abs(np.angle(np.exp(1j * (hue[candidates] - np.radians(target)))))
        corners.append(int(candidates[np.argmin(error)]))

    n = len(boundary)
    s = np.arange(samples + 1) / samples

    def side(i0, i1):
        # the shorter way round the boundary from i0 to i1, at constant speed in Lab
        step = 1 if (i1 - i0) % n < n / 2 else -1
        index = (i0 + step * np.arange(((i1 - i0) * step) % n + 1)) % n
        lab = rgb_to_lab(boundary[index])
        cumulative = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(lab, axis=0), axis=1))])
        cumulative /= cumulative[-1]
        return np.stack([np.interp(s, cumulative, lab[:, c]) for c in range(3)], -1)

    tl, tr, br, bl = corners
    luts = np.stack([side(tl, tr), side(bl, br), side(tl, bl), side(tr, br)])
    luts[..., 0] = lightness
    corner_lab = rgb_to_lab(boundary[corners])
    corner_lab[:, 0] = lightness

    # the patch must stay in the gamut and not fold over
    u = (np.arange(512) + 0.5) / 512
    X, Y = np.meshgrid(u, u)
    patch = evaluate_lab(luts, corner_lab, X, Y)
    rgb = lab_to_rgb(patch)
    dx, dy = np.diff(patch[..., 1:], axis=1)[:-1], np.diff(patch[..., 1:], axis=0)[:, :-1]
    jacobian = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    assert rgb.min() > -1e-4 and rgb.max() < 1 + 1e-4, "patch leaves the gamut"
    assert (jacobian > 0).all() or (jacobian < 0).all(), "patch folds over"
    return luts.astype(np.float32), corner_lab.astype(np.float32), polygon


def evaluate_lab(luts: np.ndarray, corners: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lab of the patch at unit-square (x, y) (CPU reference of the GPU evaluation)."""
    samples = luts.shape[1] - 1

    def curve(lut, t):
        f = np.clip(t, 0, 1) * samples
        i = np.minimum(f.astype(int), samples - 1)
        w = (f - i)[..., None]
        return (1 - w) * lut[i] + w * lut[i + 1]

    X, Y = x[..., None], y[..., None]
    p00, p10, p11, p01 = corners
    return ((1 - Y) * curve(luts[0], x) + Y * curve(luts[1], x)
            + (1 - X) * curve(luts[2], y) + X * curve(luts[3], y)
            - ((1 - X) * (1 - Y) * p00 + X * (1 - Y) * p10 + X * Y * p11 + (1 - X) * Y * p01))


def evaluate(luts: np.ndarray, corners: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear Adobe RGB of the patch at unit-square (x, y), clipped to [0, 1]."""
    return np.clip(lab_to_rgb(evaluate_lab(luts, corners, x, y)), 0, 1)


def color_cube_luts(lightness: float, samples: int = 1 << 16):
    """The patch exactly as color-cube's scripts/lightness_slice.py builds it (copied unchanged
    to experiments/lightness_slice.py): vertex corners chosen in its fixed plane frame, sides
    at constant speed in linear RGB, Coons blend in linear RGB.

    Returns (luts, corners, polygon, sides) in the layout of ``coons_luts`` but in linear
    Adobe RGB, so the same Coons evaluation gives linear RGB directly.
    """
    import lightness_slice as ls

    normal = ls.rgb_to_xyz("adobe")[1]
    level = ls.WHITES["D65"][1] * ls.lab_finv((lightness + 16) / 116)
    polygon, angle, sides = ls.slice_polygon(normal, level)
    corners = ls.pick_corners(polygon, angle)
    m = len(polygon)
    edges = [polygon[[(corners[i] + j) % m for j in range((corners[(i + 1) % 4] - corners[i]) % m + 1)]]
             for i in range(4)]
    s = np.arange(samples + 1) / samples
    # as ls.coons: bottom and left run backwards along their edges
    luts = np.stack([ls.polyline(edges[0], s), ls.polyline(edges[2], 1 - s),
                     ls.polyline(edges[3], 1 - s), ls.polyline(edges[1], s)])
    return luts.astype(np.float32), polygon[corners].astype(np.float32), polygon, sides


def main() -> None:
    """Write the patch as a square PNG, as color-cube's lightness_slice.py does: pixel
    centers sampled, Adobe RGB gamma, 8 or 16 bit, Adobe RGB (1998) ICC profile embedded."""
    import argparse
    import struct
    import zlib
    from pathlib import Path

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--L", type=float, default=67.5, help="CIELAB L* of the slice")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--bits", type=int, choices=(8, 16), default=8)
    parser.add_argument("--icc", type=Path, default=Path("/usr/share/color/icc/colord/AdobeRGB1998.icc"))
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()

    luts, corners, polygon = coons_luts(args.L)
    u = (np.arange(args.size) + 0.5) / args.size
    x, y = np.meshgrid(u, u)
    encoded = evaluate(luts, corners, x, y) ** (1 / ADOBE_GAMMA)
    pixels = np.round(encoded * ((1 << args.bits) - 1)).astype(">u2" if args.bits == 16 else "u1")

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    out = args.out or Path(f"adobe-L{args.L:g}-lab-coons.png")
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    out.write_bytes(b"\x89PNG\r\n\x1a\n"
                    + chunk(b"IHDR", struct.pack(">IIBBBBB", args.size, args.size, args.bits, 2, 0, 0, 0))
                    + chunk(b"iCCP", b"Adobe RGB (1998)\0\0" + zlib.compress(args.icc.read_bytes(), 9))
                    + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    names = ["top left", "top right", "bottom right", "bottom left"]
    print(f"Adobe RGB (1998)  L* = {args.L:g}  -> {len(polygon)}-gon, Lab Coons patch, "
          f"{args.size}x{args.size} {args.bits}-bit  {out}")
    for name, lab in zip(names, corners):
        rgb = lab_to_rgb(lab)
        print(f"  {name:12s} linear RGB ({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})  "
              f"hue {np.degrees(np.arctan2(lab[2], lab[1])) % 360:5.1f} deg")


if __name__ == "__main__":
    main()
