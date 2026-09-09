"""Physical wrist-lens model shared by the simulator and the real camera path.

One ``LensModel`` defines the 256x256 policy observation on both sides:

* **Real camera:** the raw 1920x1080 frame is squashed to 256x256 with an exact
  area (box) filter (``real_operator``). Nothing else touches the pixels, so a
  deployed policy sees the lens exactly as calibrated.
* **Simulator:** MuJoCo renders only pinholes, so the adapter renders a wider
  pinhole view and ``sim_operator`` resamples it into the same 256x256
  observation *as the real lens would have imaged it*: every observation
  pixel's footprint in the raw frame is supersampled, each sample is mapped raw
  -> undistorted -> render pixel through the calibrated model, and the bilinear
  taps are accumulated into one fixed sparse operator.

The distortion model is OpenCV's pinhole family (``k1 k2 p1 p2 k3 [k4 k5 k6]``),
i.e. what ``tools/calibrate_camera_intrinsics.py`` writes. Everything here is
numpy only and deterministic, because the operator's output is part of every
capture's content hash. The model is only invertible where the distorted radius
still grows with the undistorted radius; ``valid_radius`` records that limit and
``undistort`` refuses points beyond it instead of returning garbage (the
2026-09-08 centre-only fit failed exactly this way outside 929 px).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

LENS_MODELS = ("opencv_pinhole_5", "opencv_rational_8")
FRAMINGS = ("full_frame_squash",)
OBSERVATION_SIZE = 256


@dataclass(frozen=True)
class LensModel:
    model: str
    image_size: tuple[int, int]            # raw (width, height)
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...]                # k1 k2 p1 p2 k3 [k4 k5 k6]
    render_fovy_deg: float                 # symmetric pinhole the simulator renders
    render_size: tuple[int, int]           # (width, height) of that render
    framing: str = "full_frame_squash"
    observation_size: int = OBSERVATION_SIZE
    supersample: tuple[int, int] = (8, 5)  # samples per observation pixel (x, y) for the sim operator
    intrinsics_sha256: str | None = None   # provenance of the calibration file the values came from
    calibration_rms_px: float | None = None

    def __post_init__(self):
        if self.model not in LENS_MODELS:
            raise ValueError(f"unsupported lens model {self.model!r}")
        n = 5 if self.model == "opencv_pinhole_5" else 8
        if len(self.dist) != n or not np.isfinite(self.dist).all():
            raise ValueError(f"{self.model} needs {n} finite distortion coefficients")
        if self.framing not in FRAMINGS:
            raise ValueError(f"unsupported framing {self.framing!r}")
        w, h = self.image_size
        if w <= 0 or h <= 0 or not (0 < self.cx < w and 0 < self.cy < h):
            raise ValueError("principal point must lie inside the image")
        if not (self.fx > 0 and self.fy > 0):
            raise ValueError("focal lengths must be positive")
        if not 1.0 <= self.render_fovy_deg <= 150.0:
            raise ValueError("render fovy out of range")
        rw, rh = self.render_size
        if rw <= 0 or rh <= 0:
            raise ValueError("render size must be positive")
        if self.observation_size != OBSERVATION_SIZE:
            raise ValueError("v2 observations are fixed at 256x256")
        if min(self.supersample) < 1:
            raise ValueError("supersample must be at least 1x1")
        # Normalise container types so the frozen dataclass hashes/compares by value.
        object.__setattr__(self, "image_size", (int(w), int(h)))
        object.__setattr__(self, "render_size", (int(rw), int(rh)))
        object.__setattr__(self, "dist", tuple(float(v) for v in self.dist))
        object.__setattr__(self, "supersample", (int(self.supersample[0]), int(self.supersample[1])))

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_intrinsics_file(cls, path, *, render_fovy_deg, render_size, framing="full_frame_squash", supersample=(8, 5)):
        """Build from ``camera_intrinsics.json`` (its ``selected`` block)."""
        data = Path(path).read_bytes()
        sel = json.loads(data)["selected"]
        name = sel["model"]
        if name.startswith("pinhole_rational_8coef"):
            model, n = "opencv_rational_8", 8
        elif name.startswith("pinhole_5coef"):
            model, n = "opencv_pinhole_5", 5
        else:
            raise ValueError(f"cannot build a LensModel from calibration model {name!r}")
        K = sel["K"]
        w, h = sel.get("image_size", (1920, 1080))
        return cls(model=model, image_size=(int(w), int(h)), fx=float(K[0][0]), fy=float(K[1][1]), cx=float(K[0][2]), cy=float(K[1][2]),
                   dist=tuple(float(v) for v in sel["dist"][:n]), render_fovy_deg=float(render_fovy_deg), render_size=tuple(render_size),
                   framing=framing, supersample=tuple(supersample), intrinsics_sha256=hashlib.sha256(data).hexdigest(),
                   calibration_rms_px=(float(sel["rms_px"]) if "rms_px" in sel else None))

    @classmethod
    def from_mapping(cls, raw):
        raw = dict(raw)
        for key in ("image_size", "render_size", "dist", "supersample"):
            if key in raw and raw[key] is not None:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def identity(self):
        return asdict(self)

    # ---- geometry ---------------------------------------------------------
    @property
    def fovy_deg(self):
        return float(2 * np.degrees(np.arctan(self.image_size[1] / (2 * self.fy))))

    @property
    def fovx_deg(self):
        return float(2 * np.degrees(np.arctan(self.image_size[0] / (2 * self.fx))))

    @property
    def _coefficients(self):
        d = np.zeros(8)
        d[: len(self.dist)] = self.dist
        return d

    def _radial(self, r2):
        k1, k2, _p1, _p2, k3, k4, k5, k6 = self._coefficients
        num = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        den = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        dnum = k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2
        dden = k4 + 2 * k5 * r2 + 3 * k6 * r2 ** 2
        g = num / den
        dg = (dnum * den - num * dden) / den ** 2
        return g, dg

    @property
    def valid_radius(self):
        """Largest distorted normalised radius the radial model can still invert (first peak of r_d(r))."""
        r = np.linspace(0.0, np.tan(np.deg2rad(80.0)), 40001)
        g, _ = self._radial(r * r)
        rd = r * g
        drop = np.nonzero(np.diff(rd) <= 0)[0]
        i = int(drop[0]) if len(drop) else len(rd) - 1
        return float(rd[i])

    def distort(self, x, y):
        """Undistorted normalised coordinates -> distorted normalised coordinates (OpenCV forward model)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        _k1, _k2, p1, p2, *_ = self._coefficients
        r2 = x * x + y * y
        g, _ = self._radial(r2)
        xd = x * g + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * g + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd

    def undistort(self, xd, yd, *, iterations=30, tolerance=1e-10):
        """Inverse of ``distort`` by Newton iteration with the analytic Jacobian; raises outside the invertible domain."""
        xd = np.asarray(xd, dtype=np.float64)
        yd = np.asarray(yd, dtype=np.float64)
        limit = self.valid_radius
        if np.any(np.hypot(xd, yd) >= 0.999 * limit):
            raise ValueError(f"point beyond the lens model's invertible radius ({limit:.3f} normalised)")
        _k1, _k2, p1, p2, *_ = self._coefficients
        x = xd.copy()
        y = yd.copy()
        for _ in range(iterations):
            r2 = x * x + y * y
            g, dg = self._radial(r2)
            fx_ = x * g + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd
            fy_ = y * g + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - yd
            j11 = g + 2 * x * x * dg + 2 * p1 * y + 6 * p2 * x
            j12 = 2 * x * y * dg + 2 * p1 * x + 2 * p2 * y
            j21 = 2 * x * y * dg + 2 * p1 * x + 2 * p2 * y
            j22 = g + 2 * y * y * dg + 6 * p1 * y + 2 * p2 * x
            det = j11 * j22 - j12 * j21
            dx = (j22 * fx_ - j12 * fy_) / det
            dy = (-j21 * fx_ + j11 * fy_) / det
            x = x - dx
            y = y - dy
            if np.max(np.abs(dx)) < tolerance and np.max(np.abs(dy)) < tolerance:
                break
        bx, by = self.distort(x, y)
        residual = np.max(np.hypot(bx - xd, by - yd)) if xd.size else 0.0
        if not np.isfinite(residual) or residual > 1e-8:
            raise ValueError(f"lens inverse did not converge (residual {residual:.2e})")
        return x, y

    def raw_to_normalised(self, u, v):
        return (np.asarray(u, dtype=np.float64) - self.cx) / self.fx, (np.asarray(v, dtype=np.float64) - self.cy) / self.fy

    @property
    def render_focal_px(self):
        return self.render_size[1] / 2.0 / np.tan(np.deg2rad(self.render_fovy_deg) / 2.0)

    def render_pixel(self, x, y):
        """Undistorted normalised coordinates -> render pixel (pixel-centre convention, row 0 at the top).

        MuJoCo's symmetric pinhole puts the optical axis at the image centre; OpenCV's normalised
        y points down, which matches image rows in both the raw frame and the render.
        """
        f = self.render_focal_px
        rw, rh = self.render_size
        return (rw - 1) / 2.0 + f * np.asarray(x), (rh - 1) / 2.0 + f * np.asarray(y)

    def required_render_fovy_deg(self, margin=1.05):
        """Smallest symmetric 16:9-or-wider pinhole fovy whose render covers the whole undistorted raw frame."""
        w, h = self.image_size
        xs = np.linspace(0, w - 1, 193)
        ys = np.linspace(0, h - 1, 109)
        border = np.array([[x, 0] for x in xs] + [[x, h - 1] for x in xs] + [[0, y] for y in ys] + [[w - 1, y] for y in ys])
        x, y = self.undistort(*self.raw_to_normalised(border[:, 0], border[:, 1]))
        rw, rh = self.render_size
        need = max(np.abs(y).max(), np.abs(x).max() * rh / rw)
        return float(2 * np.degrees(np.arctan(need * margin)))

    # ---- operators ----------------------------------------------------------
    def sim_operator(self):
        return _sim_operator(self)

    def real_operator(self):
        return _real_operator(self)


class Resampler:
    """Fixed sparse linear operator: uint8 image (H, W, C) -> uint8 image (rows, cols, C)."""

    def __init__(self, source_shape, output_shape, row_ptr, col, weight):
        self.source_shape = tuple(source_shape)          # (height, width)
        self.output_shape = (int(output_shape), int(output_shape)) if np.isscalar(output_shape) else tuple(int(v) for v in output_shape)
        self.row_ptr = np.asarray(row_ptr, dtype=np.int64)
        self.col = np.asarray(col, dtype=np.int64)
        self.weight = np.asarray(weight, dtype=np.float32)
        self._starts = self.row_ptr[:-1]

    @property
    def nnz(self):
        return int(self.col.size)

    def apply(self, image):
        image = np.asarray(image)
        if image.shape[:2] != self.source_shape or image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError(f"resampler expects a uint8 image of shape {self.source_shape + (3,)}, got {image.shape} {image.dtype}")
        flat = image.reshape(-1, image.shape[2])
        gathered = flat[self.col].astype(np.float32) * self.weight[:, None]
        summed = np.add.reduceat(gathered, self._starts, axis=0)
        out = np.clip(np.rint(summed), 0, 255).astype(np.uint8)
        return out.reshape(self.output_shape[0], self.output_shape[1], image.shape[2])


def _coalesce(rows, cols, weights, n_rows):
    order = np.lexsort((cols, rows))
    rows, cols, weights = rows[order], cols[order], weights[order]
    key_change = np.ones(rows.size, dtype=bool)
    key_change[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    starts = np.nonzero(key_change)[0]
    summed = np.add.reduceat(weights.astype(np.float64), starts)
    rows_u, cols_u = rows[starts], cols[starts]
    counts = np.bincount(rows_u, minlength=n_rows)
    if np.any(counts == 0):
        raise ValueError("resampler has an empty output row")
    row_ptr = np.concatenate([[0], np.cumsum(counts)])
    # Normalise each output row to unit weight so brightness is preserved exactly.
    totals = np.add.reduceat(summed, row_ptr[:-1])
    summed = summed / np.repeat(totals, counts)
    return row_ptr, cols_u, summed.astype(np.float32)


def _footprint_samples(lens):
    """Raw-frame sample positions (pixel-centre convention) for every observation pixel: (N, S, 2)."""
    w, h = lens.image_size
    n = lens.observation_size
    sx, sy = w / n, h / n
    ssx, ssy = lens.supersample
    ox = (np.arange(ssx) + 0.5) / ssx * sx
    oy = (np.arange(ssy) + 0.5) / ssy * sy
    j = np.arange(n)
    i = np.arange(n)
    u = (j[:, None] * sx - 0.5)[:, :, None] + ox[None, None, :]      # (1, n, ssx) -> broadcast later
    v = (i[:, None] * sy - 0.5)[:, :, None] + oy[None, None, :]
    # Build (n_rows, n_cols, ssy, ssx) grids.
    U = np.broadcast_to(u.reshape(1, n, 1, ssx), (n, n, ssy, ssx))
    V = np.broadcast_to(v.reshape(n, 1, ssy, 1), (n, n, ssy, ssx))
    return U.reshape(n * n, ssy * ssx), V.reshape(n * n, ssy * ssx)


def _bilinear_entries(rows, u, v, width, height, label):
    """Bilinear taps at continuous positions (u, v) in a (height, width) grid; raises if outside.

    Positions within half a pixel beyond the last pixel centre are still inside that pixel's
    area and are clamped to it; anything further out is a coverage failure.
    """
    if np.any(u < -0.5) or np.any(v < -0.5) or np.any(u > width - 0.5) or np.any(v > height - 0.5):
        raise ValueError(f"{label}: sample falls outside the source image; widen the render field of view or size")
    u = np.clip(u, 0.0, width - 1.0)
    v = np.clip(v, 0.0, height - 1.0)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.minimum(u0 + 1, width - 1)
    v1 = np.minimum(v0 + 1, height - 1)
    fu = u - u0
    fv = v - v0
    taps_col = np.stack([v0 * width + u0, v0 * width + u1, v1 * width + u0, v1 * width + u1], axis=-1)
    taps_w = np.stack([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv], axis=-1)
    taps_row = np.broadcast_to(rows[..., None], taps_col.shape)
    return taps_row.ravel(), taps_col.ravel(), taps_w.ravel()


@lru_cache(maxsize=4)
def _sim_operator(lens):
    n = lens.observation_size
    U, V = _footprint_samples(lens)
    rows = np.broadcast_to(np.arange(n * n)[:, None], U.shape)
    xd, yd = lens.raw_to_normalised(U.ravel(), V.ravel())
    x, y = lens.undistort(xd, yd)
    ru, rv = lens.render_pixel(x, y)
    rw, rh = lens.render_size
    r, c, w = _bilinear_entries(rows.ravel(), ru, rv, rw, rh, "sim lens operator")
    row_ptr, col, weight = _coalesce(r, c, w, n * n)
    return Resampler((rh, rw), n, row_ptr, col, weight)


class SeparableResampler:
    """Exact area (box) filter raw (H, W, 3) -> (n, n, 3) as two small dense matrices: out = Ay @ img @ Ax^T."""

    def __init__(self, source_shape, output_size, ay, ax):
        self.source_shape = tuple(source_shape)
        self.output_size = int(output_size)
        self.ay = np.asarray(ay, dtype=np.float32)   # (n, H)
        self.ax = np.asarray(ax, dtype=np.float32)   # (n, W)

    def apply(self, image):
        image = np.asarray(image)
        if image.shape[:2] != self.source_shape or image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError(f"resampler expects a uint8 image of shape {self.source_shape + (3,)}, got {image.shape} {image.dtype}")
        out = np.empty((self.output_size, self.output_size, image.shape[2]), dtype=np.uint8)
        axt = self.ax.T
        for channel in range(image.shape[2]):
            plane = image[:, :, channel].astype(np.float32)
            out[:, :, channel] = np.clip(np.rint(self.ay @ plane @ axt), 0, 255).astype(np.uint8)
        return out


def _box_weights(size, count):
    """(count, size) matrix of exact overlap fractions between output cells and source pixels."""
    scale = size / count
    matrix = np.zeros((count, size), dtype=np.float64)
    for j in range(count):
        lo, hi = j * scale, (j + 1) * scale
        for k in range(int(np.floor(lo)), min(int(np.ceil(hi)), size)):
            ov = min(hi, k + 1) - max(lo, k)
            if ov > 1e-12:
                matrix[j, k] = ov / scale
    return matrix


@lru_cache(maxsize=4)
def _real_operator(lens):
    """Exact area (box) filter of the raw frame onto the observation grid (INTER_AREA semantics)."""
    w, h = lens.image_size
    n = lens.observation_size
    return SeparableResampler((h, w), n, _box_weights(h, n), _box_weights(w, n))
