"""Per-axis normalization of a point cloud (notes.md, "Normalization").

Each axis is rescaled to [-1, 1] ("neg1_1") or [0, 1] ("zero_1") using the
cloud's own per-axis min/max. The original bounding box — and any geospatial
information carried alongside — is preserved in the returned metadata, and
the per-axis affine parameters are recorded so the transform is invertible.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def normalize_points(xyz: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict]:
    """Rescale each axis of `xyz` to the range selected by `mode`.

    Returns (xyz_out, meta). meta holds the original bounding box and the
    per-axis affine (scale, offset) with xyz_out = scale * xyz + offset, so
    the original coordinates are recoverable exactly (up to float round-off).
    A zero-extent axis maps to 0 in both modes (no division by zero).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError(f"xyz must be (N, 3) with N > 0, got {xyz.shape}")
    if mode == "none":
        lo = xyz.min(axis=0)
        hi = xyz.max(axis=0)
        return xyz.copy(), {
            "mode": "none",
            "original_bbox": [lo.tolist(), hi.tolist()],
            "scale": [1.0, 1.0, 1.0],
            "offset": [0.0, 0.0, 0.0],
        }
    if mode == "neg1_1":
        lo_out, hi_out = -1.0, 1.0
    elif mode == "zero_1":
        lo_out, hi_out = 0.0, 1.0
    else:
        raise ValueError(f"unknown normalize mode {mode!r}")

    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    extent = hi - lo
    scale = np.where(extent > 0, (hi_out - lo_out) / np.where(extent > 0, extent, 1.0), 0.0)
    offset = np.where(extent > 0, lo_out - scale * lo, 0.0)  # zero-extent axis -> 0
    out = xyz * scale + offset
    return out, {
        "mode": mode,
        "range": [lo_out, hi_out],
        "original_bbox": [lo.tolist(), hi.tolist()],
        "scale": scale.tolist(),
        "offset": offset.tolist(),
    }
