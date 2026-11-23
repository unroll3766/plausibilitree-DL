"""Subsampling and the drop/pad checks (notes.md, "Subsampling").

Two downsampling methods:

- voxel: equivalent to PDAL filters.voxeldownsize with mode=center,
  verified empirically against PDAL 3.5.5 on this box: the voxel grid of
  edge VOXEL_RES is anchored at the cloud's FIRST point a (PDAL anchors the
  grid at the first point read, not at the origin or the header offset);
  per axis the voxel id is floor((p - a)/res + 0.5) (half-cell ties round
  up) and the voxel center is a + id*res. For each populated voxel the
  first point found is retained with its x/y/z overwritten to the voxel
  center; all other points in the voxel are dropped, and the retained
  point's non-coordinate dims are carried unchanged (PDAL's documented
  behavior, including its warning that center mode overwrites X/Y/Z only).
- fps: farthest point sampling down to N_FPS points. The starting point is
  drawn from the seeded generator; afterwards each step adds the point with
  the largest squared distance to the already-selected set (ties: lowest
  index). If the cloud has <= N_FPS points it is returned unchanged.

After subsampling the drop/pad checks run, in this order (notes.md): if the
count is below MIN_POINTS_DROP the plot is dropped; otherwise, if below
MIN_POINTS_PAD, points are randomly duplicated with a uniform per-axis jitter
of +/- PAD_JITTER until the count reaches MIN_POINTS_PAD.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def voxel_downsample_center(xyz: np.ndarray, dims: Dict[str, np.ndarray],
                            res: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """PDAL voxeldownsize mode=center equivalent. See module docstring.

    The grid is anchored at the first point `a = xyz[0]`; voxel id per axis
    is floor((p - a)/res + 0.5) and the voxel center is a + id*res.
    """
    if res <= 0:
        raise ValueError(f"voxel resolution must be > 0, got {res}")
    if len(xyz) == 0:
        return xyz.copy(), {k: np.asarray(v)[:0] for k, v in dims.items()}
    anchor = xyz[0].astype(np.float64)
    ids = np.floor((xyz - anchor) / res + 0.5).astype(np.int64)
    uniq, first = np.unique(ids, axis=0, return_index=True)
    order = np.argsort(first, kind="stable")  # keep first-occurrence order
    centers = anchor + uniq[order].astype(np.float64) * res
    keep = first[order]
    out_dims = {k: np.asarray(v)[keep] for k, v in dims.items()}
    return centers, out_dims


def farthest_point_sample(xyz: np.ndarray, n_fps: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Indices of `n_fps` farthest-point-sampled rows of `xyz` (sorted).
    Returns arange(n) when the cloud is smaller than n_fps."""
    n = len(xyz)
    if n_fps < 1:
        raise ValueError(f"n_fps must be >= 1, got {n_fps}")
    if n <= n_fps:
        return np.arange(n)
    idx = np.empty(n_fps, dtype=np.int64)
    idx[0] = int(rng.integers(0, n))
    d2 = np.sum((xyz - xyz[idx[0]]) ** 2, axis=1)
    for k in range(1, n_fps):
        j = int(np.argmax(d2))
        idx[k] = j
        d2 = np.minimum(d2, np.sum((xyz - xyz[j]) ** 2, axis=1))
        d2[j] = -1.0  # never re-select
    return np.sort(idx)


def subsample(xyz: np.ndarray, dims: Dict[str, np.ndarray], method: str,
              voxel_res: Optional[float], n_fps: Optional[int],
              rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Dispatch on the subsample method; 'none' returns the input unchanged."""
    if method == "none":
        return xyz, dims
    if method == "voxel":
        return voxel_downsample_center(xyz, dims, float(voxel_res))
    if method == "fps":
        keep = farthest_point_sample(xyz, int(n_fps), rng)
        return xyz[keep], {k: np.asarray(v)[keep] for k, v in dims.items()}
    raise ValueError(f"unknown subsample method {method!r}")


def drop_or_pad_count(n_points: int, min_points_drop: Optional[int],
                      min_points_pad: Optional[int]) -> Optional[int]:
    """The drop/pad decision for a post-subsample count, in notes.md order:
    drop check first. Returns None to keep the cloud as-is, or the pad target
    when padding is needed; raises DropPlot when it must be excluded."""
    if min_points_drop is not None and n_points < min_points_drop:
        raise DropPlot(
            f"{n_points} points < MIN_POINTS_DROP={min_points_drop}; plot excluded")
    if min_points_pad is not None and n_points < min_points_pad:
        return int(min_points_pad)
    return None


class DropPlot(Exception):
    """The plot is excluded from the outputs (too few points)."""


def pad_points(xyz: np.ndarray, dims: Dict[str, np.ndarray], target: int,
               jitter: float, rng: np.random.Generator
               ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Duplicate randomly chosen points (with +/- `jitter` uniform noise per
    axis, PAD_JITTER) until the cloud has `target` points. Originals come
    first, in their original order; the padded copies are appended."""
    n = len(xyz)
    if n == 0:
        raise DropPlot("cannot pad an empty cloud")
    if target < n:
        raise ValueError(f"pad target {target} < current count {n}")
    add = target - n
    sel = rng.integers(0, n, size=add)
    noise = (rng.uniform(-jitter, jitter, size=(add, 3)) if jitter > 0
             else np.zeros((add, 3)))
    xyz_out = np.vstack([xyz, xyz[sel] + noise])
    dims_out = {k: np.concatenate([np.asarray(v), np.asarray(v)[sel]])
                for k, v in dims.items()}
    return xyz_out, dims_out
