"""Feature and label extraction (notes.md, "Feature and label extraction").

Each plot becomes one feature/label pair of files sharing the plot uid:

    <out>/features/<plot_uid>.npz    the subsampled/normalized point cloud,
                                     its extra dims, optional CHM/DTM rasters,
                                     and a `meta` JSON string (bounding box,
                                     geospatial info, normalization params)
    <out>/labels/<plot_uid>.json     the labels generated from the plot info
                                     JSON (NUM_TREES, BASAL_AREA, ...)

The pair is portable to any deep learning framework: .npz/.json are
loadable everywhere (numpy, torch, tensorflow, jax, ...).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np


def build_labels(info: dict, label_keys: list[str],
                 normalize_by_area: bool) -> tuple[dict, list[str]]:
    """Labels from the plot info JSON. Returns (labels, missing_keys).
    With normalize_by_area, every numeric label is divided by PLOT_AREA."""
    labels: Dict[str, Optional[float]] = {}
    missing: list[str] = []
    area = info.get("PLOT_AREA")
    for key in label_keys:
        if key not in info:
            missing.append(key)
            continue
        v = info[key]
        if normalize_by_area:
            if not isinstance(v, (int, float)) or not isinstance(area, (int, float)) \
                    or area <= 0:
                raise ValueError(
                    f"cannot normalize label {key!r} by plot area: value={v!r}, "
                    f"PLOT_AREA={area!r}")
            v = v / area
        labels[key] = v
    return labels, missing


def read_raster(path: str) -> tuple[np.ndarray, dict]:
    """Read a raster (CHM/DTM) into (array, geo meta) via rasterio."""
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = {
            "path": path,
            "height": src.height,
            "width": src.width,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform)[:6],
        }
    return arr, meta


def feature_meta(plot_uid: str, source_cloud: str, source_name: str,
                 n_points_in: int, n_points_out: int,
                 norm_meta: dict, info: dict, scene: Optional[dict],
                 subsample_meta: dict, rasters: Dict[str, dict]) -> dict:
    """The metadata record stored inside the feature .npz.

    The original bounding box is preserved here twice over: the cloud's own
    pre-normalization bbox (norm_meta["original_bbox"]) and the info JSON's
    BOUNDING_BOX (lon/lat/elev when the plot is georeferenced, local metres
    otherwise), plus PLOT_CENTER and the scene geo block when present.
    """
    meta = {
        "plot_uid": plot_uid,
        "source_cloud": source_cloud,
        "source": source_name,
        "n_points_in": n_points_in,
        "n_points_out": n_points_out,
        "normalization": norm_meta,
        "subsample": subsample_meta,
        "bounding_box_info": info.get("BOUNDING_BOX"),
        "plot_center": info.get("PLOT_CENTER"),
        "plot_area": info.get("PLOT_AREA"),
        "plot_shape": info.get("PLOT_SHAPE"),
        "coordinate_system": info.get("meta", {}).get("coordinate_system"),
        "units": info.get("meta", {}).get("units"),
        "geo": (scene or {}).get("geo"),
        "rasters": rasters,
    }
    return meta


def write_features(path: str, xyz: np.ndarray, dims: Dict[str, np.ndarray],
                   meta: dict, rasters: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Write the feature .npz. Points are stored float32 (the usual DL
    dtype); extra dims keep their on-disk dtypes; `meta` is a JSON string."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays: Dict[str, np.ndarray] = {"points": xyz.astype(np.float32)}
    arrays.update({k: np.asarray(v) for k, v in dims.items()})
    arrays["meta"] = np.array(json.dumps(meta))
    if rasters:
        for name, arr in rasters.items():
            arrays[name] = np.asarray(arr)
    np.savez_compressed(path, **arrays)


def write_labels(path: str, plot_uid: str, labels: dict, info: dict,
                 normalize_by_area: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "plot_uid": plot_uid,
        "labels": labels,
        "normalized_by_area": normalize_by_area,
        "plot_area": info.get("PLOT_AREA"),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
