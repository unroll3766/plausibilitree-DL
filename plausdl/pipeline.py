"""Per-plot preparation pipeline: discover -> load -> normalize ->
subsample -> drop/pad -> export feature/label pair.

Batch behavior mirrors plausibilitree's run_pipeline.py: a failing plot is
logged and skipped (exit code 3 at the end if any plot failed);
--stop-on-error aborts on the first failure (exit 2). Completed plots are
skipped on a re-run unless --force, so batches are resumable. All
stochastic draws (FPS start point, pad selection/jitter) come from one
seeded Generator per run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

from .config import PrepConfig
from .discover import Plot, PlotError, discover_plots
from .features import (build_labels, feature_meta, read_raster, write_features,
                       write_labels)
from .normalize import normalize_points
from .pointcloud import read_laz
from .subsample import DropPlot, drop_or_pad_count, pad_points, subsample

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_SOME_FAILED = 3


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def process_plot(plot: Plot, out_dir: str, cfg: PrepConfig,
                 rng: np.random.Generator) -> dict:
    """Process one plot; returns its manifest record. Raises DropPlot when
    the plot is excluded, PlotError/other exceptions on failure."""
    rec: dict = {"plot_uid": plot.uid}
    cloud = plot.cloud_path(cfg.source)
    source_name = os.path.splitext(os.path.basename(cloud))[0].split(f"{plot.uid}_")[-1]
    rec["source_cloud"] = os.path.relpath(cloud, plot.dir)
    rec["source"] = source_name

    xyz, dims = read_laz(cloud)
    n_in = len(xyz)
    rec["n_points_in"] = n_in
    if n_in == 0:
        raise DropPlot("empty source cloud")

    xyz, norm_meta = normalize_points(xyz, cfg.normalize)

    xyz, dims = subsample(xyz, dims, cfg.subsample, cfg.voxel_res, cfg.n_fps, rng)
    rec["n_points_subsampled"] = len(xyz)
    rec["subsample"] = {
        "method": cfg.subsample,
        "voxel_res": cfg.voxel_res,
        "n_fps": cfg.n_fps,
        "min_points_drop": cfg.min_points_drop,
        "min_points_pad": cfg.min_points_pad,
        "pad_jitter": cfg.pad_jitter,
    }

    # drop check before padding (notes.md)
    pad_target = drop_or_pad_count(len(xyz), cfg.min_points_drop, cfg.min_points_pad)
    if pad_target is not None:
        xyz, dims = pad_points(xyz, dims, pad_target, cfg.pad_jitter, rng)
        rec["padded_to"] = pad_target
    rec["n_points_out"] = len(xyz)

    info = plot.info()
    scene = None
    sp = plot.scene_path()
    if sp:
        with open(sp, "r", encoding="utf-8") as fh:
            scene = json.load(fh)

    rasters: Dict[str, np.ndarray] = {}
    raster_meta: Dict[str, dict] = {}
    if cfg.include_chm:
        p = plot.raster_path("CHM_PATH")
        if p is None:
            raise PlotError(f"plot {plot.uid!r}: CHM_PATH not found in info JSON")
        arr, rmeta = read_raster(p)
        rasters["chm"] = arr
        raster_meta["chm"] = rmeta
    if cfg.include_dtm:
        p = plot.raster_path("DTM_PATH",
                             layout_rel=os.path.join("rasters", f"{plot.uid}_dtm.tif"))
        if p is None:
            raise PlotError(f"plot {plot.uid!r}: DTM_PATH not found in info JSON")
        arr, rmeta = read_raster(p)
        rasters["dtm"] = arr
        raster_meta["dtm"] = rmeta

    labels, missing = build_labels(info, cfg.labels, cfg.normalize_labels_by_area)
    if missing:
        _log(f"warning: plot {plot.uid}: label keys missing from info JSON: "
             f"{', '.join(missing)}")
        rec["missing_labels"] = missing

    meta = feature_meta(plot.uid, rec["source_cloud"], source_name, n_in, len(xyz),
                        norm_meta, info, scene, rec["subsample"], raster_meta)
    write_features(os.path.join(out_dir, "features", f"{plot.uid}.npz"),
                   xyz, dims, meta, rasters)
    write_labels(os.path.join(out_dir, "labels", f"{plot.uid}.json"),
                 plot.uid, labels, info, cfg.normalize_labels_by_area)
    rec["status"] = "ok"
    return rec


def run(input_dir: str, out_dir: str, cfg: PrepConfig) -> int:
    """Run the preparation over `input_dir` into `out_dir`. Returns the
    process exit code: 0 ok, 2 aborted (config error or --stop-on-error),
    3 finished but some plots failed."""
    try:
        cfg.validate()
    except ValueError as exc:
        _log(f"error: {exc}")
        return EXIT_CONFIG_ERROR

    out_dir = os.path.abspath(out_dir)
    features_dir = os.path.join(out_dir, "features")
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    try:
        plots = discover_plots(input_dir)
    except PlotError as exc:
        _log(f"error: {exc}")
        return EXIT_CONFIG_ERROR
    if cfg.limit is not None:
        plots = plots[: cfg.limit]

    rng = np.random.default_rng(cfg.seed)
    manifest: dict = {
        "input_dir": os.path.abspath(input_dir),
        "out_dir": out_dir,
        "config": asdict(cfg),
        "plots": [],
    }
    n_ok = n_dropped = n_failed = n_skipped = 0
    stopped = False
    t0 = time.time()
    for plot in plots:
        feat = os.path.join(features_dir, f"{plot.uid}.npz")
        lab = os.path.join(labels_dir, f"{plot.uid}.json")
        if not cfg.force and os.path.exists(feat) and os.path.exists(lab):
            manifest["plots"].append({"plot_uid": plot.uid, "status": "skipped_existing"})
            n_skipped += 1
            print(f"{plot.uid}: skipped (outputs exist)")
            continue
        try:
            rec = process_plot(plot, out_dir, cfg, rng)
        except DropPlot as exc:
            rec = {"plot_uid": plot.uid, "status": "dropped", "reason": str(exc)}
            n_dropped += 1
            print(f"{plot.uid}: dropped ({exc})")
        except Exception as exc:  # isolate failures per plot
            rec = {"plot_uid": plot.uid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            n_failed += 1
            _log(f"error: plot {plot.uid}: {type(exc).__name__}: {exc}")
            print(f"{plot.uid}: FAILED ({type(exc).__name__}: {exc})")
            manifest["plots"].append(rec)
            if cfg.stop_on_error:
                stopped = True
                break
            continue
        manifest["plots"].append(rec)
        if rec["status"] == "ok":
            n_ok += 1
            print(f"{plot.uid}: ok ({rec['n_points_in']} -> {rec['n_points_out']} points, "
                  f"source={rec['source']})")

    manifest["summary"] = {
        "n_plots": len(manifest["plots"]),
        "ok": n_ok,
        "dropped": n_dropped,
        "failed": n_failed,
        "skipped_existing": n_skipped,
        "seconds": round(time.time() - t0, 3),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    s = manifest["summary"]
    print(f"done: {s['ok']} ok, {s['dropped']} dropped, {s['failed']} failed, "
          f"{s['skipped_existing']} skipped (existing) in {s['seconds']} s -> {out_dir}")
    if stopped:
        return EXIT_CONFIG_ERROR
    return EXIT_SOME_FAILED if n_failed else EXIT_OK
