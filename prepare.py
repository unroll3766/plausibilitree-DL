#!/usr/bin/env python3
"""prepare.py: turn a plausibilitree run_pipeline.py output folder into
per-plot deep-learning feature/label pairs.

    conda run -n ssf python prepare.py <run_output_dir> -o <out_dir> \
        --source understory --normalize neg1_1 --subsample voxel \
        --voxel-res 0.25 --min-points-drop 100 --min-points-pad 1024 \
        --pad-jitter 0.01 --normalize-labels-by-area \
        --include-chm --include-dtm --seed 42

Input  : a folder of plot_*/ directories as written by run_pipeline.py
         (or a single plot directory).
Output : <out_dir>/features/<plot_uid>.npz  (points + extra dims + meta)
         <out_dir>/labels/<plot_uid>.json    (labels from the plot info JSON)
         <out_dir>/manifest.json             (per-plot status + config)

Exit codes: 0 ok, 2 config/usage error, 3 some plots failed. See notes.md.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plausdl.config import DEFAULT_LABELS, PrepConfig
from plausdl.pipeline import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare plausibilitree plot clouds for deep learning "
                    "(features + labels, one pair per plot).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="run_pipeline.py output folder (plot_*/ dirs) "
                                  "or a single plot directory")
    p.add_argument("-o", "--out", required=True, help="output folder")
    p.add_argument("--source", default="understory",
                   choices=["understory", "terrain", "trimmed", "raw", "auto"],
                   help="which plot cloud to read ('auto' = furthest-processed present)")
    p.add_argument("--normalize", default="none", choices=["none", "neg1_1", "zero_1"],
                   help="rescale each axis to [-1,1] (neg1_1) or [0,1] (zero_1); "
                        "the original bounding box is preserved in the feature meta")
    p.add_argument("--subsample", default="none", choices=["none", "voxel", "fps"],
                   help="downsampling method")
    p.add_argument("--voxel-res", type=float, default=None,
                   help="VOXEL_RES: voxel edge in metres (voxel downsample, "
                        "PDAL voxeldownsize mode=center equivalent)")
    p.add_argument("--n-fps", type=int, default=None,
                   help="N_FPS: number of points for farthest-point subsampling")
    p.add_argument("--min-points-drop", type=int, default=None,
                   help="MIN_POINTS_DROP: exclude the plot if it has fewer points "
                        "after subsampling (checked before padding)")
    p.add_argument("--min-points-pad", type=int, default=None,
                   help="MIN_POINTS_PAD: pad with jittered duplicates up to this "
                        "many points when the cloud has fewer")
    p.add_argument("--pad-jitter", type=float, default=0.01,
                   help="PAD_JITTER: per-axis uniform jitter (m) added to padded points")
    p.add_argument("--labels", default=",".join(DEFAULT_LABELS),
                   help="comma-separated label keys to take from the plot info JSON")
    p.add_argument("--normalize-labels-by-area", action="store_true",
                   help="divide each numeric label by PLOT_AREA")
    p.add_argument("--include-chm", action="store_true",
                   help="include the plot CHM raster in the feature .npz")
    p.add_argument("--include-dtm", action="store_true",
                   help="include the plot DTM raster in the feature .npz")
    p.add_argument("--seed", type=int, default=42, help="seed for all stochastic draws")
    p.add_argument("--limit", type=int, default=None, help="process only the first N plots")
    p.add_argument("--force", action="store_true",
                   help="redo plots whose outputs already exist")
    p.add_argument("--stop-on-error", action="store_true",
                   help="abort the batch on the first plot failure (exit 2)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PrepConfig(
        source=args.source,
        normalize=args.normalize,
        subsample=args.subsample,
        voxel_res=args.voxel_res,
        n_fps=args.n_fps,
        min_points_drop=args.min_points_drop,
        min_points_pad=args.min_points_pad,
        pad_jitter=args.pad_jitter,
        labels=[s.strip() for s in args.labels.split(",") if s.strip()],
        normalize_labels_by_area=args.normalize_labels_by_area,
        include_chm=args.include_chm,
        include_dtm=args.include_dtm,
        seed=args.seed,
        limit=args.limit,
        force=args.force,
        stop_on_error=args.stop_on_error,
    )
    return run(args.input, args.out, cfg)


if __name__ == "__main__":
    sys.exit(main())
