# CHECKLOG — plausibilitree-DL

Unique verification runs; check `$?` discipline. Date: 2026-11-23.

- 2026-11-23 — Static sanity: `conda run -n ssf python check.py` → exit 0
  (10 files compiled, all `plausdl` modules import OK).
- 2026-11-23 — Test suite: `conda run -n ssf python tests/run_all.py` → exit 0,
  15 passed / 0 failed. Covers: neg1_1/zero_1/none normalization (incl.
  zero-extent axis and affine invertibility), voxel downsample vs hand-computed
  PDAL mode=center semantics (first point per voxel, coords -> voxel center,
  incl. negative coords), FPS count/determinism/no-op, drop-before-pad ordering,
  pad jitter bounds, config validation (MIN_POINTS_DROP > MIN_POINTS_PAD error,
  voxel/fps required params), label extraction + PLOT_AREA normalization,
  end-to-end run on a synthetic 2-plot folder (ok/dropped/failed statuses,
  exit 3, resume skip), source=auto fallback, and integration against
  ../plausibilitree/examples/plot_demo (26988 -> 57 points at voxel 0.5).
- 2026-11-23 — CLI end-to-end: `conda run -n ssf python prepare.py
  ../plausibilitree/examples/plot_demo -o tests/_tmp/cli_demo --normalize zero_1
  --subsample fps --n-fps 512 --min-points-pad 512 --normalize-labels-by-area
  --include-chm --seed 1` → exit 0; labels/plot_demo.json shows values divided
  by PLOT_AREA (NUM_TREES 4 -> 0.0056588...), manifest.json summary 1 ok.
