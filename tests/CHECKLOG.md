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
- 2026-11-23 — Audit pass vs notes.md. Fixes: --stop-on-error now exits 2
  (was 3); empty source cloud is "dropped" not "failed"; feature meta stores
  the source cloud path relative to the plot dir (portable); check.py also
  compiles tests/. Added tests t_stop_on_error_exit_code, t_empty_cloud_dropped.
- 2026-11-23 — Voxel downsample re-verified against real PDAL (env `pdal`,
  PDAL 3.5.5): `conda run -n pdal python tests/_tmp/pdal_crosscheck.py` on
  31988 points (plot_demo understory + 5000 synthetic, incl. negative coords),
  cell 0.5 → EXACT MATCH, 10981 voxels, centers + carried dims identical.
  Probing showed PDAL anchors the voxel grid at the FIRST point read
  (not origin/header offset), id = floor((p-a)/cell + 0.5), center = a+id*cell;
  subsample.py now implements exactly that (earlier origin-anchored grid was
  wrong — 10922 vs 10981 voxels on the same cloud). Note: PDAL's readers.text
  needs "skip": 1 to drop the file header or it parses it as a (0,0,0) point.
- 2026-11-23 — Re-run after fixes: check.py exit 0 (28 files);
  tests/run_all.py 17/17 exit 0; CLI voxel run on plot_demo exit 0
  (26988 -> 72 points at res 0.5).
- 2026-11-23 — Owner decision: origin-anchored voxel grid (voxelization must
  be identical across any point cloud; PDAL's first-point anchor rejected).
  subsample.py now uses id = floor(p/res + 0.5), center = id*res (origin
  anchor). New test t_voxel_order_invariant proves centers/count are identical
  under input permutations; t_voxel_origin_anchored covers tie/dup semantics.
  check.py exit 0 (11 files; _tmp scratch excluded); tests/run_all.py 18/18
  exit 0; CLI voxel run on plot_demo exit 0 (26988 -> 78 points at res 0.5).
  PDAL crosscheck (tests/_tmp/pdal_crosscheck.py) now differs from PDAL by
  design (PDAL anchors at first point); kept only as a scratch reference.
- 2026-11-23 — notebooks/train_pointnet.ipynb generated (tests/_tmp/gen_nb.py)
  and validated (tests/_tmp/validate_nb.py): nbformat 4.5, 7 cells, all 6
  code cells compile, knobs present (DEVICE_CHOICE cpu/mps/cuda/auto,
  FEATURES_DIR/LABELS_DIR/LABEL_KEY). NOT executed end-to-end: no torch on
  this box (checked ssf/NeedLR/pdal/raster_diagrams/base). The data contract
  (npz `points` + json `labels[<key>]`) is covered by tests/run_all.py.
- 2026-11-23 — Notebook audit pass (user asked to check work): read all 7
  cells, found 3 real bugs and fixed them — train loader now uses
  drop_last (a final 1-sample batch crashes BatchNorm1d in train mode);
  label std guards n==1 (NaN) and zero-std; PlotDataset validates LABEL_KEY
  at construction with the available keys listed. Also train loss now
  divides by seen samples (drop_last). Regenerated via tests/_tmp/gen_nb.py,
  revalidated (6 code cells compile, fix markers present), check.py exit 0,
  tests/run_all.py 18/18 exit 0. Still not executed end-to-end (no torch).
- 2026-11-23 — Notebook: added a descriptive-stats cell for the chosen label
  (n, mean, std, min, max over ALL plots + optional matplotlib histogram),
  placed right after the dataset cell. Regenerated + revalidated: nbformat
  4.5, 8 cells, 7 code cells compile, stats markers present
  (tests/_tmp/validate_nb.py). check.py exit 0, tests/run_all.py 18/18 exit 0.
  Still not executed end-to-end (no torch on this box).
- 2026-11-23 — User-reported bug fixed: eval cell re-scaled the TRUE labels
  (y.numpy()*y_std+y_mean) although dataset labels are already in original
  units — NUM_TREES max 655 plotted up to ~50k. Now trues.append(y.numpy());
  preds still un-standardized. Validator asserts the wrong pattern is gone.
  Regenerated + revalidated (8 cells, 7 compile, fix marker present),
  check.py exit 0, tests/run_all.py 18/18 exit 0.
- 2026-11-23 — Notebook: EXTRA_DIMS knob — user picks which per-point channels
  from the .npz to train on (e.g. ["semantic_code"]); they are stacked onto
  xyz and PointNetReg takes in_channels = 3 + len(EXTRA_DIMS). Config cell
  prints available npz arrays + label keys; dataset validates EXTRA_DIMS /
  LABEL_KEY against the first files. Validated structurally (8 cells, 7 code
  cells compile, markers present). Numpy-only contract check against a real
  prepare.py output (tests/_tmp/nb_contract.py): npz arrays = points +
  classification, tree_id, shrub_id, semantic_code, branch_code, branch_id,
  height_above_ground + meta; points+semantic_code stack to (N, 4); label
  pairing by shared name OK. check.py exit 0, tests 18/18.
