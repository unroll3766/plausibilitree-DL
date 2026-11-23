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
