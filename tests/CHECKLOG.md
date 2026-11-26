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
- 2026-11-23 — notebooks/autoencoder.ipynb added (tests/_tmp/gen_ae.py):
  PointNet-style autoencoder, unsupervised (features only). Knobs:
  DEVICE_CHOICE (cpu/mps/cuda/auto), FEATURES_DIR, EXTRA_DIMS, N_POINTS,
  N_SAMPLES (int or "all"; 1 = load one plot and overfit it — the train loop
  repeats a lone batch to BATCH_SIZE so BatchNorm1d doesn't crash at B=1),
  LATENT_DIM, BATCH_SIZE, EPOCHS, LR, SEED. Eval cell prints recon MSE of the
  first sample + optional original-vs-reconstructed scatter. Validated
  structurally (nbformat 4.5, 7 cells, 6 code cells compile, 16 markers) via
  tests/_tmp/validate_nb.py, which now checks BOTH notebooks. check.py exit 0,
  tests/run_all.py 18/18 exit 0. Not executed end-to-end (no torch on box).
- 2026-11-23 — User asked to double-check the autoencoder's final sample plot
  (same bug as the regression eval cell, or model fit?). Verdict: NOT the
  scaling bug — the AE notebook has no label standardization; both scatter
  series come from the same (C, N) model space (slicing contract verified
  numpy-only vs a real prepare.py output, tests/_tmp/ae_eval_shapes.py:
  batch (1,4,1024), scatter slice (2,1024), range [-0.9978, 1] at neg1_1).
  Added an input-value-range print to the eval cell as a sanity check.
  Regenerated + revalidated (17 markers), check.py exit 0, tests 18/18.
- 2026-11-23 — Autoencoder LOSS knob added: "mse" | "l1" | "huber" |
  "chamfer" (symmetric xyz Chamfer via torch.cdist; O(N_POINTS^2) memory
  noted in a comment). Train loop prints the objective plus MSE every 5
  epochs; eval prints both. Chamfer formula verified in numpy
  (tests/_tmp/chamfer_check.py): self=0, symmetric, single-point case exact.
  Notebooks revalidated (22 AE markers), check.py exit 0, tests 18/18.
- 2026-11-23 — Autoencoder eval cell now plots original vs reconstructed points
  in all three projections (x-y, x-z, y-z) as a 1x3 subplot row instead of
  x-y only. Regenerated + revalidated (26 AE markers incl. the three plane
  pairs), check.py exit 0, tests 18/18.
- 2026-11-23 — Autoencoder OUTPUT_RANGE knob: decoder xyz output bounded via
  tanh (neg1_1, [-1,1]) or sigmoid (zero_1, [0,1]), "none" = linear; extra
  dims unbounded. Eval cell also prints the recon xyz range. Bound logic
  verified in numpy (tests/_tmp/bound_check.py): xyz within range, extra dims
  bit-identical. Regenerated + revalidated (31 AE markers), check.py exit 0,
  tests 18/18.
- 2026-11-24 — Optional early stopping added to BOTH notebooks: PATIENCE
  (None = off) and MIN_DELTA knobs in the config cell, train loops monitor the
  val loss, track best_val, restore the best checkpoint via load_state_dict
  after the loop, and print an early-stopping message. Both notebooks split a
  validation set via random_split with VAL_FRAC (0.2); with a single sample
  there is no val split and early stopping prints an "inactive" note.
  Notebooks regenerated from tests/_tmp generators and structurally validated
  (nbformat 4.5, all code cells compile, 29 PointNet + 39 AE markers),
  check.py exit 0, tests 18/18. NOT executed - no torch on this box.
- 2026-11-24 — Autoencoder ARCH knob: "pointnet" (existing) or "pointnet++".
  PointNet++ = two SetAbstraction levels (FPS + ball query + point MLP +
  group max) -> global max -> FC latent -> FC decoder -> two
  FeaturePropagation levels (3-NN inverse-distance interp + skip concat) ->
  per-point head; OUTPUT_RANGE bound applies to both archs via bound_xyz.
  Knobs SA_POINTS/SA_RADII/SA_NSAMPLE; guard: max(SA_POINTS) <= N_POINTS.
  Shapes + FPS/ball-query/interp math verified with a numpy mirror
  (tests/_tmp/pp_shapes.py): SA (B,128,512)->(B,256,128), FP->(B,C,N);
  caught and fixed a torch.gather ndim mismatch in index_points (3-D points
  vs 4-D grouped index). Regenerated + revalidated (56 AE markers), check.py
  exit 0, tests 18/18. NOT executed - no torch on this box.
- 2026-11-24 — Deep re-verification of the PointNet++ notebook (a6099a1) after
  user asked to check the work. tests/_tmp/pp_shapes.py now mirrors every model
  cell function in numpy and asserts: index_points == brute-force gather in
  both branches; FPS greedy-farthest property per step; ball_query picks are
  in-ball or the nearest point; SA chain shapes (B,128,512)->(B,256,128);
  FP shapes incl. M==1 expand branch and k=min(3,M) clamp; bound_xyz ranges
  (neg1_1/zero_1/none, extra dims untouched); chamfer self=0 + symmetric;
  build_model guards (SA_POINTS<=N_POINTS, len match, bad arch/range raise);
  decoder view (B,C*N)->(B,C,N). Added k=min(3,M) so SA_POINTS[1] < 3 cannot
  crash topk. Regen idempotent (byte-identical), 57 AE markers, check.py
  exit 0, tests 18/18. Still NOT executed - no torch on this box.
- 2026-11-24 — User-reported bug in ARCH="pointnet++": training crashed with
  "dimension out of range (expected [-1, 0], got 1)" at any sample size;
  "pointnet" unaffected. Root cause: farthest_point_sample used
  dist.max(dim=1).values.argmax(dim=1) — the .values is already 1-D (B,), so
  the second argmax(dim=1) is invalid. Fixed to dist.argmax(dim=1). The numpy
  mirror had the correct form, which is why it passed. Full audit of every
  dim-argument op in the model cell (chamfer, gather x3, topk, max/min/sum
  dims, expand/unsqueeze/view) re-checked by hand; diff is exactly the one
  line. Validator gained a FORBIDDEN pattern (max(dim=1).values.argmax) so this
  cannot regress. Regenerated + revalidated (58 AE markers), check.py exit 0,
  tests 18/18. Still not executed under torch - no torch on this box.
- 2026-11-24 — REAL execution under torch (new `pytorch` conda env created by
  user; installed pytorch 2.15.0 cpuonly + numpy + matplotlib via conda).
  Executed the actual notebook cells (tests/_tmp/exec_real.py runner, synthetic
  6-plot dataset in /tmp/nbtest): train_pointnet.ipynb defaults OK (20 epochs,
  label stats, eval RMSE/MAE + scatter); autoencoder.ipynb ARCH=pointnet
  defaults OK (N_SAMPLES=1 overfit mode, 50 epochs); ARCH=pointnet++ mse OK
  (loss 2.48 -> 0.99, the FPS fix confirmed - no more dim-out-of-range);
  pointnet++ + LOSS=chamfer + N_SAMPLES=all + PATIENCE=3 OK (early stopping
  fired, best checkpoint restored); pointnet + N_SAMPLES=all + PATIENCE=2 +
  OUTPUT_RANGE=zero_1 OK (recon xyz in [0,1]). Also a numpy torch-shim
  (tests/_tmp/torch_shim.py + exec_nb.py) executed all cells with torch-like
  dim errors before the real env existed. check.py exit 0, tests 18/18.
- 2026-11-24 — Embedding->label prediction added. autoencoder.ipynb: CKPT_PATH
  knob saves {arch, in_channels, n_points, latent_dim, output_range, sa_*,
  extra_dims, state_dict} after training (torch.save); both model classes got
  an encode() method. New notebooks/predict_from_embeddings.ipynb: loads the
  ckpt (torch.load weights_only=True), rebuilds the encoder via build_model
  from the saved config (model code extracted verbatim from autoencoder.ipynb
  at build time - no drift), freezes it (requires_grad_(False)), embeds all
  plots once (n, LATENT_DIM), trains a Linear(L->256)->ReLU->Dropout->Linear
  head on z with the same LABEL_KEY/VAL_FRAC/STANDARDIZE_LABELS/PATIENCE
  patterns as train_pointnet.ipynb; eval prints RMSE/MAE + true-vs-pred
  scatter + embedding-norm histogram. REAL torch 2.15 CPU round trip verified
  on the /tmp/nbtest synthetic set: AE pointnet -> ckpt -> predict (early
  stop fired, best head restored); AE pointnet++ -> ckpt -> predict with a
  different LABEL_KEY. Both NOTEBOOK EXECUTED OK. Structural: 62 AE + 53 pred
  markers, check.py exit 0, tests 18/18.
- 2026-11-24 — User-reported CUDA OOM (94.41 GB) in predict_from_embeddings:
  the notebook embedded ALL plots in one encoder forward, building a
  (n, 1024, N_POINTS) point-MLP intermediate on the GPU. Fixed by chunked
  encoding (EMBED_BATCH=32 knob, torch.cat of per-chunk encode). Validator
  FORBIDS the old full-batch stack pattern. Verified under real torch 2.15
  CPU: chunked vs single-batch encode identical for pointnet (max|diff|
  2e-7) and within FPS-seed noise for pointnet++ (max|diff| ~1e-3; FPS draws
  its seed point per batch row via torch.randint, so chunk boundaries shift
  the RNG sequence - expected, not a bug); full notebook re-executed with
  EMBED_BATCH=2 against the pointnet++ ckpt: OK, early stopping + best-head
  restore + eval all fine. 56 pred markers, check.py exit 0, tests 18/18.
