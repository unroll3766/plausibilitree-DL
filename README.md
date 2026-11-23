# plausibilitree-DL

Deep-learning companion library for [plausibilitree](../plausibilitree).
plausibilitree generates synthetic forest plots; this library turns a
`run_pipeline.py` output folder into per-plot **feature/label file pairs**
that any deep learning framework can load. The design document is
[`notes.md`](notes.md).

## Quick start

```bash
# 1. produce plots with plausibilitree (see its README_run_pipeline.md)
conda run -n ssf python ../plausibilitree/fia_helper.py ../plausibilitree/CA_TREE_SMALL.csv -o plots -r 15 --seed 42
conda run -n ssf python ../plausibilitree/run_pipeline.py plots --radius 15 --seed 42 --generate-dtm --understory

# 2. prepare the DL dataset
conda run -n ssf python prepare.py plots -o dl_dataset \
    --source understory \
    --normalize neg1_1 \
    --subsample voxel --voxel-res 0.25 \
    --min-points-drop 100 --min-points-pad 1024 --pad-jitter 0.01 \
    --normalize-labels-by-area \
    --include-chm --include-dtm \
    --seed 42
```

## Input

A folder created by plausibilitree's `run_pipeline.py`: one `plot_*/`
directory per plot (or a single plot directory), each with
`plot_placement/las/<uid>_{raw,trimmed,terrain,understory}.laz`,
`plot_placement/textual/<uid>_info.json`, and `<uid>_scene.json`.

## Output

```
dl_dataset/
  features/<plot_uid>.npz   # points (N,3) float32, extra dims (semantic_code,
                            # tree_id, ...), optional chm/dtm arrays, meta JSON:
                            # original bbox + geospatial info + normalization
                            # affine params + subsample record
  labels/<plot_uid>.json    # labels from the plot info JSON (NUM_TREES,
                            # BASAL_AREA, VOL_TREES, ...), optionally / PLOT_AREA
  manifest.json             # config + per-plot status + summary
```

Features and labels are separate files with a shared name (`<plot_uid>`),
portable to any DL library.

## Processing per plot (notes.md)

1. **Normalization** (optional): each axis rescaled to [-1,1] (`neg1_1`) or
   [0,1] (`zero_1`). The original bounding box — including geospatial info
   when the plot is anchored — is preserved in the feature `meta`, along
   with the per-axis affine params so the transform is invertible.
2. **Subsampling**: voxel downsample on an **origin-anchored** grid
   (PDAL `filters.voxeldownsize` mode=center style — one point per voxel at
   the voxel center — but anchored at the origin rather than PDAL's first
   point, so the voxelization is identical for every cloud;
   `--voxel-res VOXEL_RES`) or farthest-point subsampling (`--n-fps N_FPS`). Afterwards, in this order: plots with
   fewer than `--min-points-drop MIN_POINTS_DROP` points are excluded;
   plots with fewer than `--min-points-pad MIN_POINTS_PAD` points are padded
   with randomly duplicated, jittered (+/- `--pad-jitter PAD_JITTER`) points.
   `MIN_POINTS_DROP > MIN_POINTS_PAD` is a config error.
3. **Feature/label extraction**: features = the prepared cloud + meta;
   labels = the selected keys of the plot info JSON, optionally divided by
   PLOT_AREA. `--include-chm` / `--include-dtm` add the plot's CHM/DTM
   rasters to the feature file.

## CLI

```
conda run -n ssf python prepare.py --help
```

Key flags: `--source {understory,terrain,trimmed,raw,auto}` (default
`understory`), `--normalize {none,neg1_1,zero_1}`, `--subsample
{none,voxel,fps}`, `--voxel-res`, `--n-fps`, `--min-points-drop`,
`--min-points-pad`, `--pad-jitter`, `--labels`, `--normalize-labels-by-area`,
`--include-chm`, `--include-dtm`, `--seed`, `--limit`, `--force`,
`--stop-on-error`.

Exit codes: 0 ok, 2 config/usage error, 3 some plots failed. Re-running a
batch skips plots whose outputs already exist (resumable), unless
`--force`.

## Training a model on the prepared data

[`notebooks/train_pointnet.ipynb`](notebooks/train_pointnet.ipynb) — bare-bones
single-head **PointNet regression** in PyTorch. Set `DEVICE_CHOICE`
(`"cpu"` / `"mps"` for the Apple integrated GPU / `"cuda"` / `"auto"`),
`FEATURES_DIR` and `LABELS_DIR` (the `features/` and `labels/` folders written
by `prepare.py`), and `LABEL_KEY` (e.g. `"VOL_TREES"`) in the config cell.
Run with a PyTorch environment (`notebooks/requirements.txt`), e.g.
`jupyter lab notebooks/train_pointnet.ipynb`.

## Development

- Env: conda env `ssf` (shared with plausibilitree), always via
  `conda run -n ssf python ...`. The notebook needs a separate training env
  with `torch` (see `notebooks/requirements.txt`); `ssf` has no torch.
- `conda run -n ssf python check.py` — compile + import every module.
- `conda run -n ssf python tests/run_all.py` — test suite (check `$?`).
- Verification log: `tests/CHECKLOG.md`.
