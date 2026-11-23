# Global Agent Rules

## Scope
- Work only on this local machine. Never connect to, modify, or run jobs on remote HPC, cloud, or server infrastructure, even when the topic of discussion involves them. Generating scripts *for* those targets is fine; executing anything against them is not.
- Default to Python. Never solve a problem with JavaScript or TypeScript unless I explicitly ask for it. Use Python (or plain shell) for deliverables, scripts, tooling, and configuration.
- When a GUI is needed, use Tkinter (or another stdlib/simple Python GUI), not a web app. A plain, ugly Tkinter GUI is preferred over a slick browser-based interface.

## Tone
- Be plain and direct. No buzzwords: avoid "seamlessly", "state-of-the-art", "cloud-native", "enterprise-grade", "robust", "leverage", "revolutionize", and similar marketing language. Describe what the code actually does.
- Say no more than necessary. Short replies for small tasks; expand only for genuinely complex analysis or consequential decisions.
- Distinguish estimates from exact values, and uncertainty from verification. Never claim something was tested or works unless it was actually run.

## Honesty
- Never fabricate data, metrics, output, or results. Do not invent mock datasets, fake statistics, or illustrative output to make something look finished.
- If something is not computed or retrieved yet, show an explicit placeholder ("N/A", "pending") rather than plausible filler.
- If a command or check fails, report the actual failure. Never substitute fallback or made-up results.

## Engineering
- Python with type hints (PEP 484) on public functions and methods. Docstrings for non-obvious code; do not narrate obvious code with comments.
- No magic numbers or hardcoded paths for anything that belongs in configuration.
- Keep exploratory and trial-and-error code out of production/source paths. Keep scratch and experiment work in a clearly separate location until the user approves promoting it.
- When a bug is found and fixed, fix the root cause over patching symptoms; where a recurring failure mode exists, add a fast, explicit assertion or check rather than silent patching.
- Prefer existing code and libraries over writing new code: check the project's own modules and the standard library first. Reuse before reinventing.
- Each repo gets its own environment: a dedicated conda env (or an existing shared one that fits). Never install into system Python or base conda. Run scripts with the explicit interpreter path of that repo's env.
- Packages already installed in the repo's environment are free to use — check (`conda list` / `pip list`) before assuming something is available.
- Adding any new package (`pip install`, `conda install`) requires asking first: check the env, and if something is missing, name the package, say why, and wait for agreement.
- All diagnostics to stderr; reserve stdout for program output.
- Write batch jobs to be resumable: completed units skip on retry, failures retried in isolation.
- Long-running operations: use timeouts; never block indefinitely on input prompts.
- **Check log:** when running code to verify work in a project, keep a running log of each *unique* check in the project's tests directory (default `tests/CHECKLOG.md`). For each entry record: date, what was verified, the command or method used, and the actual result. Do not re-log repeated identical checks — only first occurrences or changed behavior. Keep entries short (one to three lines). This log is the pool we mine for future test suites.

## Pushback
- Do not be a passive yes-man. Push back on hacks, brittle shortcuts, and unnecessary complexity before writing code, with the cleaner alternative.
- If a request is ambiguous enough to admit incompatible implementations, ask 1-2 sharp questions instead of guessing.
- If asked to implement a workaround for a symptom, state the apparent root cause and ask before proceeding.

## Git
- Small, atomic, single-purpose commits.
- Commit after completing coherent units of work, but never push without explicit approval in the session.
- Never track files larger than ~100 MB; large data goes in `.gitignore`.
- Never commit secrets; keep them in a gitignored `.env` or local config.

## Memory
Two-tier long-term memory; the session scratchpad is not reliable storage.

**Tier 1 — Global: `~/agent-memory/`** (local git repo, never pushed anywhere)
- `global.md`: cross-project lessons, environment quirks, stable preferences.
- `projects/<name>.md`: per project — current state, decisions and why, dead ends ("graveyard").
- Protocol: read the relevant files at the start of a session; append short, dated (`YYYY-MM-DD`) entries at meaningful checkpoints (task completed, bug understood, dead end discovered). Never rewrite old entries, only append. Commit with the message "memory: <summary>".
- Keep entries short and factual. If a note stops being true, say so in a new entry rather than editing history.

**Tier 2 — Per project: that repo's own `AGENTS.md`**
- Durable facts that belong to the codebase (build commands, test commands, project conventions, settled architecture decisions) go into the project's `AGENTS.md` and are committed like code.
- When a session discovers something of lasting value about a project, write it there, not just in `~/agent-memory/`.

## Safety
- If told to stop, not proceed, or plan only: stop immediately. Do not write code, edit files, or run commands until told to continue.
- Never run destructive teardowns (project deletions, `terraform destroy`, `rm -rf` outside a project temp area) without explicit instruction naming the target.
- Long searches and bulk operations on this box: bound them (max 2 attempts / reasonable limits), then stop and ask rather than grinding.

---

# plausibilitree-DL — project addendum

Everything above (the global agent rules) applies in full; this section
adds facts specific to this repository.

## What this repo is
- The deep-learning companion library to `../plausibilitree`. plausibilitree
  generates synthetic forest plots; this repo turns a `run_pipeline.py`
  output folder into per-plot feature/label file pairs that any DL framework
  can load. `notes.md` (owner-written, do not edit) is the design document
  and wins on conflict.
- plausibilitree ruled (2026-11 review, W21): normalization is out of scope
  there — the DL side owns it. This repo is that DL side.

## Input layout (produced by ../plausibilitree, see its layout.py)
- Input folder = a `run_pipeline.py` output folder: `plot_*/` directories,
  each with `plot_placement/las/<uid>_{raw,trimmed,terrain,understory}.laz`
  (LAS 1.4, point format 0, canonical extra dims: tree_id, shrub_id,
  semantic_code, classification, branch_code, branch_id,
  height_above_ground), `plot_placement/textual/<uid>_info.json` (plot
  labels: NUM_TREES, BASAL_AREA, VOL_*, PLOT_AREA, BOUNDING_BOX, CHM_PATH,
  DTM_PATH, PLOT_CENTER, meta), and `<uid>_scene.json` (geo block when the
  plot is anchored). A single plot directory is also accepted as input.
- Default source cloud is `understory` (the furthest-processed cloud);
  `--source auto` falls back understory > terrain > trimmed > raw.

## Output layout (written by prepare.py)
- `<out>/features/<plot_uid>.npz` — `points` (N,3) float32 plus every extra
  dim at its stored dtype, optional `chm`/`dtm` rasters, and `meta` (JSON
  string: original bbox + geospatial info + normalization affine params +
  subsample record).
- `<out>/labels/<plot_uid>.json` — labels from the plot info JSON, optionally
  divided by PLOT_AREA.
- `<out>/manifest.json` — config + per-plot status (ok / dropped / failed /
  skipped_existing) + summary. Re-runs skip existing outputs unless --force.

## Environment
- Conda env: `ssf` (shared with plausibilitree; fits: numpy, laspy 2.7 +
  lazrs, scipy, rasterio). Run everything through the wrapper:
  `conda run -n ssf python <script>`. Invoking the env's python (or
  `python3`) standalone is blocked on this box.
- requirements.txt lists what the library itself needs. A dedicated env
  will be created when an actual DL framework dependency (torch etc.) is
  needed — installing anything new requires asking first.

## Commands
- Static sanity check after code edits: `conda run -n ssf python check.py`
  (compiles and imports every module).
- Test suite (exit-code discipline: check `$?`, never eyeball output):
  `conda run -n ssf python tests/run_all.py`
- Run the prep: `conda run -n ssf python prepare.py <run_dir> -o <out_dir> [flags]`
  (see `prepare.py --help`).

## Conventions
- Units: metres in the local plot frame unless the plot is georeferenced
  (info JSON BOUNDING_BOX becomes lon/lat/elev and PLOT_CENTER appears);
  geospatial info is preserved verbatim in the feature meta.
- All stochastic draws (FPS start, pad selection/jitter) come from one
  seeded `numpy.random.Generator` per run (seed via `--seed`); never use
  global random state.
- Diagnostics to stderr, per-plot progress + summary to stdout. Exit codes:
  0 ok, 2 aborted (config/usage error incl. MIN_POINTS_DROP > MIN_POINTS_PAD,
  or --stop-on-error), 3 finished but some plots failed.
- Voxel downsample is PDAL `filters.voxeldownsize` mode=center in style but
  ORIGIN-anchored (owner decision 2026-11-23: voxelization must be identical
  across any point cloud, independent of point order — PDAL itself anchors at
  the first point read, we deliberately do not): id = floor(p/res + 0.5) per
  axis (voxel centers on multiples of res, half-cell ties round up), center
  = id*res; one retained point per occupied voxel (the first point found,
  x/y/z overwritten to the voxel center), other dims carried unchanged.
- Drop check runs before padding (notes.md); both apply to the
  post-subsample count.
