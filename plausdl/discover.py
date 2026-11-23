"""Discover plots in a folder produced by plausibilitree's run_pipeline.py.

A run output folder contains one `plot_*/` directory per plot (the same
folder the orchestrator read its `plot_*.csv` tables from). Each plot
directory follows plausibilitree's layout.py:

    PLOT_UID/plot_placement/las/PLOT_UID_{raw,trimmed,terrain,understory}.laz
    PLOT_UID/plot_placement/textual/PLOT_UID_info.json
    PLOT_UID/plot_placement/textual/PLOT_UID_scene.json

A single plot directory (containing `plot_placement/` directly) is also
accepted, so examples/plot_demo works as a one-plot input.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional

# furthest-processed first; "auto" picks the first present
SOURCE_ORDER = ("understory", "terrain", "trimmed", "raw")


class PlotError(Exception):
    """A per-plot problem that should skip the plot, not kill the batch."""


@dataclass
class Plot:
    uid: str
    dir: str
    info_json: str

    @property
    def placement(self) -> str:
        return os.path.join(self.dir, "plot_placement")

    def las_dir(self) -> str:
        return os.path.join(self.placement, "las")

    def textual_dir(self) -> str:
        return os.path.join(self.placement, "textual")

    def cloud_path(self, source: str) -> str:
        """Resolve the source cloud for this plot. `source` is one of
        SOURCE_ORDER, or "auto" = furthest-processed present. Canonical name
        is `.laz`; a legacy `.las` file is used when only that exists."""
        names = SOURCE_ORDER if source == "auto" else (source,)
        for name in names:
            for ext in (".laz", ".las"):
                p = os.path.join(self.las_dir(), f"{self.uid}_{name}{ext}")
                if os.path.exists(p):
                    return p
        raise PlotError(
            f"no source cloud for plot {self.uid!r}: looked for "
            + ", ".join(f"{self.uid}_{n}.laz/.las" for n in names)
            + f" in {self.las_dir()}")

    def info(self) -> dict:
        with open(self.info_json, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def scene_path(self) -> Optional[str]:
        p = os.path.join(self.textual_dir(), f"{self.uid}_scene.json")
        return p if os.path.exists(p) else None

    def raster_path(self, info_key: str, layout_key_dir: Optional[str] = None) -> Optional[str]:
        """Resolve an optional raster recorded in the info JSON (CHM_PATH /
        DTM_PATH). Those paths are stored relative to the plot directory;
        the collection root and the layout's rasters/ dir are fallbacks."""
        info = self.info()
        rel = info.get(info_key)
        candidates = []
        if rel:
            candidates += [os.path.join(self.dir, rel),
                           os.path.join(os.path.dirname(self.dir), rel),
                           rel]
        if layout_key_dir:
            candidates.append(os.path.join(self.placement, layout_key_dir,
                                           f"{self.uid}_{'dtm'}.tif"))
        for c in candidates:
            if os.path.exists(c):
                return c
        return None


def discover_plots(input_dir: str) -> List[Plot]:
    """All plots under `input_dir` (sorted by uid). Accepts either a
    run_pipeline.py output folder (plot_*/ subdirectories) or a single plot
    directory."""
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise PlotError(f"input folder not found: {input_dir}")
    if os.path.isdir(os.path.join(input_dir, "plot_placement")):
        dirs = [input_dir]
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(input_dir, "plot_*"))
                      if os.path.isdir(d))
    plots: List[Plot] = []
    for d in dirs:
        infos = sorted(glob.glob(os.path.join(d, "plot_placement", "textual",
                                              "*_info.json")))
        if not infos:
            continue
        plots.append(Plot(uid=os.path.basename(d), dir=d, info_json=infos[0]))
    if not plots:
        raise PlotError(
            f"no plots found in {input_dir} (expected plot_*/plot_placement/textual/"
            "*_info.json — run plausibilitree's run_pipeline.py first)")
    return plots
