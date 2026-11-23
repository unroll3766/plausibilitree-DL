"""Configuration for the point-cloud preparation pipeline.

Field names follow notes.md: VOXEL_RES, N_FPS, MIN_POINTS_DROP,
MIN_POINTS_PAD, PAD_JITTER. Validation encodes the notes' rules: the drop
check runs before padding, and MIN_POINTS_DROP > MIN_POINTS_PAD is an
error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Label keys generated automatically from the plot info JSON (notes.md:
# "e.g. NUM_TREES, BASAL_AREA, VOL_TREES, etc."). BOUNDING_BOX / PLOT_AREA /
# paths are not labels; PLOT_AREA only feeds --normalize-labels-by-area.
DEFAULT_LABELS: List[str] = [
    "NUM_TREES",
    "BASAL_AREA",
    "NUM_SHRUBS",
    "VOL_TREES",
    "VOL_TRUNK",
    "VOL_BRANCH",
    "VOL_FOLIAGE",
    "VOL_SHRUB",
    "VOL_GRASS",
]

NORMALIZE_MODES = ("none", "neg1_1", "zero_1")   # [-1,1] / [0,1] rescale
SUBSAMPLE_METHODS = ("none", "voxel", "fps")
SOURCES = ("understory", "terrain", "trimmed", "raw", "auto")


@dataclass
class PrepConfig:
    """One preparation run over a run_pipeline.py output folder."""

    source: str = "understory"          # plot cloud to read per plot
    normalize: str = "none"             # none | neg1_1 ([-1,1]) | zero_1 ([0,1])
    subsample: str = "none"             # none | voxel | fps
    voxel_res: Optional[float] = None   # VOXEL_RES, required for voxel
    n_fps: Optional[int] = None         # N_FPS, required for fps
    min_points_drop: Optional[int] = None   # MIN_POINTS_DROP
    min_points_pad: Optional[int] = None    # MIN_POINTS_PAD
    pad_jitter: float = 0.01            # PAD_JITTER, metres
    labels: List[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    normalize_labels_by_area: bool = False
    include_chm: bool = False
    include_dtm: bool = False
    seed: int = 42
    limit: Optional[int] = None
    force: bool = False                 # redo plots whose outputs exist
    stop_on_error: bool = False

    def validate(self) -> None:
        """Raise ValueError on an invalid combination (incl. the notes'
        MIN_POINTS_DROP > MIN_POINTS_PAD error)."""
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.normalize not in NORMALIZE_MODES:
            raise ValueError(
                f"normalize must be one of {NORMALIZE_MODES}, got {self.normalize!r}")
        if self.subsample not in SUBSAMPLE_METHODS:
            raise ValueError(
                f"subsample must be one of {SUBSAMPLE_METHODS}, got {self.subsample!r}")
        if self.subsample == "voxel":
            if self.voxel_res is None or self.voxel_res <= 0:
                raise ValueError("--voxel-res (VOXEL_RES) must be > 0 with --subsample voxel")
        elif self.voxel_res is not None:
            raise ValueError("--voxel-res (VOXEL_RES) is only valid with --subsample voxel")
        if self.subsample == "fps":
            if self.n_fps is None or self.n_fps < 1:
                raise ValueError("--n-fps (N_FPS) must be >= 1 with --subsample fps")
        elif self.n_fps is not None:
            raise ValueError("--n-fps (N_FPS) is only valid with --subsample fps")
        for name in ("min_points_drop", "min_points_pad"):
            v = getattr(self, name)
            if v is not None and v < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be >= 1, got {v}")
        if self.pad_jitter < 0:
            raise ValueError(f"--pad-jitter (PAD_JITTER) must be >= 0, got {self.pad_jitter}")
        if (self.min_points_drop is not None and self.min_points_pad is not None
                and self.min_points_drop > self.min_points_pad):
            raise ValueError(
                "MIN_POINTS_DROP > MIN_POINTS_PAD is invalid: the drop check runs "
                "before padding (notes.md), so a cloud could be dropped even though "
                "it already meets the pad floor")
        if self.limit is not None and self.limit < 1:
            raise ValueError(f"--limit must be >= 1, got {self.limit}")
