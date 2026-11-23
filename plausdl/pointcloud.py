"""Point-cloud IO: read plausibilitree's canonical LAZ clouds.

Clouds are LAS 1.4 / point format 0 with the canonical extra dims of
plausibilitree's schema.py (tree_id, shrub_id, semantic_code,
classification, branch_code, branch_id, height_above_ground). The reader is
generic: every extra dimension present on disk is returned, so clouds from
any pipeline stage read the same way.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def read_laz(path: str) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Read a LAZ/LAS file into (xyz, dims).

    xyz  : (N, 3) float64 scaled coordinates.
    dims : {extra_dim_name: array} plus "classification"; every extra
           dimension on disk is included at its stored dtype.
    """
    import laspy

    las = laspy.read(path)
    xyz = np.column_stack([np.asarray(las.x, dtype=np.float64),
                           np.asarray(las.y, dtype=np.float64),
                           np.asarray(las.z, dtype=np.float64)])
    dims: Dict[str, np.ndarray] = {
        "classification": np.asarray(las.classification, dtype=np.uint8)
    }
    names = []
    pf = las.point_format
    if hasattr(pf, "extra_dimension_names"):
        names = list(pf.extra_dimension_names)
    else:  # laspy < 2.4 fallback
        names = [d.name for d in pf.extra_dimensions]
    for name in names:
        dims[name] = np.asarray(getattr(las, name))
    return xyz, dims
