#!/usr/bin/env python3
"""Test suite for plausibilitree-DL. Run: conda run -n ssf python tests/run_all.py

Builds a synthetic plot folder in tests/_tmp/ (gitignored) and checks the
normalization, subsampling (voxel PDAL-center equivalence, FPS), drop/pad
ordering, config validation, label extraction, export files, resumability,
exit codes, and — when present — the real ../plausibilitree/examples/plot_demo.

Exit code 0 = all passed; 1 = at least one failure. Check `$?`, not output.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from plausdl.config import PrepConfig  # noqa: E402
from plausdl.discover import discover_plots  # noqa: E402
from plausdl.features import build_labels  # noqa: E402
from plausdl.normalize import normalize_points  # noqa: E402
from plausdl.pipeline import run  # noqa: E402
from plausdl.subsample import (DropPlot, drop_or_pad_count,  # noqa: E402
                               farthest_point_sample, pad_points,
                               voxel_downsample_center)

TMP = os.path.join(HERE, "_tmp")
PLAUSIBILITREE = os.path.normpath(os.path.join(ROOT, "..", "plausibilitree"))

PASS = 0
FAIL = 0


def check(name: str, fn) -> None:
    global PASS, FAIL
    try:
        fn()
    except Exception as exc:
        FAIL += 1
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    else:
        PASS += 1
        print(f"PASS {name}")


# --------------------------------------------------------------------------
# synthetic plot fixture
# --------------------------------------------------------------------------

CANONICAL_DIMS = [  # name, dtype — mirrors plausibilitree schema.py
    ("tree_id", np.int32), ("shrub_id", np.int32), ("semantic_code", np.int8),
    ("classification", np.uint8), ("branch_code", np.int8),
    ("branch_id", np.int32), ("height_above_ground", np.float32),
]


def write_laz(path: str, xyz: np.ndarray, sem: np.ndarray) -> None:
    import laspy

    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = xyz.min(axis=0) if len(xyz) else [0.0, 0.0, 0.0]
    header.add_extra_dims([laspy.ExtraBytesParams(name=n, type=dt)
                           for n, dt in CANONICAL_DIMS])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    n = len(xyz)
    las.tree_id = np.ones(n, dtype=np.int32)
    las.shrub_id = np.zeros(n, dtype=np.int32)
    las.semantic_code = sem.astype(np.int8)
    las.classification = np.full(n, 5, dtype=np.uint8)
    las.branch_code = np.zeros(n, dtype=np.int8)
    las.branch_id = np.zeros(n, dtype=np.int32)
    las.height_above_ground = xyz[:, 2].astype(np.float32)
    las.write(path)


def write_plot(root: str, uid: str, xyz: np.ndarray, sem: np.ndarray,
               sources=("understory",), with_chm: bool = False) -> str:
    pdir = os.path.join(root, uid)
    for src in sources:
        write_laz(os.path.join(pdir, "plot_placement", "las", f"{uid}_{src}.laz"),
                  xyz, sem)
    info = {
        "NUM_TREES": 4, "BASAL_AREA": 0.5, "NUM_SHRUBS": 2,
        "VOL_TREES": 35.0, "VOL_TRUNK": 5.0, "VOL_BRANCH": 22.0,
        "VOL_FOLIAGE": 8.0, "VOL_SHRUB": 0.4, "VOL_GRASS": 10.0,
        "PLOT_SHAPE": "circle", "PLOT_AREA": 100.0,
        "BOUNDING_BOX": ([float(xyz[:, i].min()) for i in range(3)]
                         + [float(xyz[:, i].max()) for i in range(3)]
                         if len(xyz) else [0.0] * 6),
        "meta": {"plot_uid": uid, "coordinate_system": "local", "units": "meters"},
    }
    if with_chm:
        import rasterio
        from rasterio.transform import from_origin
        chm_rel = os.path.join("plot_placement", "deep_learning", "canopy_map",
                               "features", f"{uid}_canopy_map_raw.tif")
        chm_path = os.path.join(pdir, chm_rel)
        os.makedirs(os.path.dirname(chm_path), exist_ok=True)
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        with rasterio.open(chm_path, "w", driver="GTiff", height=3, width=4,
                           count=1, dtype="float32", crs="EPSG:32610",
                           transform=from_origin(500000, 4000000, 0.5, 0.5)) as dst:
            dst.write(arr, 1)
        info["CHM_PATH"] = chm_rel
    os.makedirs(os.path.join(pdir, "plot_placement", "textual"), exist_ok=True)
    with open(os.path.join(pdir, "plot_placement", "textual", f"{uid}_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    with open(os.path.join(pdir, "plot_placement", "textual", f"{uid}_scene.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"plot_uid": uid, "geo": None}, fh)
    return pdir


def make_run_dir() -> str:
    root = os.path.join(TMP, "run1")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    rng = np.random.default_rng(7)
    xyz = rng.uniform(0, 10, size=(200, 3))
    sem = rng.integers(0, 4, size=200).astype(np.int8)
    write_plot(root, "plot_001", xyz, sem, sources=("raw", "understory"), with_chm=True)
    # plot_002 has an info JSON but no clouds -> must fail in isolation
    pdir = os.path.join(root, "plot_002")
    os.makedirs(os.path.join(pdir, "plot_placement", "textual"), exist_ok=True)
    with open(os.path.join(pdir, "plot_placement", "textual", "plot_002_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"NUM_TREES": 1, "PLOT_AREA": 50.0,
                   "meta": {"plot_uid": "plot_002"}}, fh)
    return root


# --------------------------------------------------------------------------
# unit checks
# --------------------------------------------------------------------------

def t_normalize_neg1_1():
    xyz = np.array([[0.0, 5.0, -2.0], [10.0, 5.0, 8.0], [4.0, 5.0, 3.0]])
    out, meta = normalize_points(xyz, "neg1_1")
    assert np.allclose(out.min(axis=0), [-1.0, 0.0, -1.0]), out.min(axis=0)
    assert np.allclose(out.max(axis=0), [1.0, 0.0, 1.0]), out.max(axis=0)
    assert meta["original_bbox"] == [[0.0, 5.0, -2.0], [10.0, 5.0, 8.0]]
    assert np.allclose(out[:, 1], 0.0)  # zero-extent axis maps to 0, no NaN
    scale, offset = np.asarray(meta["scale"]), np.asarray(meta["offset"])
    nz = scale != 0
    back = (out - offset) / np.where(nz, scale, 1.0)
    assert np.allclose(back[:, nz], xyz[:, nz])  # affine transform is invertible


def t_normalize_zero_1():
    xyz = np.array([[2.0, 0.0, 0.0], [6.0, 4.0, -1.0]])
    out, meta = normalize_points(xyz, "zero_1")
    assert np.allclose(out.min(axis=0), [0, 0, 0])
    assert np.allclose(out.max(axis=0), [1, 1, 1])
    assert meta["range"] == [0.0, 1.0]


def t_normalize_none_passthrough():
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out, meta = normalize_points(xyz, "none")
    assert np.array_equal(out, xyz) and meta["mode"] == "none"


def t_voxel_origin_anchored():
    # Origin-anchored grid (owner decision 2026-11-23): id = floor(p/res+0.5)
    # per axis, center = id*res, ties round up; first point per voxel
    # retained, coords overwritten to the center, other dims carried.
    xyz = np.array([[0.1, 0.1, 0.1],    # id (0,0,0)
                    [1.1, 0.2, 0.2],    # id (1,0,0)
                    [1.4, 0.1, 0.1],    # id (1,0,0) dup
                    [-0.4, 0.1, 0.1],   # floor(-0.4+0.5)=0 -> id (0,0,0) dup
                    [-0.6, 0.1, 0.1],   # floor(-0.6+0.5)=-1 -> id (-1,0,0)
                    [0.6, 1.1, 2.1],    # id (1,1,2)
                    [1.6, 0.1, 0.1],    # floor(1.6+0.5)=2 -> id (2,0,0) [tie up]
                    [0.5, -0.5, 0.0]])  # floor(1.0,0.0,0.5) -> id (1,0,0) dup
    dims = {"semantic_code": np.array([1, 3, 2, 9, 7, 5, 4, 8], dtype=np.int8)}
    out, od = voxel_downsample_center(xyz, dims, 1.0)
    assert out.shape == (5, 3), out
    exp = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [1, 1, 2], [2, 0, 0]],
                   dtype=np.float64)
    assert np.allclose(out, exp), out
    assert list(od["semantic_code"]) == [1, 3, 7, 5, 4], od  # first per voxel wins


def t_voxel_order_invariant():
    # the grid is anchored at the origin, so voxelization (centers + count)
    # must be identical for any permutation of the input points; only the
    # retained point's dims can differ (first point per voxel wins)
    rng = np.random.default_rng(11)
    xyz = rng.uniform(-20, 20, size=(2000, 3))
    dims = {"semantic_code": rng.integers(0, 10, size=2000).astype(np.int8)}
    out0, _ = voxel_downsample_center(xyz, dims, 0.5)
    for seed in (1, 2, 3):
        perm = np.random.default_rng(seed).permutation(len(xyz))
        out1, _ = voxel_downsample_center(
            xyz[perm], {"semantic_code": dims["semantic_code"][perm]}, 0.5)
        assert len(out1) == len(out0)
        assert np.array_equal(np.sort(out0, axis=0), np.sort(out1, axis=0))


def t_fps():
    rng_pts = np.random.default_rng(1).uniform(0, 1, size=(50, 3))
    a = farthest_point_sample(rng_pts, 10, np.random.default_rng(42))
    b = farthest_point_sample(rng_pts, 10, np.random.default_rng(42))
    assert len(a) == 10 and len(set(a.tolist())) == 10
    assert np.array_equal(a, b)  # seeded -> deterministic
    assert set(a.tolist()) <= set(range(50))
    all_idx = farthest_point_sample(rng_pts, 100, np.random.default_rng(0))
    assert len(all_idx) == 50  # n <= N_FPS -> unchanged


def t_drop_pad_order_and_counts():
    # drop check before padding
    try:
        drop_or_pad_count(4, min_points_drop=5, min_points_pad=8)
    except DropPlot:
        pass
    else:
        raise AssertionError("n=4 with drop=5 must drop, not pad")
    assert drop_or_pad_count(4, 3, 8) == 8          # survives drop -> pad to 8
    assert drop_or_pad_count(10, 3, 8) is None      # above pad floor -> keep
    assert drop_or_pad_count(10, None, None) is None


def t_pad_points():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    dims = {"semantic_code": np.array([1, 2, 3], dtype=np.int8)}
    out, od = pad_points(xyz, dims, 6, 0.5, np.random.default_rng(3))
    assert out.shape == (6, 3) and len(od["semantic_code"]) == 6
    assert np.array_equal(out[:3], xyz)                       # originals untouched, in order
    for p in out[3:]:
        d = np.abs(xyz - p).max(axis=1).min()
        assert d <= 0.5 + 1e-12, (p, d)                        # each copy near an original
    assert list(od["semantic_code"][:3]) == [1, 2, 3]


def t_config_validation():
    def bad(**kw):
        try:
            PrepConfig(**kw).validate()
        except ValueError:
            return True
        return False
    assert bad(min_points_drop=10, min_points_pad=5)      # notes.md error rule
    assert bad(subsample="voxel")                          # VOXEL_RES required
    assert bad(subsample="voxel", voxel_res=0.0)
    assert bad(subsample="fps")                            # N_FPS required
    assert bad(subsample="none", voxel_res=0.5)            # misplaced param
    assert bad(min_points_drop=0)
    assert bad(pad_jitter=-0.1)
    PrepConfig(subsample="voxel", voxel_res=0.25, min_points_drop=5,
               min_points_pad=100).validate()              # valid combo


def t_labels_and_area_normalization():
    info = {"NUM_TREES": 4, "BASAL_AREA": 0.5, "PLOT_AREA": 100.0}
    labels, missing = build_labels(info, ["NUM_TREES", "BASAL_AREA", "VOL_TREES"], False)
    assert labels == {"NUM_TREES": 4, "BASAL_AREA": 0.5} and missing == ["VOL_TREES"]
    labels2, _ = build_labels(info, ["NUM_TREES", "BASAL_AREA"], True)
    assert labels2 == {"NUM_TREES": 0.04, "BASAL_AREA": 0.005}


# --------------------------------------------------------------------------
# end-to-end checks on the synthetic run dir
# --------------------------------------------------------------------------

def base_cfg(**kw) -> PrepConfig:
    defaults = dict(source="understory", normalize="neg1_1", subsample="voxel",
                    voxel_res=0.5, min_points_drop=10, min_points_pad=300,
                    pad_jitter=0.05, seed=42)
    defaults.update(kw)
    return PrepConfig(**defaults)


def t_end_to_end_and_resume():
    root = make_run_dir()
    out = os.path.join(TMP, "out1")
    rc = run(root, out, base_cfg())
    assert rc == 3, rc  # plot_002 fails (no cloud), plot_001 ok
    with open(os.path.join(out, "manifest.json")) as fh:
        man = json.load(fh)
    rec1 = next(p for p in man["plots"] if p["plot_uid"] == "plot_001")
    rec2 = next(p for p in man["plots"] if p["plot_uid"] == "plot_002")
    assert rec1["status"] == "ok" and rec2["status"] == "failed", man["plots"]
    assert rec1["n_points_in"] == 200 and rec1["n_points_out"] == 300  # padded
    assert rec1["source"] == "understory"

    with np.load(os.path.join(out, "features", "plot_001.npz")) as z:
        pts = z["points"]
        meta = json.loads(str(z["meta"]))
        assert z["semantic_code"].shape == (300,)
        # neg1_1 range holds before subsampling; voxel-center mode can push a
        # point up to voxel_res/2 past the extreme, and padding adds <= jitter
        assert pts.min() >= -1.0 - 0.25 - 0.05 - 1e-6
        assert pts.max() <= 1.0 + 0.25 + 0.05 + 1e-6
        assert meta["normalization"]["mode"] == "neg1_1"
        from plausdl.pointcloud import read_laz
        src_xyz, _ = read_laz(os.path.join(root, "plot_001", "plot_placement", "las",
                                           "plot_001_understory.laz"))
        ob = meta["normalization"]["original_bbox"]
        assert np.allclose(ob[0], src_xyz.min(axis=0))  # original bbox preserved
        assert np.allclose(ob[1], src_xyz.max(axis=0))
        assert meta["bounding_box_info"] == json.load(
            open(os.path.join(root, "plot_001", "plot_placement", "textual",
                              "plot_001_info.json")))["BOUNDING_BOX"]
        assert meta["subsample"]["method"] == "voxel" and meta["subsample"]["voxel_res"] == 0.5
    with open(os.path.join(out, "labels", "plot_001.json")) as fh:
        lab = json.load(fh)
    assert lab["labels"]["NUM_TREES"] == 4 and lab["labels"]["VOL_TREES"] == 35.0
    assert lab["normalized_by_area"] is False

    # rerun without --force -> plot_001 skipped as existing; plot_002 fails again
    rc2 = run(root, out, base_cfg())
    assert rc2 == 3
    with open(os.path.join(out, "manifest.json")) as fh:
        man2 = json.load(fh)
    assert next(p for p in man2["plots"] if p["plot_uid"] == "plot_001")["status"] \
        == "skipped_existing"


def t_area_normalized_labels_and_chm():
    root = make_run_dir()
    out = os.path.join(TMP, "out2")
    cfg = base_cfg(subsample="fps", n_fps=64, voxel_res=None,
                   min_points_pad=None, normalize_labels_by_area=True,
                   include_chm=True)
    rc = run(root, out, cfg)
    assert rc == 3
    with open(os.path.join(out, "labels", "plot_001.json")) as fh:
        lab = json.load(fh)
    assert lab["normalized_by_area"] is True
    assert np.isclose(lab["labels"]["NUM_TREES"], 4 / 100.0)
    assert np.isclose(lab["labels"]["BASAL_AREA"], 0.005)
    with np.load(os.path.join(out, "features", "plot_001.npz")) as z:
        assert z["points"].shape == (64, 3)
        assert "chm" in z and z["chm"].shape == (3, 4)
        meta = json.loads(str(z["meta"]))
        assert meta["rasters"]["chm"]["crs"] == "EPSG:32610"


def t_drop_excludes_plot():
    root = make_run_dir()
    out = os.path.join(TMP, "out3")
    cfg = base_cfg(subsample="none", voxel_res=None, min_points_drop=1000,
                   min_points_pad=None)
    rc = run(root, out, cfg)
    assert rc == 3  # plot_002 still fails
    with open(os.path.join(out, "manifest.json")) as fh:
        man = json.load(fh)
    rec = next(p for p in man["plots"] if p["plot_uid"] == "plot_001")
    assert rec["status"] == "dropped" and "MIN_POINTS_DROP" in rec["reason"]
    assert not os.path.exists(os.path.join(out, "features", "plot_001.npz"))


def t_config_error_exit_code():
    root = make_run_dir()
    out = os.path.join(TMP, "out4")
    rc = run(root, out, base_cfg(min_points_drop=500, min_points_pad=100))
    assert rc == 2, rc  # MIN_POINTS_DROP > MIN_POINTS_PAD


def t_source_auto_fallback():
    root = os.path.join(TMP, "run_auto")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 5, size=(50, 3))
    sem = np.ones(50, dtype=np.int8)
    write_plot(root, "plot_010", xyz, sem, sources=("raw",))  # only raw present
    plots = discover_plots(root)
    assert plots[0].cloud_path("auto").endswith("plot_010_raw.laz")
    out = os.path.join(TMP, "out_auto")
    rc = run(root, out, base_cfg(source="auto", subsample="none", voxel_res=None,
                                 min_points_drop=None, min_points_pad=None))
    assert rc == 0, rc
    with open(os.path.join(out, "manifest.json")) as fh:
        assert json.load(fh)["plots"][0]["source"] == "raw"


def t_stop_on_error_exit_code():
    root = make_run_dir()
    out = os.path.join(TMP, "out5")
    rc = run(root, out, base_cfg(stop_on_error=True))
    assert rc == 2, rc  # plot_002 fails -> batch aborted with exit 2
    with open(os.path.join(out, "manifest.json")) as fh:
        man = json.load(fh)
    assert man["plots"][-1]["status"] == "failed"


def t_empty_cloud_dropped():
    root = os.path.join(TMP, "run_empty")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    write_plot(root, "plot_020", np.zeros((0, 3)), np.zeros(0, dtype=np.int8))
    out = os.path.join(TMP, "out_empty")
    rc = run(root, out, base_cfg(subsample="none", voxel_res=None,
                                 min_points_drop=5, min_points_pad=None))
    assert rc == 0, rc
    with open(os.path.join(out, "manifest.json")) as fh:
        rec = json.load(fh)["plots"][0]
    assert rec["status"] == "dropped" and "empty" in rec["reason"], rec
    assert not os.path.exists(os.path.join(out, "features", "plot_020.npz"))


def t_integration_plot_demo():
    demo = os.path.join(PLAUSIBILITREE, "examples", "plot_demo")
    if not os.path.isdir(demo):
        print("SKIP integration: plausibilitree/examples/plot_demo not found")
        return
    out = os.path.join(TMP, "out_demo")
    shutil.rmtree(out, ignore_errors=True)
    cfg = base_cfg(subsample="voxel", voxel_res=0.5, min_points_drop=None,
                   min_points_pad=None, include_chm=True, include_dtm=True)
    rc = run(demo, out, cfg)
    assert rc == 0, rc
    with open(os.path.join(out, "labels", "plot_demo.json")) as fh:
        lab = json.load(fh)
    info = json.load(open(os.path.join(demo, "plot_placement", "textual",
                                      "plot_demo_info.json")))
    assert lab["labels"]["NUM_TREES"] == info["NUM_TREES"] == 4
    assert np.isclose(lab["labels"]["VOL_TREES"], info["VOL_TREES"])
    with np.load(os.path.join(out, "features", "plot_demo.npz")) as z:
        pts = z["points"]
        meta = json.loads(str(z["meta"]))
        assert meta["source"] == "understory"
        assert pts.shape[0] == z["semantic_code"].shape[0] > 0
        # voxel centers may overshoot the normalized range by up to res/2
        assert pts.min() >= -1.0 - 0.25 - 1e-6 and pts.max() <= 1.0 + 0.25 + 1e-6
        assert meta["bounding_box_info"] == info["BOUNDING_BOX"]
        assert "chm" in z and "dtm" in z
    with open(os.path.join(out, "manifest.json")) as fh:
        man = json.load(fh)
    assert man["summary"]["ok"] == 1 and man["plots"][0]["n_points_in"] > \
        man["plots"][0]["n_points_out"]  # voxel downsample shrank the cloud


def main() -> int:
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("t_") and callable(v)}.items()):
        check(name, fn)
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
