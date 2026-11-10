from copy import deepcopy

import numpy as np
import pytest

from rung.adapters.robolab.matched_start import audit_state, load_start, overlay_state

h5py = pytest.importorskip("h5py")


def fixture(path):
    with h5py.File(path, "w") as f:
        d = f.create_group("data/demo_0/initial_state")
        d.create_dataset("articulation/robot/joint_position", data=np.zeros((2, 3)))
        d.create_dataset(
            "rigid_object/banana/root_pose", data=np.tile([0.0, 0, 0, 1, 0, 0, 0], (2, 1))
        )
        d.create_dataset("cameras/scene/position", data=np.zeros((2, 3)))
        d.create_dataset("cameras/scene/orientation", data=np.tile([1, 0, 0, 0], (2, 1)))


def test_load_exact_demo_and_keep_hidden_state_outside_policy(tmp_path):
    path = tmp_path / "run_0.hdf5"
    fixture(path)
    start = load_start(path, "demo_0")
    assert start.provenance["initial_snapshot_count"] == 2
    assert len(start.provenance["source_file_sha256"]) == 64
    assert "cameras" not in start.state
    assert "scene" in start.cameras
    actual = deepcopy(start.state)
    actual["rigid_object"]["table"] = {"root_pose": np.zeros((1, 7))}
    restored = overlay_state(actual, start)
    assert "table" in restored["rigid_object"]
    restored["rigid_object"]["banana"]["root_pose"][..., 3:] *= -1
    assert audit_state(start.state, restored)["passed"]
    restored["rigid_object"]["banana"]["root_pose"][0, 0] = 0.01
    assert not audit_state(start.state, restored)["passed"]


def test_no_fallback_or_skipping_unknown_assets(tmp_path):
    path = tmp_path / "run_0.hdf5"
    fixture(path)
    with pytest.raises(ValueError, match="fallback"):
        load_start(path, "demo_99")
    start = load_start(path, "demo_0")
    with pytest.raises(ValueError, match="missing"):
        overlay_state({"articulation": start.state["articulation"]}, start)
    with h5py.File(path, "a") as f:
        f["data/demo_0/initial_state/articulation/robot/joint_position"][1, 0] = 0.1
    with pytest.raises(ValueError, match="differing"):
        load_start(path, "demo_0")


def test_export_roundtrip_tolerance_is_explicit_and_bounded(tmp_path):
    path = tmp_path / "run_0.hdf5"
    fixture(path)
    with h5py.File(path, "a") as f:
        f["data/demo_0/initial_state/cameras/scene/position"][1, 0] = 4e-7
    with pytest.raises(ValueError, match="differing"):
        load_start(path, "demo_0")
    start = load_start(path, "demo_0", duplicate_tolerance=1e-6)
    assert start.provenance["snapshot_max_abs_difference"] == pytest.approx(4e-7)
    with pytest.raises(ValueError, match="at most"):
        load_start(path, "demo_0", duplicate_tolerance=0.01)
