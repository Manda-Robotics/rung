"""Reject missing or misaligned controls rather than implying a usable trajectory."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def reader():
    path = Path(__file__).resolve().parents[1] / "examples/read_robolab.py"
    spec = importlib.util.spec_from_file_location("native_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_reader_reports_aligned_controls_and_absence_of_images(tmp_path):
    path = tmp_path / "run.hdf5"
    with h5py.File(path, "w") as file:
        group = file.create_group("data/demo_0")
        for key, width in (
            ("actions", 7),
            ("ee_pose/position", 3),
            ("ee_pose/orientation", 4),
            ("states/articulation/robot/joint_position", 13),
        ):
            group.create_dataset(key, data=np.zeros((3, width), dtype=np.float32))
    result = reader().inspect(path)
    assert result["control_steps"] == 3
    assert result["image_datasets"] == []
    assert result["arrays"]["actions"]["shape"] == [3, 7]
    with h5py.File(path, "a") as file:
        del file["data/demo_0/ee_pose/position"]
        file.create_dataset("data/demo_0/ee_pose/position", data=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="control-step rows"):
        reader().inspect(path)
