"""Inspect dense actions and robot state in a native RoboLab HDF5 recording.

Requires h5py: pip install h5py. No GPU or model key is needed to read a recording.
Videos and per-decision model inputs remain separate files; this is not a VLA dataset converter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def inspect(path: Path, demo: str = "demo_0") -> dict:
    import h5py

    with h5py.File(path, "r") as recording:
        group = recording[f"data/{demo}"]
        fields = {
            "actions": "actions",
            "tcp_position": "ee_pose/position",
            "tcp_orientation": "ee_pose/orientation",
            "joint_position": "states/articulation/robot/joint_position",
        }
        arrays = {name: np.asarray(group[key]) for name, key in fields.items()}
        count = len(arrays["actions"])
        if count == 0:
            raise ValueError("recording has no control steps")
        for name, array in arrays.items():
            if array.ndim != 2 or len(array) != count or not np.isfinite(array).all():
                raise ValueError(f"{name}: expected {count} finite control-step rows")
        images = []

        def find_images(name, obj):
            if isinstance(obj, h5py.Dataset) and any(
                word in name.lower() for word in ("rgb", "image")
            ):
                images.append(name)

        group.visititems(find_images)
        return {
            "demo": demo,
            "control_steps": count,
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                for name, array in arrays.items()
            },
            "image_datasets": images,
            "first_action": arrays["actions"][0].tolist(),
            "note": "Read the run config for action space, units and control rate. "
            "Videos and model-decision images are separate; this reader does not "
            "align or decode them. Use the native JSON result for the task grade.",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--demo", default="demo_0")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.recording, args.demo), indent=2))
    except ImportError:
        parser.exit(2, "Install the optional reader dependency: pip install h5py\n")
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f"Cannot read recording: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
