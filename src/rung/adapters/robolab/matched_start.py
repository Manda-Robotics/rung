"""Strict HDF5 start-state loading and audit, outside the policy information boundary.

Repeated initial snapshots belong to one demo. Never silently choose among differing
snapshots, skip an unknown asset, or fall back to an ordinary random reset.
The simulator-facing restore is qualified separately before benchmark use.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def flatten(tree, prefix=""):
    result = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, path))
        else:
            result[path] = value
    return result


def _nest(flat):
    tree = {}
    for path, value in flat.items():
        current = tree
        keys = path.split("/")
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
    return tree


@dataclass
class MatchedStart:
    state: dict
    cameras: dict
    provenance: dict


def load_start(path: Path, demo: str, *, duplicate_tolerance: float = 0.0) -> MatchedStart:
    import h5py

    path = Path(path).resolve()
    if not np.isfinite(duplicate_tolerance) or not 0 <= duplicate_tolerance <= 1e-5:
        raise ValueError("duplicate tolerance must be finite, nonnegative and at most 1e-5")
    before = path.stat()
    file_hash = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            file_hash.update(chunk)
    flat = {}
    with h5py.File(path, "r") as f:
        if f"data/{demo}/initial_state" not in f:
            raise ValueError("recorded initial state missing; fallback is forbidden")
        f[f"data/{demo}/initial_state"].visititems(
            lambda n, o: flat.update({n: o[()]}) if isinstance(o, h5py.Dataset) else None
        )
        runtime = {
            k: str(f["data"].attrs[k])
            for k in ("isaaclab_version", "isaacsim_version")
            if k in f["data"].attrs
        }
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source recording changed during loading; use a stable mirror")
    if (
        not flat
        or not any(p.startswith("articulation/") for p in flat)
        or not any(p.startswith("rigid_object/") for p in flat)
    ):
        raise ValueError("robot and object state required")
    sizes = {a.shape[0] for a in flat.values() if a.ndim}
    if len(sizes) != 1 or not sizes or any(a.ndim == 0 for a in flat.values()):
        raise ValueError("ambiguous initial-state shapes")
    count = next(iter(sizes))
    if count < 1:
        raise ValueError("initial state empty")
    rows = {}
    digest = hashlib.sha256()
    snapshot_difference = 0.0
    for name, values in sorted(flat.items()):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite initial state: {name}")
        difference = float(np.max(np.abs(values - values[0])))
        snapshot_difference = max(snapshot_difference, difference)
        if difference > duplicate_tolerance:
            raise ValueError(f"differing initial snapshots: {name}")
        row = values[0]
        digest.update(
            json.dumps([name, row.dtype.str, list(row.shape)], separators=(",", ":")).encode()
            + b"\0"
            + row.tobytes()
        )
        rows[name] = row[None].copy()
    tree = _nest(rows)
    cameras = tree.pop("cameras", {})
    return MatchedStart(
        tree,
        cameras,
        {
            "source_file": str(path),
            "source_demo": demo,
            "source_task": path.parent.name,
            "source_file_sha256": file_hash.hexdigest(),
            "source_runtime": runtime,
            "initial_state_sha256": digest.hexdigest(),
            "initial_snapshot_count": count,
            "duplicate_tolerance": duplicate_tolerance,
            "snapshot_max_abs_difference": snapshot_difference,
            "status": "loaded_not_restored_or_qualified",
        },
    )


def overlay_state(current: dict, start: MatchedStart, convert=lambda a, template: a):
    """Preserve static entries; reject absent recorded assets/fields before any mutation."""
    recorded, actual = flatten(start.state), flatten(current)
    missing = sorted(set(recorded) - set(actual))
    if missing:
        raise ValueError(f"recorded fields missing in target scene: {missing}")
    for name, value in recorded.items():
        if tuple(value.shape) != tuple(actual[name].shape):
            raise ValueError(f"shape mismatch at {name}; matched restore requires one environment")
    result = copy.deepcopy(current)
    for name, value in recorded.items():
        keys = name.split("/")
        leaf = result
        for key in keys[:-1]:
            leaf = leaf[key]
        leaf[keys[-1]] = convert(value.copy(), actual[name])
    return result


def audit_state(expected: dict, actual: dict, tolerance: float = 1e-5):
    """Numerical audit, including quaternion sign equivalence for pose fields."""
    expected, actual = flatten(expected), flatten(actual)
    errors = {}
    for name, target in expected.items():
        if name not in actual:
            raise ValueError(f"restored field absent: {name}")
        value = actual[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        target, value = np.asarray(target), np.asarray(value)
        if target.shape != value.shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid restored field: {name}")
        value = value.copy()
        if name.endswith("orientation") and target.shape[-1] == 4:
            if float(np.sum(target * value)) < 0:
                value *= -1
        elif name.endswith("root_pose") and target.shape[-1] == 7:
            if float(np.sum(target[..., 3:] * value[..., 3:])) < 0:
                value[..., 3:] *= -1
        errors[name] = float(np.max(np.abs(target - value)))
    worst = max(errors.values(), default=float("inf"))
    return {
        "passed": worst <= tolerance,
        "tolerance": tolerance,
        "max_abs_error": worst,
        "field_errors": errors,
    }


def install_matched_reset(env, start: MatchedStart, audit_callback):
    """Instrument a single-env development runner without editing frozen evaluator code.

    Isaac's reset_to captures recorder initial state before it installs the target;
    refresh that recorder term at post-reset so exported HDF5 matches the actual start.
    A camera refresh precedes observation computation. Qualification must check both
    the resulting images and exported initial states before this is a benchmark path.
    """
    import importlib.metadata

    import torch
    from packaging.version import Version

    from rung.adapters.robolab.compat import isaaclab_compat

    _compat = isaaclab_compat()
    as_torch = _compat.as_torch
    quat_isaaclab_to_wxyz = _compat.quat_isaaclab_to_wxyz
    quat_wxyz_to_isaaclab = _compat.quat_wxyz_to_isaaclab
    scene_state_from_isaaclab = _compat.scene_state_from_isaaclab
    scene_state_to_isaaclab = _compat.scene_state_to_isaaclab

    if env.num_envs != 1:
        raise ValueError("matched reset is qualified for one environment only")
    for name, distribution in (("isaaclab_version", "isaaclab"), ("isaacsim_version", "isaacsim")):
        source = start.provenance["source_runtime"].get(name)
        if source is None or Version(source) != Version(importlib.metadata.version(distribution)):
            raise ValueError(f"recording/runtime mismatch: {name}")
    for name in start.cameras:
        if name not in env.scene.sensors:
            raise ValueError(f"recorded camera absent: {name}")
    initial_terms = [
        term
        for term in env.recorder_manager._terms.values()
        if term.__class__.__name__ == "InitialStateRecorder"
    ]
    if len(initial_terms) != 1:
        raise ValueError("expected exactly one native initial-state recorder")
    original_reset, original_post = env.reset, env.recorder_manager.record_post_reset
    restoring = False

    def post_reset(env_ids):
        if restoring:
            for name, values in start.cameras.items():
                camera = env.scene.sensors[name]
                camera_pos = as_torch(camera.data.pos_w)
                camera_quat = as_torch(camera.data.quat_w_ros)
                positions = torch.as_tensor(
                    values["position"], device=env.device, dtype=camera_pos.dtype
                )
                positions = positions + as_torch(env.scene.env_origins)
                orientations = torch.as_tensor(
                    values["orientation"], device=env.device, dtype=camera_quat.dtype
                )
                camera.set_world_poses(
                    positions, quat_wxyz_to_isaaclab(orientations), convention="ros"
                )
            env.sim.render()
            for name in start.cameras:
                env.scene.sensors[name].update(0.0, force_recompute=True)
            initial_terms[0].reset(env_ids=env_ids)
        original_post(env_ids)

    def reset(seed=None, env_ids=None, options=None):
        nonlocal restoring
        if env_ids is not None or options:
            raise ValueError("partial/options reset not supported by matched development profile")
        current = _nest(
            {
                k: as_torch(v)
                for k, v in flatten(
                    scene_state_from_isaaclab(env.scene.get_state(is_relative=True))
                ).items()
            }
        )
        state = overlay_state(
            current,
            start,
            lambda a, template: torch.as_tensor(a, device=template.device, dtype=template.dtype),
        )
        restoring = True
        try:
            result = env.reset_to(
                scene_state_to_isaaclab(state), env_ids=None, seed=seed, is_relative=True
            )
        finally:
            restoring = False
        actual = _nest(
            {
                k: as_torch(v)
                for k, v in flatten(
                    scene_state_from_isaaclab(env.scene.get_state(is_relative=True))
                ).items()
            }
        )
        state_audit = audit_state(start.state, actual)
        actual_cameras = {
            name: {
                "position": as_torch(camera.data.pos_w) - as_torch(env.scene.env_origins),
                "orientation": quat_isaaclab_to_wxyz(as_torch(camera.data.quat_w_ros)),
            }
            for name, camera in env.scene.sensors.items()
            if name in start.cameras
        }
        camera_audit = audit_state(start.cameras, actual_cameras)
        audit_callback(
            {
                **start.provenance,
                "state_audit": state_audit,
                "camera_audit": camera_audit,
                "status": "restored"
                if state_audit["passed"] and camera_audit["passed"]
                else "restore_failed",
            }
        )
        if not state_audit["passed"] or not camera_audit["passed"]:
            raise ValueError("matched start audit failed; no policy call permitted")
        return result

    env.reset, env.recorder_manager.record_post_reset = reset, post_reset

    def uninstall():
        env.reset, env.recorder_manager.record_post_reset = original_reset, original_post

    return uninstall
