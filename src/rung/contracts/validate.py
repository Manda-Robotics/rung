"""Contract validation. Raise early, with a message that names the field and the rule.

Adapters call ``validate_observation`` on every reset and on the first N steps; the
mock embodiment calls it on every step. The harness calls ``validate_target`` before
handing a target to an adapter.
"""

from __future__ import annotations

import numpy as np

from rung.contracts.action import TCPTarget
from rung.contracts.capabilities import EmbodimentCapabilities
from rung.contracts.frames import CAMERA_CONVENTIONS
from rung.contracts.observation import CameraFrame, Observation


class ContractError(ValueError):
    pass


def _finite(name: str, a) -> None:
    if not np.all(np.isfinite(a)):
        raise ContractError(f"{name} contains non-finite values")


def validate_camera(frame: CameraFrame, caps: EmbodimentCapabilities | None = None) -> None:
    rgb = frame.rgb
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ContractError(f"{frame.name}.rgb must be HxWx3 uint8, got {rgb.shape} {rgb.dtype}")
    if frame.convention not in CAMERA_CONVENTIONS:
        raise ContractError(f"{frame.name}.convention {frame.convention!r} unknown")
    if frame.convention not in ("opencv", "ros"):
        raise ContractError(
            f"{frame.name}.convention must be 'opencv' inside the harness; "
            "adapters convert with frames.convert_camera_pose"
        )
    if frame.depth is not None:
        d = frame.depth
        if d.shape != rgb.shape[:2]:
            raise ContractError(f"{frame.name}.depth shape {d.shape} != rgb shape {rgb.shape[:2]}")
        if d.dtype != np.float32:
            raise ContractError(f"{frame.name}.depth must be float32 metres, got {d.dtype}")
        valid = np.isfinite(d)
        if valid.any() and np.nanmin(d[valid]) < 0:
            raise ContractError(f"{frame.name}.depth has negative values")
        if frame.K is None or frame.pose is None:
            raise ContractError(f"{frame.name} has depth but no K or pose; geometry is unusable")
    if frame.K is not None:
        k = np.asarray(frame.K)
        if k.shape != (3, 3):
            raise ContractError(f"{frame.name}.K must be 3x3")
        _finite(f"{frame.name}.K", k)
        if k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ContractError(f"{frame.name}.K focal lengths must be positive")
        if not (0 < k[0, 2] < rgb.shape[1] and 0 < k[1, 2] < rgb.shape[0]):
            raise ContractError(
                f"{frame.name}.K principal point {k[0, 2]:.1f},{k[1, 2]:.1f} outside the image "
                f"{rgb.shape[1]}x{rgb.shape[0]}; K must match the delivered resolution"
            )
    if frame.pose is not None:
        _finite(f"{frame.name}.pose.pos", frame.pose.pos)
    if caps is not None:
        spec = caps.camera(frame.name)
        if (spec.width, spec.height) != (frame.width, frame.height):
            raise ContractError(
                f"{frame.name} delivered {frame.width}x{frame.height}, capabilities declare "
                f"{spec.width}x{spec.height}"
            )
        if spec.has_depth and frame.depth is None:
            raise ContractError(f"{frame.name} declares depth but delivered none")


def validate_observation(obs: Observation, caps: EmbodimentCapabilities | None = None) -> None:
    if not np.isfinite(obs.t):
        raise ContractError("observation.t must be finite")
    validate_camera(obs.scene, caps)
    if obs.wrist is not None:
        validate_camera(obs.wrist, caps)
    _finite("tcp.pos", obs.tcp.pos)
    g = obs.gripper
    if not (0.0 <= g.travel <= 1.0):
        raise ContractError(f"gripper.travel must be in [0, 1], got {g.travel}")
    if g.opening_m is not None and g.opening_m < 0:
        raise ContractError("gripper.opening_m must be non-negative")
    if caps is not None:
        if not caps.workspace.contains(obs.tcp.pos, margin=-0.05):
            raise ContractError(
                f"tcp {np.round(obs.tcp.pos, 3).tolist()} is outside the declared workspace "
                "by more than 5 cm; check the TCP transform and root frame"
            )
        if obs.scene.t > obs.t + 1e-6:
            raise ContractError("scene capture time is after observation time")


def validate_target(target: TCPTarget, caps: EmbodimentCapabilities) -> None:
    _finite("target.pose.pos", target.pose.pos)
    _finite(
        "target timing and speeds",
        [target.timeout_s, target.max_lin_speed_mps, target.max_ang_speed_rps, target.min_dwell_s],
    )
    if target.min_dwell_s < 0:
        raise ContractError("target.min_dwell_s must be nonnegative")
    if target.timeout_s <= 0:
        raise ContractError("target.timeout_s must be positive")
    if target.max_lin_speed_mps <= 0 or target.max_ang_speed_rps <= 0:
        raise ContractError("target speeds must be positive")
    if target.max_lin_speed_mps > caps.max_lin_speed_mps + 1e-9:
        raise ContractError(
            f"target.max_lin_speed_mps {target.max_lin_speed_mps} exceeds embodiment cap "
            f"{caps.max_lin_speed_mps}"
        )
    if target.max_ang_speed_rps > caps.max_ang_speed_rps + 1e-9:
        raise ContractError(
            f"target.max_ang_speed_rps {target.max_ang_speed_rps} exceeds embodiment cap "
            f"{caps.max_ang_speed_rps}"
        )
    if "tcp_target" not in caps.command_kinds:
        raise ContractError(f"embodiment {caps.name} does not accept tcp_target commands")
    gripper = caps.gripper
    if target.arm is not None or target.paired:
        if not caps.arms:
            raise ContractError(f"embodiment {caps.name} has one arm; targets name no arm")
        names = [target.arm or caps.arms[0].name, *(p.arm for p in target.paired)]
        if any(n is None for n in names) or len(set(names)) != len(names):
            raise ContractError("paired targets need distinct, named arms")
        try:
            gripper = caps.arm(names[0]).gripper
            for p in target.paired:
                if p.paired:
                    raise ContractError("paired targets do not nest")
                validate_target(p, caps)
        except KeyError as exc:
            raise ContractError(f"embodiment {caps.name} has no arm {exc}") from None
    g = target.gripper
    if g.opening_m is not None and not (0 <= g.opening_m <= gripper.max_opening_m + 1e-9):
        raise ContractError(f"gripper opening {g.opening_m} outside [0, {gripper.max_opening_m}]")
