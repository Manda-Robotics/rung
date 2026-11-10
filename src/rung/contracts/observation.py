"""The observation contract: what an embodiment adapter hands the harness each step.

Everything is per environment and flat. Every field names its frame and unit; see
``frames.py`` for the conventions. Adapters do all conversion; the core never guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from rung.contracts.frames import Pose


@dataclass
class CameraFrame:
    """One camera's capture, with the geometry needed to use it.

    ``depth`` is optical-axis depth in metres, ``float32``, ``NaN`` where invalid.
    ``K`` is valid for ``rgb`` at the delivered resolution (and ``depth``, which must
    share it). ``pose`` is the camera in the robot root frame, OpenCV axes.
    ``t`` is the capture time in simulated seconds. ``state_at_capture`` is the TCP
    pose at the moment the image was taken, so a decision made on an old frame can
    be reasoned about.
    """

    name: str
    rgb: npt.NDArray[np.uint8]
    t: float
    depth: npt.NDArray[np.float32] | None = None
    K: npt.NDArray[np.float64] | None = None
    pose: Pose | None = None
    convention: str = "opencv"
    state_at_capture: Pose | None = None

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def has_geometry(self) -> bool:
        return self.depth is not None and self.K is not None and self.pose is not None


@dataclass
class GripperState:
    """``travel`` is 0 fully open, 1 fingers touching. ``opening_m`` is the jaw gap when
    the embodiment reports it; otherwise derived by the adapter from travel and the
    gripper geometry. ``velocity`` is d(travel)/dt; ``effort`` is normalised 0..1 or None."""

    travel: float
    opening_m: float | None = None
    velocity: float | None = None
    effort: float | None = None


@dataclass
class JointState:
    positions: npt.NDArray[np.float64]
    velocities: npt.NDArray[np.float64] | None = None
    names: tuple[str, ...] | None = None


@dataclass
class ArmState:
    """Measured state of one arm of a multi-arm rig, in the shared root frame."""

    tcp: Pose
    gripper: GripperState
    wrist: CameraFrame | None = None


@dataclass
class Observation:
    """One step's observation.

    ``tcp`` is the tool centre point: the fingertip midpoint with the closing axis as the
    local ``+y`` and the approach direction as local ``+z``. The transform from the
    embodiment's reported end-effector frame to this TCP lives in the capabilities and
    is applied by the adapter.
    """

    t: float
    scene: CameraFrame
    tcp: Pose
    gripper: GripperState
    wrist: CameraFrame | None = None
    joints: JointState | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    arms: dict[str, ArmState] | None = None
    """Per-arm state on a multi-arm rig, keyed by ``ArmSpec.name``; None for one arm.
    ``tcp``, ``gripper`` and ``wrist`` then mirror the primary arm."""

    @property
    def cameras(self) -> dict[str, CameraFrame]:
        out = {self.scene.name: self.scene}
        if self.wrist is not None:
            out[self.wrist.name] = self.wrist
        for arm in (self.arms or {}).values():
            if arm.wrist is not None:
                out[arm.wrist.name] = arm.wrist
        return out
