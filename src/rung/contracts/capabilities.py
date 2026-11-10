"""What an embodiment can do and how it is built. Declared once by the adapter.

Nothing here is a tunable; these are facts about the rig that the harness needs to plan
grasps, check reach and render prompts accurately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from rung.contracts.frames import Pose

COMMAND_KINDS = ("tcp_target",)


@dataclass(frozen=True)
class GripperGeometry:
    """Parallel-jaw gripper dimensions, metres."""

    max_opening_m: float
    finger_length_m: float
    pad_width_m: float
    close_time_s: float = 1.0

    def opening_for_travel(self, travel: float) -> float:
        return float(max(0.0, min(1.0, 1.0 - travel)) * self.max_opening_m)


@dataclass(frozen=True)
class Workspace:
    """Axis-aligned reachable box in the root frame, plus the table plane height."""

    min_xyz: npt.NDArray[np.float64]
    max_xyz: npt.NDArray[np.float64]
    table_z: float
    table_extent: tuple[float, float, float, float] | None = None
    """Declared table-top footprint in the root frame as (x_min, x_max, y_min, y_max),
    from the rig's fixture geometry, not from perception. None when the rig has not
    declared one. The reachable box above is larger than the table on purpose."""

    def table_margin(self, p: npt.ArrayLike) -> float | None:
        """Signed distance from a root-frame point to the nearest table edge, in metres:
        positive inside the footprint, negative beyond it. None without an extent."""
        if self.table_extent is None:
            return None
        x, y = (float(v) for v in np.asarray(p, dtype=np.float64).reshape(3)[:2])
        x_min, x_max, y_min, y_max = self.table_extent
        return float(min(x - x_min, x_max - x, y - y_min, y_max - y))

    def contains(self, p: npt.ArrayLike, margin: float = 0.0) -> bool:
        q = np.asarray(p, dtype=np.float64).reshape(3)
        return bool(np.all(q >= self.min_xyz + margin) and np.all(q <= self.max_xyz - margin))


@dataclass(frozen=True)
class CameraSpec:
    name: str
    width: int
    height: int
    has_depth: bool
    mounted_on: str = "world"  # "world" or "tcp"


@dataclass(frozen=True)
class ArmSpec:
    """One arm of a multi-arm rig. Single-arm embodiments declare none.

    Each arm has its own reachable box, gripper and TCP transform, all in the shared rig
    root frame, so one target vocabulary addresses every arm. ``base`` is where the arm's
    own base frame sits in the rig root; it is a fact for audit and kinematics, never a
    frame the model is asked to use.
    """

    name: str
    workspace: Workspace
    gripper: GripperGeometry
    tcp_from_ee: Pose
    base: Pose
    wrist_camera: str | None = None
    """Role name of the camera riding this arm's gripper, as delivered in ``ArmState``."""


@dataclass(frozen=True)
class EmbodimentCapabilities:
    name: str
    cameras: tuple[CameraSpec, ...]
    gripper: GripperGeometry
    workspace: Workspace
    control_period_s: float
    tcp_from_ee: Pose
    """TCP expressed in the embodiment's reported end-effector frame. Adapters apply it;
    it is recorded here so a trace can be audited."""
    command_kinds: tuple[str, ...] = ("tcp_target",)
    has_joint_state: bool = False
    max_lin_speed_mps: float = 0.25
    max_ang_speed_rps: float = 1.0
    steps_per_efold: float = 3.0
    """Control steps this rig needs to shrink the remaining distance by a factor of e.

    An ideal servo needs about one; real controllers undershoot and converge
    geometrically. Sets how long a motion is allowed before it is reported blocked.
    Measure it per rig. RoboLab's DROID relative IK is about 30, so a 6 cm move takes
    roughly ninety control steps.
    """
    notes: dict[str, str] = field(default_factory=dict)
    arms: tuple[ArmSpec, ...] = ()
    """Per-arm declarations for a multi-arm rig, primary arm first. Empty for one arm.

    When set, the top-level ``gripper``, ``workspace`` and ``tcp_from_ee`` describe the
    primary arm, so code that knows only one arm keeps working on it."""

    def arm(self, name: str) -> ArmSpec:
        for a in self.arms:
            if a.name == name:
                return a
        raise KeyError(name)

    def time_to_converge(
        self, distance_m: float, tolerance_m: float = 0.002, safety: float = 2.0
    ) -> float:
        """How long to allow a motion of this length, in seconds.

        Geometric convergence never formally arrives, so the budget is the number of
        e-folds between ``distance_m`` and ``tolerance_m``, times ``steps_per_efold`` and
        ``safety``. ``safety`` defaults to 2 because ``steps_per_efold`` is measured on a
        bare servo loop and execution through the executor is slower. A short budget
        reports a reachable target as unreachable, which misleads the policy; a long one
        only costs simulated time.
        """
        import math

        distance_m = max(float(distance_m), 0.0)
        tolerance_m = max(float(tolerance_m), 1e-6)
        if distance_m <= tolerance_m:
            return max(4.0 * self.control_period_s, 0.5)
        efolds = math.log(distance_m / tolerance_m)
        return safety * self.steps_per_efold * efolds * self.control_period_s

    def camera(self, name: str) -> CameraSpec:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(name)
