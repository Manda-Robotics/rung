"""The action contract: what the harness asks an embodiment to do, and what it reports back.

The core speaks TCP targets only. An adapter translates a target into relative IK
steps, Cartesian impedance commands or joint trajectories, plays it out for at most
``timeout_s`` or until a newer target supersedes it, and reports what actually happened.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from rung.contracts.frames import Pose


class GripperAction(StrEnum):
    OPEN = "open"
    CLOSE = "close"
    HOLD = "hold"


@dataclass(frozen=True)
class GripperCommand:
    action: GripperAction = GripperAction.HOLD
    opening_m: float | None = None  # target jaw gap for OPEN; None means fully open


@dataclass(frozen=True)
class TCPTarget:
    """An absolute TCP pose in the root frame, reached along a straight line at bounded speed.

    The timeout includes gripper actuation as well as motion. REACHED describes command
    execution; an obstructed close does not establish that an object is attached.
    """

    pose: Pose
    gripper: GripperCommand = field(default_factory=GripperCommand)
    max_lin_speed_mps: float = 0.10
    max_ang_speed_rps: float = 0.50
    timeout_s: float = 5.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    meta: dict = field(default_factory=dict)
    """Free-form provenance: interface level, primitive, model call id. Never read by adapters."""
    min_dwell_s: float = 0.0
    """Minimum simulated seconds after first pose arrival. Included in timeout; zero is legacy."""
    arm: str | None = None
    """Which arm of a multi-arm rig this pose is for (``ArmSpec.name``). None means the
    only arm, or the primary arm of a multi-arm rig. Arms without a goal hold still."""
    paired: tuple[TCPTarget, ...] = ()
    """Goals for other arms played out concurrently with this one, each with its own
    ``arm``. The pair reaches when every goal has; empty for one arm at a time."""


class ExecutionOutcome(StrEnum):
    REACHED = "reached"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    CLAMPED = "clamped"
    REJECTED = "rejected"
    BLOCKED = "blocked"


@dataclass
class ExecutionReport:
    """What the adapter did with a target. Attempted and executed are logged separately
    so controller interventions are visible."""

    target_id: str
    outcome: ExecutionOutcome
    t_start: float
    t_end: float
    final_tcp: Pose
    residual_m: float
    residual_axis: str  # "x" | "y" | "z" | "rot"
    residual_rot_rad: float = 0.0
    executed_lin_m: float = 0.0
    executed_rot_rad: float = 0.0
    clamped_to: Pose | None = None
    reason: str | None = None
    gripper_travel_end: float | None = None
    waited_s: float = 0.0
    arms: dict[str, dict] | None = None
    """Per-arm receipts on a multi-arm rig: outcome, residuals, final TCP (lists) and jaw
    travel for every arm the target addressed. The top-level fields describe the primary
    goal; None for one arm."""
    """Real seconds spent on the harness (perception, model, planning) before this motion
    started. In real-time mode the robot held still for this long first."""

    @property
    def duration_s(self) -> float:
        return float(self.t_end - self.t_start)


def residual_of(target: TCPTarget, actual: Pose) -> tuple[float, str, float]:
    """Largest remaining positional offset (m) and its axis, plus rotation residual (rad)."""
    from rung.contracts.frames import quat_angle_between

    diff = np.abs(np.asarray(actual.pos) - np.asarray(target.pose.pos))
    i = int(np.argmax(diff))
    rot = quat_angle_between(actual.quat, target.pose.quat)
    return float(diff[i]), "xyz"[i], float(rot)
