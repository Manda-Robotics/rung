"""What a grasp planner proposes, and why it rejected what it did not propose.

Rejections are first-class. Silently dropped candidates tell the policy nothing; a reason
such as "too wide for the jaw at every height" lets it pick another object or approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from rung.contracts.frames import Pose


@dataclass
class GraspCandidate:
    """One way to grasp one object.

    ``pose`` is where the TCP must be to close, ``pre_pose`` is the standoff it approaches
    from, and the straight line between them is the approach the checks cleared.
    """

    candidate_id: str
    object_id: str
    approach: str  # "top" or "side"
    pose: Pose
    pre_pose: Pose
    width_m: float
    """The object's width along the closing direction at this grasp height."""
    opening_m: float
    """What the jaw must open to before closing. Wider than ``width_m`` by a margin."""
    height_m: float
    """Grasp height in the root frame."""
    clearance_m: float
    """Distance from the finger volume to the nearest point that is not this object."""
    score: float
    rejected: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.rejected is None

    def describe(self) -> str:
        head = (
            f"{self.candidate_id}: {self.approach} grasp at {self.height_m * 100:.0f} cm, "
            f"jaw across {self.width_m * 100:.1f} cm, clearance {self.clearance_m * 100:.0f} cm"
        )
        if self.rejected:
            return f"{head}, REJECTED: {self.rejected}"
        return head


@dataclass
class GraspPlan:
    """Everything the planner found for one object, feasible or not."""

    object_id: str
    candidates: list[GraspCandidate]
    planner: str
    version: str

    @property
    def feasible(self) -> list[GraspCandidate]:
        return [c for c in self.candidates if c.feasible]

    def best(self) -> GraspCandidate | None:
        ok = self.feasible
        return max(ok, key=lambda c: c.score) if ok else None

    def get(self, candidate_id: str) -> GraspCandidate | None:
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                return c
        return None

    def render(self, max_shown: int = 4) -> str:
        """Feasible candidates first, then why the others were not offered.

        Rejections are shown even when something is feasible: they name a measurement and a
        limit of the rig, which a policy can act on.
        """
        if not self.candidates:
            return f"No grasp candidates for {self.object_id}: no usable geometry."
        ok = sorted(self.feasible, key=lambda c: -c.score)[:max_shown]
        lines = [c.describe() for c in ok]
        reasons = sorted({c.rejected for c in self.candidates if c.rejected})
        if not ok:
            lines.append(f"No feasible grasp for {self.object_id}. Reasons: {'; '.join(reasons)}.")
        elif reasons:
            n = len(self.candidates) - len(self.feasible)
            lines.append(f"{n} other candidates were rejected: {'; '.join(reasons)}.")
        return "\n".join(lines)


def closing_yaw(direction: npt.ArrayLike) -> float:
    """Yaw a top-down TCP must take so its closing axis lies along ``direction``.

    The TCP's local +y is the closing axis. ``top_down_pose(pos, yaw)`` puts that axis at
    ``yaw - 90`` degrees in the root frame, so the yaw needed is the direction's angle plus
    90 degrees.
    """
    d = np.asarray(direction, dtype=np.float64).reshape(2)
    return float(np.arctan2(d[1], d[0]) + np.pi / 2)
