"""Policies for RoboDojo episodes: a scripted two-arm motion check, or the Direct profile.

``scripted_bimanual`` needs no model and is the first thing to run on a new install: its
printed lines show each arm's measured TCP and the receipts of the step before, so frames,
IK and the gripper mapping can be read off the log. ``direct_policy`` builds the bimanual
Direct interface from a profile JSON (``configs/gemini-4-argon-robodojo.json``) over a paid
model API, or over the scripted fake model when the profile's model is ``fake``.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from rung.contracts import GripperAction, GripperCommand, Pose, TCPTarget
from rung.policy import Decision, Verdict


class ScriptedBimanual:
    """Each arm alone, both together, then each gripper closes; logs what it measured."""

    interface = name = "scripted-bimanual"

    def __init__(self, lift_m: float = 0.05, forward_m: float = 0.04, log=print):
        self.lift_m, self.forward_m, self.log = lift_m, forward_m, log

    def reset(self, instruction, caps):
        self.caps, self.step = caps, 0

    def act(self, observation, feedback):
        tcp = {a: s.tcp for a, s in observation.arms.items()}
        self.log(
            "SCRIPTED",
            self.step,
            {a: p.pos.round(4).tolist() for a, p in tcp.items()},
            {a: round(s.gripper.travel, 3) for a, s in observation.arms.items()},
            "receipts",
            feedback.structured.get("execution"),
        )
        forward = self.caps.notes.get("forward_axis", "x")
        ahead = np.asarray({"x": [1.0, 0, 0], "y": [0, 1.0, 0]}[forward])

        def goal(arm, dz=0.0, ahead_m=0.0):
            pos = tcp[arm].pos + [0, 0, dz] + ahead_m * ahead
            return TCPTarget(Pose(pos, tcp[arm].quat), timeout_s=8, arm=arm)

        first, second = (a.name for a in self.caps.arms)
        close = GripperCommand(GripperAction.CLOSE)
        plan = [
            lambda: [goal(first, self.lift_m)],
            lambda: [goal(second, self.lift_m)],
            lambda: [
                replace(goal(first, 0, self.forward_m), paired=(goal(second, 0, self.forward_m),))
            ],
            lambda: [TCPTarget(tcp[first], close, timeout_s=4, arm=first)],
            lambda: [TCPTarget(tcp[second], close, timeout_s=4, arm=second)],
        ]
        if self.step >= len(plan):
            return Decision("done", "motion check complete", verdict=Verdict.DONE)
        targets = plan[self.step]()
        self.step += 1
        return Decision("scripted", f"step {self.step}", targets=targets)


def direct_policy(
    profile_path: str | Path,
    *,
    horizon_s: float,
    max_calls: int,
    trace_root: str | Path | None = None,
    api_key: str | None = None,
):
    """The bimanual Direct policy a profile names, for one RoboDojo episode.

    The model key comes from ``api_key`` or the provider's environment variable
    (``GEMINI_API_KEY``). ``time_mode`` is ``paused``: RoboDojo's ``step_lim`` counts
    actions, so the model's thinking time does not run the task's clock.
    """
    from rung.direct_profile import build_direct_policy, load_profile

    profile = load_profile(profile_path)
    if not profile.direct_config.bimanual:
        raise ValueError(f"{profile_path}: RoboDojo needs a profile with direct_config.bimanual")
    trace_dir = Path(trace_root) / f"episode-{time.time_ns()}" if trace_root else None
    return build_direct_policy(
        profile,
        api_key=api_key,
        trace_dir=trace_dir,
        budgets={"horizon_s": horizon_s, "max_model_calls": max_calls, "time_mode": "paused"},
    )
