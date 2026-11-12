"""Two-arm bridge: one harness ``Embodiment`` over a simulator that asks for actions.

The same inversion as ``rung.adapters.robolab.bridge``: the simulator's loop calls
``BimanualDriver.get_action(raw_obs)`` for every control step, the harness loop calls
``BimanualEmbodiment.execute(target)`` and blocks until that target is played out. They
meet at two queues; only one side runs at a time.

What differs from the single-arm bridge:

- Every arm keeps a Cartesian **setpoint**. An arm with a goal walks its setpoint toward
  the goal at the target's speed; every other arm keeps its setpoint, so it holds still.
  A new goal for an arm restarts its setpoint from the measured TCP, so a blocked arm
  does not wind up.
- A target addresses one arm (``TCPTarget.arm``, None meaning the primary arm) plus any
  ``paired`` goals for other arms, all played out concurrently. The target finishes when
  every goal has arrived and settled, when any goal stalls, or at the deadline.
- How a setpoint becomes the rig's action (IK to joint positions for YAM, an end-effector
  pose for RoboDojo) is the ``RigCodec``'s job, so this module never sees a rig layout.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from rung.contracts.action import ExecutionOutcome, ExecutionReport, GripperAction, TCPTarget
from rung.contracts.capabilities import EmbodimentCapabilities
from rung.contracts.frames import Pose, quat_angle_between, quat_from_axis_angle, quat_mul
from rung.contracts.observation import Observation
from rung.contracts.validate import validate_target
from rung.embodiment import EpisodeStopped

TIME_MODES = ("realtime", "paused")
ARRIVED_M = 0.006
ARRIVED_RAD = 0.05
STALL_S = 1.5
STALL_PROGRESS_M = 0.002
STALL_PROGRESS_RAD = 0.02
JAW_STILL_S = 0.3
JAW_STILL_TRAVEL = 0.01
JAW_MIN_SETTLE_S = 0.4
"""Same meaning as the single-arm bridge: a grip has settled after the gripper's full
close time, or once the jaw has been commanded for JAW_MIN_SETTLE_S and changed by less
than JAW_STILL_TRAVEL over JAW_STILL_S. RoboDojo reports the commanded rather than the
measured gripper value, so there the full close time is what ends the wait."""


class EpisodeFinished(EpisodeStopped):
    """Raised on the harness side once the simulator side has closed the episode."""


class RigCodec(Protocol):
    """Rig-specific translation. Implemented by the YAM and RoboDojo adapters."""

    capabilities: EmbodimentCapabilities

    def observation(self, raw: Any, t: float) -> Observation:
        """One control step's raw observation as an ``Observation`` with ``arms`` set."""

    def command(
        self, observation: Observation, setpoints: dict[str, Pose], grips: dict[str, float]
    ) -> Any:
        """One control step's action: every arm's TCP setpoint and grip travel (0 open,
        1 closed), in the rig's own action format."""


def step_toward(current: Pose, goal: Pose, max_lin: float, max_rot: float) -> Pose:
    """Move ``current`` toward ``goal`` by at most ``max_lin`` metres and ``max_rot`` rad."""
    delta = np.asarray(goal.pos) - np.asarray(current.pos)
    distance = float(np.linalg.norm(delta))
    pos = goal.pos if distance <= max_lin else current.pos + delta * (max_lin / distance)
    angle = quat_angle_between(current.quat, goal.quat)
    if angle <= max_rot or angle < 1e-9:
        return Pose(pos, goal.quat)
    # Relative rotation current -> goal, scaled down to max_rot.
    q_rel = quat_mul(goal.quat, current.quat * np.array([1, -1, -1, -1]))
    if q_rel[0] < 0:
        q_rel = -q_rel
    axis = q_rel[1:] / max(np.linalg.norm(q_rel[1:]), 1e-12)
    return Pose(pos, quat_mul(quat_from_axis_angle(axis, max_rot), current.quat))


@dataclass
class _Goal:
    arm: str
    target: TCPTarget
    grip: float
    settle_s: float
    actuation_t: float | None = None
    best_lin: float = float("inf")
    best_rot: float = float("inf")
    progress_t: float | None = None
    jaw: list = field(default_factory=list)
    status: dict | None = None


@dataclass
class _Pending:
    target: TCPTarget
    goals: list[_Goal]
    start: Observation
    steps_left: int
    waited_s: float = 0.0
    hold_steps_left: int = 0


class BimanualEmbodiment:
    """Harness side: an ``Embodiment`` whose ``execute`` blocks on the simulator side."""

    def __init__(self, capabilities: EmbodimentCapabilities, time_mode: str = "realtime"):
        if not capabilities.arms:
            raise ValueError("a bimanual embodiment declares its arms")
        if time_mode not in TIME_MODES:
            raise ValueError(f"time_mode must be one of {TIME_MODES}")
        self.caps = capabilities
        self.time_mode = time_mode
        self.instruction = ""
        self._to_sim: queue.Queue = queue.Queue()
        self._to_harness: queue.Queue = queue.Queue()
        self._latest: Observation | None = None
        self._closed = threading.Event()
        self._offered_at: float | None = None
        self._lock = threading.Lock()

    @property
    def capabilities(self) -> EmbodimentCapabilities:
        return self.caps

    def reset(self, episode_seed: int | None = None) -> Observation:
        return self.observe()

    def observe(self) -> Observation:
        if self._latest is None:
            result = self._to_harness.get()
            if isinstance(result, BaseException):
                raise result
            self._latest = result
        return self._latest

    def execute(self, target: TCPTarget) -> ExecutionReport:
        validate_target(target, self.caps)
        start = self.observe()
        if self._closed.is_set():
            raise EpisodeFinished("embodiment is closed")
        primary = self.caps.arms[0].name
        goals = []
        for t in (target, *target.paired):
            arm = t.arm or primary
            state = start.arms[arm]
            grip = {GripperAction.CLOSE: 1.0, GripperAction.OPEN: 0.0}.get(t.gripper.action)
            hold = grip is None
            goals.append(
                _Goal(
                    arm,
                    t,
                    float(state.gripper.travel) if hold else grip,
                    0.0 if hold else self.caps.arm(arm).gripper.close_time_s,
                )
            )
        dt = self.caps.control_period_s
        pending = _Pending(
            target, goals, start, steps_left=max(0, int(np.floor(target.timeout_s / dt + 1e-9)))
        )
        self._to_sim.put(pending)
        result = self._to_harness.get()
        if isinstance(result, BaseException):
            raise result
        self._latest, stalled = result
        report = _report(pending, self._latest)
        report.waited_s = pending.waited_s
        return report

    def close(self) -> None:
        self._closed.set()
        self._to_sim.put(None)
        self._to_harness.put(EpisodeFinished("embodiment closed"))

    # ---- simulator side ----

    def _offer(self, item) -> None:
        with self._lock:
            self._offered_at = time.monotonic()
        self._to_harness.put(item)

    def _next_pending(self) -> _Pending | None:
        pending = self._to_sim.get()
        with self._lock:
            offered, self._offered_at = self._offered_at, None
        if pending is not None and offered is not None:
            pending.waited_s = max(0.0, time.monotonic() - offered)
            if self.time_mode == "realtime":
                pending.hold_steps_left = int(round(pending.waited_s / self.caps.control_period_s))
        return pending

    def _fail(self, exc: BaseException) -> None:
        self._to_harness.put(exc)


class BimanualDriver:
    """Simulator side: called once per control step, returns that step's action."""

    def __init__(self, embodiment: BimanualEmbodiment, codec: RigCodec):
        self.emb, self.codec = embodiment, codec
        self.steps = 0
        self._pending: _Pending | None = None
        self._finished = False
        self.setpoints: dict[str, Pose] | None = None
        self.grips: dict[str, float] | None = None

    @property
    def t(self) -> float:
        return self.steps * self.emb.caps.control_period_s

    def get_action(self, raw: Any) -> Any:
        obs = self.codec.observation(raw, self.t)
        if self.setpoints is None:
            self.setpoints = {a: s.tcp for a, s in obs.arms.items()}
            self.grips = {a: float(s.gripper.travel) for a, s in obs.arms.items()}
        if not self._finished:
            self._advance(obs)
        self.steps += 1
        return self.codec.command(obs, self.setpoints, self.grips)

    def _advance(self, obs: Observation) -> None:
        while True:
            p = self._pending
            if p is None:
                self.emb._offer(obs)
            else:
                if p.hold_steps_left > 0:
                    p.hold_steps_left -= 1
                    return  # harness decision time plays out with every arm holding
                stalled = False
                complete = True
                for g in p.goals:
                    done, stuck = self._evaluate(g, obs, p.steps_left)
                    complete &= done
                    stalled |= stuck
                if p.steps_left > 0 and not complete and not stalled:
                    self._step_setpoints(p)
                    p.steps_left -= 1
                    return
                self.emb._offer((obs, stalled))
            self._pending = self.emb._next_pending()
            if self._pending is None:
                self._finished = True
                return
            for g in self._pending.goals:
                # A fresh goal restarts the arm's setpoint where the arm actually is.
                self.setpoints[g.arm] = obs.arms[g.arm].tcp

    def _evaluate(self, g: _Goal, obs: Observation, steps_left: int) -> tuple[bool, bool]:
        state = obs.arms[g.arm]
        lin = float(np.max(np.abs(state.tcp.pos - g.target.pose.pos)))
        rot = quat_angle_between(state.tcp.quat, g.target.pose.quat)
        arrived = lin <= ARRIVED_M and rot <= ARRIVED_RAD
        if arrived and g.actuation_t is None:
            g.actuation_t = obs.t
            self.grips[g.arm] = g.grip
        travel = float(state.gripper.travel)
        if g.actuation_t is not None:
            g.jaw.append((obs.t, travel))
        elapsed = None if g.actuation_t is None else obs.t - g.actuation_t + 1e-9
        settled = g.settle_s == 0 or (
            elapsed is not None
            and (elapsed >= g.settle_s or (elapsed >= JAW_MIN_SETTLE_S and _jaw_still(g, obs.t)))
        )
        dwelled = g.target.min_dwell_s == 0 or (
            elapsed is not None and elapsed >= g.target.min_dwell_s
        )
        if (
            g.progress_t is None
            or lin < g.best_lin - STALL_PROGRESS_M
            or rot < g.best_rot - STALL_PROGRESS_RAD
        ):
            g.best_lin, g.best_rot = min(lin, g.best_lin), min(rot, g.best_rot)
            g.progress_t = obs.t
        stalled = not arrived and steps_left > 0 and obs.t - g.progress_t + 1e-9 >= STALL_S
        g.status = {"arrived": arrived, "settled": settled and dwelled, "stalled": stalled}
        return arrived and settled and dwelled, stalled

    def _step_setpoints(self, p: _Pending) -> None:
        dt = self.emb.caps.control_period_s
        for g in p.goals:
            self.setpoints[g.arm] = step_toward(
                self.setpoints[g.arm],
                g.target.pose,
                g.target.max_lin_speed_mps * dt,
                g.target.max_ang_speed_rps * dt,
            )

    def report_error(self, exc: BaseException) -> None:
        self.emb._fail(exc)


def _jaw_still(g: _Goal, now: float) -> bool:
    recent = [v for t, v in g.jaw if now - t <= JAW_STILL_S + 1e-9]
    if len(recent) < 2 or now - g.jaw[0][0] + 1e-9 < JAW_STILL_S:
        return False
    return max(recent) - min(recent) < JAW_STILL_TRAVEL


def _report(p: _Pending, end: Observation) -> ExecutionReport:
    arms = {}
    for g in p.goals:
        start, final = p.start.arms[g.arm], end.arms[g.arm]
        diff = np.abs(np.asarray(final.tcp.pos) - np.asarray(g.target.pose.pos))
        axis = int(np.argmax(diff))
        rot = quat_angle_between(final.tcp.quat, g.target.pose.quat)
        status = g.status or {"arrived": False, "settled": False, "stalled": False}
        if status["arrived"] and status["settled"]:
            outcome, reason = ExecutionOutcome.REACHED, None
        elif status["stalled"]:
            outcome = ExecutionOutcome.BLOCKED
            reason = (
                f"the {g.arm} arm stopped {diff[axis] * 100:.1f} cm short on "
                f"{'xyz'[axis]} and made no progress for {STALL_S:.1f} s: something "
                "is in the way or the pose is out of this arm's reach; change the target"
            )
        else:
            outcome = ExecutionOutcome.EXPIRED
            if not status["arrived"]:
                reason = (
                    f"the {g.arm} arm ran out of time {diff[axis] * 100:.1f} cm short on "
                    f"{'xyz'[axis]} and {rot:.2f} rad from the requested orientation; "
                    "still converging, not blocked: ask again to continue"
                )
            else:
                reason = f"the {g.arm} gripper had not settled or the hold had not elapsed"
        arms[g.arm] = {
            "outcome": outcome,
            "residual_m": float(diff[axis]),
            "residual_axis": "xyz"[axis],
            "residual_rot_rad": rot,
            "reason": reason,
            "final_tcp_xyz_m": final.tcp.pos.tolist(),
            "final_tcp_quat_wxyz": final.tcp.quat.tolist(),
            "executed_lin_m": float(np.linalg.norm(final.tcp.pos - start.tcp.pos)),
            "gripper_travel_end": float(final.gripper.travel),
        }
    order = [ExecutionOutcome.BLOCKED, ExecutionOutcome.EXPIRED, ExecutionOutcome.REACHED]
    worst = min((a["outcome"] for a in arms.values()), key=order.index)
    first = arms[p.goals[0].arm]
    end_arm = end.arms[p.goals[0].arm]
    reasons = [a["reason"] for a in arms.values() if a["reason"]]
    return ExecutionReport(
        target_id=p.target.id,
        outcome=worst,
        t_start=p.start.t,
        t_end=end.t,
        final_tcp=end_arm.tcp,
        residual_m=max(a["residual_m"] for a in arms.values()),
        residual_axis=first["residual_axis"],
        residual_rot_rad=max(a["residual_rot_rad"] for a in arms.values()),
        executed_lin_m=first["executed_lin_m"],
        executed_rot_rad=quat_angle_between(
            p.start.arms[p.goals[0].arm].tcp.quat, end_arm.tcp.quat
        ),
        reason="; ".join(reasons) or None,
        gripper_travel_end=first["gripper_travel_end"],
        arms={k: {**v, "outcome": v["outcome"].value} for k, v in arms.items()},
    )


def run_in_thread(target, *args: Any, **kwargs: Any) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    return thread
