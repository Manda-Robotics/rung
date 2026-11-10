"""Bridge between RoboLab's evaluation loop and the harness episode loop.

RoboLab's runner owns the episode: it steps the simulator and calls a client's
``get_action(obs)`` whenever it needs an action chunk. The harness also owns the episode:
``run_episode`` calls a policy, which calls primitives, which call ``execute`` and block on
the result. Both are blocking loops.

The two loops run in separate threads that meet at a pair of queues. The harness thread
blocks in ``execute`` until the simulator has played the motion out; the RoboLab thread
blocks in ``get_action`` until the harness has chosen the next target. Only one side runs
at a time, so the queues are the only synchronisation.

Rewriting the harness loop as a coroutine would push this inversion into every interface
level; the adapter confines it to one place.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from rung.adapters.robolab.translate import (
    SCENE_CAMERA,
    action_chunk,
    droid_capabilities,
    steps_needed,
    to_observation,
)
from rung.contracts.action import (
    ExecutionOutcome,
    ExecutionReport,
    GripperAction,
    TCPTarget,
)
from rung.contracts.capabilities import EmbodimentCapabilities
from rung.contracts.frames import quat_angle_between
from rung.contracts.observation import CameraFrame, Observation
from rung.contracts.validate import validate_target
from rung.embodiment import EpisodeStopped

TIME_MODES = ("realtime", "paused")

ARRIVED_M = 0.006
ARRIVED_RAD = 0.05
STALL_S = 1.5
JAW_MIN_SETTLE_S = 0.5
JAW_STILL_S = 0.2
JAW_STILL_TRAVEL = 0.005
JAW_MOVED_TRAVEL = 0.02
JAW_STARTED_TRAVEL = 0.1
"""A grip command is settled once the jaw stops, or after the gripper's full close time.
A jaw closing on an object typically stops well before the full close time. The jaw
counts as stopped once it has been commanded for JAW_MIN_SETTLE_S, has moved (or already
sits at the command), and has changed by less than JAW_STILL_TRAVEL over the last
JAW_STILL_S. A jaw that has not started moving is not stopped, so actuation delay cannot
end the wait early."""
STALL_PROGRESS_M = 0.002
STALL_PROGRESS_RAD = 0.02
"""A motion whose residual has not shrunk by at least this much within STALL_S is
stalled: the arm is pressing on something, or the pose is at a joint limit. The
controller converges geometrically, so an unobstructed arm that is still short is still
moving. Without this check, an arm resting on an object spends its whole motion budget
motionless and is reported as still converging."""


class EpisodeFinished(EpisodeStopped):
    """Raised inside the RoboLab thread when the harness has stopped asking for actions."""


@dataclass
class _Pending:
    target: TCPTarget
    started_t: float
    steps_left: int
    grip: float
    settle_s: float = 0.0
    actuation_t: float | None = None
    best_lin: float = float("inf")
    best_rot: float = float("inf")
    progress_t: float | None = None
    waited_s: float = 0.0
    """Real seconds the simulator waited on the harness before this target arrived."""
    hold_steps_left: int = 0
    """Control steps the robot holds still, playing out that wait, before the motion starts."""
    jaw: list | None = None
    """(t, travel) since the grip command started, for telling when the jaw has stopped."""


class RoboLabEmbodiment:
    """The harness side of the bridge: an ``Embodiment`` backed by a running RoboLab env.

    ``execute`` hands a target to the client thread and blocks until the motion has been
    played out, the target expires, or the episode ends.
    """

    def __init__(
        self,
        capabilities: EmbodimentCapabilities | None = None,
        camera: str = SCENE_CAMERA,
        env_id: int = 0,
        chunk_steps: int = 8,
        convention: str = "opengl",
        time_mode: str = "realtime",
        tracking_gain: float = 1.0,
        history_steps: tuple[int, ...] = (),
        history_scale: float = 0.3,
        extra_cameras: tuple[str, ...] = (),
    ):
        if not 0.0 < tracking_gain <= 1.0:
            raise ValueError("tracking_gain is the realised fraction of a step, in (0, 1]")
        self.tracking_gain = tracking_gain
        if any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in history_steps):
            raise ValueError("history_steps are positive control-step ages")
        if not 0 < history_scale <= 1:
            raise ValueError("history_scale is a resize factor in (0, 1]")
        self.history_steps = tuple(sorted(set(history_steps), reverse=True))
        """Control-step ages of earlier frames handed to the policy with each observation
        (Astranomy's direct policy used t-14 and t-7 at 15 Hz). Empty keeps one frame."""
        self.history_scale = history_scale
        self.extra_cameras = tuple(extra_cameras)
        self._frame_buffer: list[tuple[int, dict]] = []
        """The share of a commanded Cartesian step the controller is expected to realise.
        Commands are divided by it (see translate.action_chunk). 1.0 is uncompensated."""
        if time_mode not in TIME_MODES:
            raise ValueError(f"time_mode must be one of {TIME_MODES}")
        self.time_mode = time_mode
        """``realtime``: time the harness spends between motions (perception, model calls,
        planning) is played out in simulated time with the robot holding still before the
        next motion, as it would pass on a real robot.
        ``paused``: the world waits for the harness at no cost, as in the official
        synchronous RoboLab runner (upstream ``robolab/eval/episode.py`` steps the
        simulation only after inference returns), so it is the leaderboard's own timing.
        Real inference latency is not in simulated time and is reported separately. Success
        is higher than under ``realtime`` (4 of 6 versus 2 of 6 on one set), so paused and
        realtime results are not comparable and each must be labelled with its mode."""
        self._offered_at: float | None = None
        self._excluded_s = 0.0
        self._clock_lock = threading.Lock()
        self.convention = convention
        self.caps = capabilities or droid_capabilities()
        self.camera = camera
        self.env_id = env_id
        self.chunk_steps = chunk_steps
        self._to_sim: queue.Queue = queue.Queue()
        self._to_harness: queue.Queue = queue.Queue()
        self._latest: Observation | None = None
        self._closed = threading.Event()
        self.instruction = ""
        self._commanded_grip: float | None = None

    @property
    def capabilities(self) -> EmbodimentCapabilities:
        return self.caps

    # ---- harness side -----------------------------------------------------

    def reset(self, episode_seed: int | None = None) -> Observation:
        """RoboLab owns resetting. The harness waits for the first observation instead."""
        return self.observe()

    def observe(self) -> Observation:
        if self._latest is None:
            self._latest = self._await_observation()
        return self._latest

    def execute(self, target: TCPTarget) -> ExecutionReport:
        validate_target(target, self.caps)
        start = self.observe()
        if self._closed.is_set():
            raise EpisodeFinished("embodiment is closed")
        grip = (
            _grip_value(target, start)
            if target.gripper.action is not GripperAction.HOLD or self._commanded_grip is None
            else self._commanded_grip
        )
        dt = self.caps.control_period_s
        pending = _Pending(
            target=target,
            started_t=start.t,
            steps_left=max(0, int(np.floor(target.timeout_s / dt + 1e-9))),
            grip=grip,
            settle_s=(
                self.caps.gripper.close_time_s
                if target.gripper.action is not GripperAction.HOLD
                else 0.0
            ),
        )
        self._to_sim.put(pending)
        result = self._to_harness.get()
        if isinstance(result, BaseException):
            raise result
        self._latest, settled, stalled = result
        report = _report(target, start, self._latest, self.caps, settled=settled, stalled=stalled)
        report.waited_s = pending.waited_s
        return report

    def exclude_wall(self, seconds: float) -> None:
        """Exclude wall time a real robot would not spend in its control loop, such as
        writing recordings, which a deployment does in the background."""
        with self._clock_lock:
            self._excluded_s += max(0.0, float(seconds))

    def close(self) -> None:
        self._closed.set()
        self._to_sim.put(None)
        self._to_harness.put(EpisodeFinished("embodiment closed"))

    def _await_observation(self) -> Observation:
        result = self._to_harness.get()
        if isinstance(result, BaseException):
            raise result
        return result

    # ---- simulator side ---------------------------------------------------

    def _remember(self, step: int, observation: Observation) -> None:
        """Keep downscaled copies of this control step's frames for the history window."""
        if not self.history_steps:
            return
        frames = {}
        for frame in (
            observation.scene,
            observation.wrist,
            *(observation.extra.get("views") or []),
        ):
            if frame is None:
                continue
            rgb = frame.rgb
            if self.history_scale != 1.0:
                from PIL import Image

                image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
                size = (
                    max(1, int(image.width * self.history_scale)),
                    max(1, int(image.height * self.history_scale)),
                )
                rgb = np.asarray(image.resize(size, Image.BILINEAR))
            frames[frame.name] = CameraFrame(
                frame.name,
                rgb,
                frame.t,
                convention="opencv",
                state_at_capture=frame.state_at_capture,
            )
        self._frame_buffer.append((step, frames))
        oldest = step - max(self.history_steps)
        self._frame_buffer = [(s, f) for s, f in self._frame_buffer if s >= oldest]

    def _with_history(self, step: int, observation: Observation) -> Observation:
        if not self.history_steps or not self._frame_buffer:
            return observation
        recent = []
        for age in self.history_steps:
            wanted = step - age
            candidates = [(s, f) for s, f in self._frame_buffer if s <= wanted]
            if not candidates:
                continue
            s, frames = candidates[-1]
            if s == step:
                continue
            for frame in frames.values():
                recent.append(((step - s) * self.caps.control_period_s, frame))
        if recent:
            observation.extra["recent_frames"] = recent
        return observation

    def _offer(
        self, observation: Observation, settled: bool | None = None, stalled: bool = False
    ) -> None:
        with self._clock_lock:
            self._offered_at = time.monotonic()
            self._excluded_s = 0.0
        self._to_harness.put(observation if settled is None else (observation, settled, stalled))

    def _next_pending(self) -> _Pending | None:
        """The next target, with the real time the harness took to produce it.

        The simulator is blocked during the wait, so the wait cannot pass in simulated
        time. In real-time mode it is charged afterwards: the robot holds its last command
        for the equivalent number of control steps before the target starts. As on a real
        robot, the decision is based on the observation from before the wait.
        """
        pending = self._to_sim.get()
        with self._clock_lock:
            offered, excluded = self._offered_at, self._excluded_s
            self._offered_at = None
        if pending is not None and offered is not None:
            waited = max(0.0, time.monotonic() - offered - excluded)
            pending.waited_s = waited
            if self.time_mode == "realtime":
                pending.hold_steps_left = int(round(waited / self.caps.control_period_s))
        return pending

    def _fail(self, exc: BaseException) -> None:
        self._to_harness.put(exc)


def _grip_value(target: TCPTarget, observation: Observation) -> float:
    if target.gripper.action is GripperAction.CLOSE:
        return 1.0
    if target.gripper.action is GripperAction.OPEN:
        return 0.0
    return float(observation.gripper.travel)


def _report(
    target: TCPTarget,
    start: Observation,
    end: Observation,
    caps: EmbodimentCapabilities,
    *,
    settled: bool | None = None,
    stalled: bool = False,
) -> ExecutionReport:
    diff = np.abs(np.asarray(end.tcp.pos) - np.asarray(target.pose.pos))
    axis = int(np.argmax(diff))
    rot = quat_angle_between(end.tcp.quat, target.pose.quat)
    if settled is None:
        settled = target.gripper.action is GripperAction.HOLD
    reached = float(diff[axis]) <= ARRIVED_M and rot <= ARRIVED_RAD and settled
    if reached:
        outcome = ExecutionOutcome.REACHED
    elif stalled:
        outcome = ExecutionOutcome.BLOCKED
    else:
        outcome = ExecutionOutcome.EXPIRED
    return ExecutionReport(
        target_id=target.id,
        outcome=outcome,
        t_start=start.t,
        t_end=end.t,
        final_tcp=end.tcp,
        residual_m=float(diff[axis]),
        residual_axis="xyz"[axis],
        residual_rot_rad=rot,
        executed_lin_m=float(np.linalg.norm(np.asarray(end.tcp.pos) - np.asarray(start.tcp.pos))),
        executed_rot_rad=quat_angle_between(start.tcp.quat, end.tcp.quat),
        reason=None
        if reached
        else _stall_reason(diff, axis, rot)
        if stalled
        else _expiry_reason(target, start, end, diff, axis, rot, bool(settled)),
        gripper_travel_end=end.gripper.travel,
    )


def _stall_reason(diff, axis: int, rot: float) -> str:
    short = float(diff[axis]) * 100
    where = "xyz"[axis]
    what = (
        f"{short:.1f} cm short on {where}"
        if float(diff[axis]) > ARRIVED_M
        else f"{rot:.2f} rad from the requested orientation"
    )
    return (
        f"the arm stopped {what} and made no progress for {STALL_S:.0f} s: something "
        "is in the way, or this pose is at the arm's limit. Asking again will not "
        "help; change the target"
    )


def _expiry_reason(
    target: TCPTarget,
    start: Observation,
    end: Observation,
    diff,
    axis: int,
    rot: float = 0.0,
    settled: bool = True,
) -> str:
    """Explain an expired motion, separating an arm that never moved from one still moving.

    With a geometrically converging controller, expiry usually means the arm had not
    finished. Reporting a joint limit or an unreachable target in that case leads the model
    to abandon objects that are reachable.
    """
    travelled = float(np.linalg.norm(np.asarray(end.tcp.pos) - np.asarray(start.tcp.pos)))
    gap_at_start = float(np.linalg.norm(np.asarray(start.tcp.pos) - np.asarray(target.pose.pos)))
    short = float(diff[axis]) * 100
    where = "xyz"[axis]

    # Arrival needs position, orientation and a settled gripper. When the position is
    # within tolerance, name whichever of the other two is missing; a position-only
    # message would read "0.0 cm short" for a motion that did not arrive.
    if float(diff[axis]) <= ARRIVED_M:
        if not settled:
            if target.gripper.action is GripperAction.HOLD and target.min_dwell_s > 0:
                return "the requested hold interval had not elapsed before the deadline"
            return (
                "the hand reached the position but the gripper had not finished "
                "opening or closing when time ran out"
            )
        if rot > ARRIVED_RAD:
            return (
                f"the position is right, within {short:.1f} cm, but the orientation is "
                f"still {rot:.3f} rad off, past the {ARRIVED_RAD:.2f} rad tolerance. "
                "Rotations converge more slowly than moves on this arm; ask again to "
                "continue turning"
            )
    if travelled < 0.002:
        return (
            f"the arm did not move, and is still {short:.1f} cm short on {where}; the "
            "target is out of reach or the arm is against a limit"
        )
    closed = 1.0 - (float(diff[axis]) / gap_at_start) if gap_at_start > 1e-9 else 0.0
    return (
        f"ran out of time {short:.1f} cm short on {where} after travelling "
        f"{travelled * 100:.1f} cm and closing {closed * 100:.0f} percent of the gap. "
        "Still converging, not blocked: ask again to continue from here"
    )


class RungInferenceClient:
    """The RoboLab side of the bridge: what its runner calls ``get_action`` on.

    One instance per environment. It converts the observation, hands it to the harness
    thread when the current motion is finished, and returns the next chunk of relative-IK
    actions. When the harness thread has ended, it returns a hold chunk so the simulator
    can finish the episode cleanly rather than crashing mid-step.
    """

    def __init__(
        self,
        embodiment: RoboLabEmbodiment,
        instruction: str,
        control_hz: float | None = None,
        chunk_steps: int = 8,
    ):
        self.emb = embodiment
        self.instruction = instruction
        expected_hz = 1.0 / embodiment.caps.control_period_s
        if control_hz is not None and not np.isclose(control_hz, expected_hz):
            raise ValueError("control_hz must match embodiment.control_period_s")
        self.control_hz = expected_hz
        self.chunk_steps = chunk_steps
        self.steps = 0
        self._pending: _Pending | None = None
        self._finished = False

    @property
    def t(self) -> float:
        return self.steps / self.control_hz

    def get_action(self, raw_obs: dict) -> np.ndarray:
        observation = to_observation(
            raw_obs,
            t=self.t,
            env_id=self.emb.env_id,
            camera=self.emb.camera,
            capabilities=self.emb.caps,
            convention=self.emb.convention,
            extra_cameras=self.emb.extra_cameras,
        )
        self.emb._remember(self.steps, observation)
        if self._finished:
            return self._finished_chunk(observation.gripper.travel)
        while True:
            p = self._pending
            if p is None:
                self.emb._offer(self.emb._with_history(self.steps, observation))
            else:
                if p.hold_steps_left > 0:
                    break  # harness decision time plays out before the motion
                lin = float(np.max(np.abs(observation.tcp.pos - p.target.pose.pos)))
                rot = quat_angle_between(observation.tcp.quat, p.target.pose.quat)
                arrived = lin <= ARRIVED_M and rot <= ARRIVED_RAD
                if arrived and p.actuation_t is None:
                    p.actuation_t = observation.t
                settled = p.settle_s == 0 or (
                    p.actuation_t is not None
                    and (
                        observation.t - p.actuation_t + 1e-9 >= p.settle_s
                        or _jaw_stopped(p, observation)
                    )
                )
                dwelled = p.target.min_dwell_s == 0 or (
                    p.actuation_t is not None
                    and observation.t - p.actuation_t + 1e-9 >= p.target.min_dwell_s
                )
                # Progress is a residual that shrank. The arm is stalled when neither the
                # position nor the orientation has closed on the target for STALL_S.
                if (
                    p.progress_t is None
                    or lin < p.best_lin - STALL_PROGRESS_M
                    or rot < p.best_rot - STALL_PROGRESS_RAD
                ):
                    p.best_lin, p.best_rot = min(lin, p.best_lin), min(rot, p.best_rot)
                    p.progress_t = observation.t
                stalled = (
                    not arrived
                    and p.steps_left > 0
                    and observation.t - p.progress_t + 1e-9 >= STALL_S
                )
                if p.steps_left > 0 and not (arrived and settled and dwelled) and not stalled:
                    break
                self.emb._offer(
                    self.emb._with_history(self.steps, observation),
                    settled=settled and dwelled,
                    stalled=stalled,
                )
            self._pending = self.emb._next_pending()
            if self._pending is None:
                self._finished = True
                break
            # Evaluate a newly requested target against this same observation, including
            # zero-step deadlines and gripper-only commands, before playing a step.

        if self._finished or self._pending is None:
            return self._finished_chunk(observation.gripper.travel)

        if self._pending.hold_steps_left > 0:
            n = min(self.chunk_steps, self._pending.hold_steps_left)
            self._pending.hold_steps_left -= n
            self.steps += n
            held = self.emb._commanded_grip
            return _hold(n, float(observation.gripper.travel) if held is None else held)

        dt = self.emb.caps.control_period_s
        target = self._pending.target
        take = min(self.chunk_steps, self._pending.steps_left)
        if self._pending.actuation_t is not None:
            self.emb._commanded_grip = self._pending.grip
        elif self.emb._commanded_grip is None:
            self.emb._commanded_grip = float(observation.gripper.travel)
        offset_m = float(self.emb.caps.tcp_from_ee.pos[0])
        linear_step = target.max_lin_speed_mps * dt
        angular_step = target.max_ang_speed_rps * dt
        horizon = max(
            take,
            steps_needed(
                observation.tcp, target.pose, linear_step, angular_step, tcp_offset_m=offset_m
            ),
        )
        chunk = action_chunk(
            current=observation.tcp,
            target_pose=target.pose,
            grip=self.emb._commanded_grip,
            horizon=horizon,
            max_dxyz=linear_step,
            max_drot=angular_step,
            tcp_offset_m=offset_m,
            gain=self.emb.tracking_gain,
        )[:take]
        self._pending.steps_left -= take
        self.steps += take
        return chunk

    def _finished_chunk(self, observed_grip: float) -> np.ndarray:
        self.steps += self.chunk_steps
        grip = self.emb._commanded_grip
        return _hold(self.chunk_steps, observed_grip if grip is None else grip)

    def report_error(self, exc: BaseException) -> None:
        """Called by the runner when the simulator fails, so the harness thread unblocks."""
        self.emb._fail(exc)


def _jaw_stopped(p: _Pending, observation: Observation) -> bool:
    travel = float(observation.gripper.travel)
    if p.jaw is None:
        p.jaw = []
    p.jaw.append((observation.t, travel))
    elapsed = observation.t - p.actuation_t
    if elapsed + 1e-9 < JAW_MIN_SETTLE_S:
        return False
    first = p.jaw[0][1]
    # Started means moved toward the command by a tenth of the travel and still there;
    # a jitter in either direction (37 of 79 closes in the 2026-10-21 screen were
    # reported done at 0.5 s with the jaw still open) is an actuation delay, not a stop.
    toward = (travel - first) if p.grip >= first else (first - travel)
    if toward < JAW_STARTED_TRAVEL and abs(travel - p.grip) >= JAW_MOVED_TRAVEL:
        return False  # not started yet: an actuation delay is not a stop
    recent = [v for t, v in p.jaw if observation.t - t <= JAW_STILL_S + 1e-9]
    if observation.t - p.jaw[0][0] + 1e-9 < JAW_STILL_S or len(recent) < 2:
        return False
    return max(recent) - min(recent) < JAW_STILL_TRAVEL


def _hold(steps: int, grip: float) -> np.ndarray:
    chunk = np.zeros((steps, 7), dtype=np.float32)
    chunk[:, 6] = grip
    return chunk


def run_in_thread(target, *args: Any, **kwargs: Any) -> threading.Thread:
    """Start the harness loop beside the simulator loop."""
    thread = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    return thread
