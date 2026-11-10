"""The episode loop: budgets, feedback, termination, tracing.

Four limits bound every episode, and the result always says which one ended it. Calls are
not comparable across interfaces (one L2 decision can execute a whole grasp, one L0
decision moves a few centimetres), so a decision limit alone is not a fair budget; the
simulated horizon and the dollar cap are what make runs comparable. See docs/protocol.md.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from rung.contracts.action import ExecutionOutcome, ExecutionReport
from rung.contracts.observation import Observation
from rung.contracts.trace import TraceWriter
from rung.embodiment import Embodiment, EpisodeStopped, Gradeable
from rung.policy import Decision, Feedback, ModelUsage, Policy, Verdict
from rung.record import Recorder


class _LimitReached(Exception):
    def __init__(self, reason):
        self.reason = reason


class _ExecutionFailed(Exception):
    pass


class EndReason(StrEnum):
    EXTERNAL = "external_episode_end"
    DONE = "policy_done"
    GAVE_UP = "policy_gave_up"
    HORIZON = "simulated_horizon"
    DECISIONS = "decision_limit"
    MODEL_CALLS = "model_call_limit"
    UNKNOWN_COST = "unaccounted_model_call"
    DOLLARS = "dollar_cap"
    WALL_CLOCK = "wall_clock_limit"
    """Real time ran out. Covers model calls and simulation together, so it does not by
    itself attribute the slowness to the model."""
    POLICY_ERROR = "policy_error"
    EMBODIMENT_ERROR = "embodiment_error"


@dataclass
class Limits:
    """Limits checked between operations. ``horizon_s`` is elapsed simulated time.

    Target deadlines are bounded by the remaining horizon. Dollar and wall-clock limits
    cannot cancel in-flight model calls; known overruns are still included in usage.

    The defaults are deliberately small so that a misconfigured run stops cheaply. Real
    runs set them from the task and the run's budget.
    """

    horizon_s: float = 120.0
    max_decisions: int = 40
    max_dollars: float = 1.0
    max_wall_s: float = 2400.0
    """Real seconds one episode may take, model calls and simulation together.

    A runaway guard, not a thinking budget. It measures wall clock from the episode start,
    so a slow simulator consumes it as fast as a slow model. An episode that ends here
    means the run was too slow to finish, not that the policy failed. Set it from the
    run's budget; an Isaac Sim episode typically takes several hundred real seconds.
    """
    max_consecutive_errors: int = 3
    max_model_calls: int | None = None
    """Includes rejected tool replies and failed API attempts for native policies."""


@dataclass
class EpisodeResult:
    episode_id: str
    task: str
    instruction: str
    policy: str
    interface: str
    success: bool | None
    grade_detail: dict[str, Any]
    end_reason: EndReason
    decisions: int
    servo_steps: int
    sim_seconds: float
    wall_seconds: float
    usage: ModelUsage
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> str:
        mark = "ungraded" if self.success is None else ("success" if self.success else "failure")
        return (
            f"{self.task} [{self.interface}/{self.policy}] {mark} after {self.decisions} "
            f"decisions, {self.sim_seconds:.0f} s sim, ${self.usage.dollars:.3f}, "
            f"ended by {self.end_reason.value}"
        )


def run_episode(
    embodiment: Embodiment,
    policy: Policy,
    *,
    task: str,
    instruction: str | None = None,
    limits: Limits | None = None,
    trace: TraceWriter | None = None,
    recorder: Recorder | None = None,
    episode_seed: int | None = None,
    episode_id: str | None = None,
) -> EpisodeResult:
    """Run one episode to a stated limit and return what happened.

    The loop never lets the grader's verdict reach the policy. ``done`` ends the policy's
    turn; whether the task succeeded is decided afterwards, by the embodiment's grader.
    """
    limits = limits or Limits()
    episode_id = episode_id or uuid.uuid4().hex[:10]
    caps = embodiment.capabilities
    obs = embodiment.reset(episode_seed)
    instruction = instruction or getattr(embodiment, "instruction", task)
    policy.reset(instruction, caps)
    attach = getattr(policy, "attach_trace", None)
    if attach is not None:
        attach(trace, episode_id)

    if trace is not None:
        trace.emit(
            "observation",
            obs.t,
            episode_id,
            stage="reset",
            instruction=instruction,
            task=task,
            policy=policy.name,
            interface=policy.interface,
            limits=vars(limits),
        )

    usage = ModelUsage()
    feedback = Feedback(lines=[f"Task: {instruction}"])
    notes: list[str] = []
    decisions = 0
    servo_steps = 0
    consecutive_errors = 0
    wall_start = time.monotonic()
    sim_start = obs.t
    end_reason: EndReason | None = None
    error: str | None = None

    def collect_usage():
        drain = getattr(policy, "drain_usage", None)
        if drain is not None:
            usage.add(drain())

    def execute_target(target, *, servo=False):
        nonlocal obs, servo_steps
        obs = embodiment.observe()
        collect_usage()
        reason = _limit_hit(
            obs, 0, usage, wall_start, replace(limits, max_model_calls=None), sim_start
        )
        if reason is not None:
            raise _LimitReached(reason)
        remaining = limits.horizon_s - (obs.t - sim_start)
        if remaining < caps.control_period_s - 1e-9:
            raise _LimitReached(EndReason.HORIZON)
        bounded = replace(target, timeout_s=min(target.timeout_s, remaining))
        if trace is not None:
            trace.emit(
                "attempt",
                obs.t,
                episode_id,
                servo=servo,
                target_id=target.id,
                pose=target.pose.pos,
                quat=target.pose.quat,
                gripper=target.gripper.action.value,
                decision_kind=decision.kind,
                timeout_s=bounded.timeout_s,
                requested_timeout_s=target.timeout_s,
                min_dwell_s=target.min_dwell_s,
                phase=(target.meta or {}).get("phase"),
                **(
                    {
                        "arm": target.arm,
                        "paired": [
                            {
                                "arm": p.arm,
                                "pose": p.pose.pos,
                                "quat": p.pose.quat,
                                "gripper": p.gripper.action.value,
                            }
                            for p in target.paired
                        ],
                    }
                    if target.arm is not None or target.paired
                    else {}
                ),
            )
        try:
            report = embodiment.execute(bounded)
        except EpisodeStopped:
            raise _LimitReached(EndReason.EXTERNAL) from None
        except Exception as exc:
            if trace is not None:
                trace.emit(
                    "execution_error",
                    obs.t,
                    episode_id,
                    target_id=target.id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise _ExecutionFailed(f"{type(exc).__name__}: {exc}") from exc
        reports.append(report)
        servo_steps += 1
        obs = embodiment.observe()
        if trace is not None:
            trace.emit(
                "execution",
                report.t_end,
                episode_id,
                servo=servo,
                target_id=report.target_id,
                outcome=report.outcome.value,
                residual_m=report.residual_m,
                residual_axis=report.residual_axis,
                residual_rot_rad=report.residual_rot_rad,
                executed_lin_m=report.executed_lin_m,
                reason=report.reason,
                waited_s=getattr(report, "waited_s", 0.0),
                gripper=target.gripper.action.value,
                gripper_travel_end=report.gripper_travel_end,
                phase=(target.meta or {}).get("phase"),
                **({"arms": report.arms} if report.arms else {}),
            )
            if report.clamped_to is not None or report.outcome is ExecutionOutcome.BLOCKED:
                trace.emit(
                    "intervention",
                    report.t_end,
                    episode_id,
                    intervention="clamp" if report.clamped_to is not None else "blocked",
                    target_id=report.target_id,
                    reason=report.reason,
                )
        return report

    recorded_prompts = [0]
    while True:
        end_reason = _limit_hit(
            obs,
            decisions,
            usage,
            wall_start,
            limits,
            sim_start,
            getattr(policy, "model_calls_attempted", usage.calls),
        )
        if end_reason is not None:
            break

        if recorder is not None:
            with _off_the_clock(embodiment):
                recorder.record_observation(episode_id, decisions, obs)

        try:
            decision = policy.act(obs, feedback)
        except Exception as exc:
            _record_turn(recorder, episode_id, policy, recorded_prompts, embodiment)
            collect_usage()
            consecutive_errors += 1
            error = f"{type(exc).__name__}: {exc}"
            if trace is not None:
                trace.emit("decision", obs.t, episode_id, error=error, index=decisions)
            if consecutive_errors >= limits.max_consecutive_errors:
                end_reason = EndReason.POLICY_ERROR
                break
            feedback = Feedback(
                lines=[f"Your last response could not be used ({error}). Try again."],
                structured={"error": error},
            )
            continue

        _record_turn(recorder, episode_id, policy, recorded_prompts, embodiment)
        collect_usage()
        consecutive_errors = 0
        decisions += 1
        if decision.note:
            notes.append(decision.note)
        step_usage = decision.payload.get("usage")
        if not hasattr(policy, "drain_usage") and isinstance(step_usage, ModelUsage):
            usage.add(step_usage)

        if trace is not None:
            trace.emit(
                "decision",
                obs.t,
                episode_id,
                index=decisions,
                decision_kind=decision.kind,
                note=decision.note,
                verdict=decision.verdict.value,
                reason=decision.reason,
                payload=_traceable(decision.payload),
            )

        reports: list[ExecutionReport] = []
        try:
            deferred = getattr(policy, "execute_decision", None)
            if deferred is not None:
                if _accepts_observe(deferred):
                    # A look captures at its viewpoint before returning (direct look tool).
                    deferred(decision, execute_target, observe=embodiment.observe)
                else:
                    deferred(decision, execute_target)
            for target in decision.targets:
                report = execute_target(target)
                if report.outcome is not ExecutionOutcome.REACHED:
                    break
        except _LimitReached as exc:
            end_reason = exc.reason
            feedback = build_feedback(decision, reports, obs)
            feedback.structured["interrupted"] = True
            break
        except _ExecutionFailed as exc:
            error = str(exc)
            end_reason = EndReason.EMBODIMENT_ERROR
            feedback = build_feedback(decision, reports, obs)
            feedback.structured["interrupted"] = True
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            end_reason = EndReason.POLICY_ERROR
            feedback = build_feedback(decision, reports, obs)
            feedback.structured["interrupted"] = True
            break
        finally:
            collect_usage()

        obs = embodiment.observe()
        feedback = build_feedback(decision, reports, obs)

        if decision.verdict is Verdict.DONE:
            end_reason = EndReason.DONE
            break
        if decision.verdict is Verdict.GIVE_UP:
            end_reason = EndReason.GAVE_UP
            break

    finalize = getattr(policy, "finalize", None)
    if finalize is not None:
        finalize(obs, feedback, end_reason or EndReason.HORIZON)
    success, detail = (None, {"reason": "awaiting external grade"})
    if isinstance(embodiment, Gradeable):
        success, detail = embodiment.grade()

    result = EpisodeResult(
        episode_id=episode_id,
        task=task,
        instruction=instruction,
        policy=policy.name,
        interface=policy.interface,
        success=success,
        grade_detail=detail,
        end_reason=end_reason or EndReason.HORIZON,
        decisions=decisions,
        servo_steps=servo_steps,
        sim_seconds=obs.t - sim_start,
        wall_seconds=time.monotonic() - wall_start,
        usage=usage,
        notes=notes,
        error=error,
    )
    if trace is not None:
        trace.emit(
            "policy_end" if success is None else "episode_end",
            obs.t,
            episode_id,
            success=success,
            grade=detail,
            end_reason=result.end_reason.value,
            decisions=decisions,
            sim_seconds=obs.t - sim_start,
            dollars=usage.dollars,
            error=error,
        )
    if recorder is not None:
        recorder.record_result(result)
    return result


def _accepts_observe(method) -> bool:
    import inspect

    try:
        return "observe" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def _limit_hit(
    obs: Observation,
    decisions: int,
    usage: ModelUsage,
    wall_start: float,
    limits: Limits,
    sim_start: float = 0.0,
    model_calls: int | None = None,
) -> EndReason | None:
    if usage.unknown_cost_calls:
        return EndReason.UNKNOWN_COST
    if obs.t - sim_start >= limits.horizon_s - 1e-9:
        return EndReason.HORIZON
    if decisions >= limits.max_decisions:
        return EndReason.DECISIONS
    if (
        limits.max_model_calls is not None
        and (usage.calls if model_calls is None else model_calls) >= limits.max_model_calls
    ):
        return EndReason.MODEL_CALLS
    if usage.dollars >= limits.max_dollars:
        return EndReason.DOLLARS
    if time.monotonic() - wall_start >= limits.max_wall_s:
        return EndReason.WALL_CLOCK
    return None


def build_feedback(
    decision: Decision, reports: list[ExecutionReport], obs: Observation
) -> Feedback:
    """What the policy is told about its last decision.

    Each non-reached move reports its residual, following RoboCurve's agent: a policy that
    is not told how far it got re-plans from scratch every turn.
    """
    lines: list[str] = []
    structured: dict[str, Any] = {"gripper_travel": obs.gripper.travel}
    for report in reports:
        parts = [f"The move {report.outcome.value}"]
        if report.outcome is not ExecutionOutcome.REACHED:
            parts.append(
                f"largest remaining offset from the requested target is "
                f"{report.residual_m * 100:.1f} cm on {report.residual_axis}"
            )
        if report.residual_rot_rad > 0.02:
            parts.append(f"rotation is {report.residual_rot_rad:.2f} rad short")
        if report.reason:
            parts.append(report.reason)
        lines.append(". ".join(parts) + ".")
    structured["outcomes"] = [r.outcome.value for r in reports]
    structured["residual_m"] = [r.residual_m for r in reports]
    structured["execution"] = [
        {
            "target_id": r.target_id,
            "outcome": r.outcome.value,
            "residual_m": r.residual_m,
            "residual_axis": r.residual_axis,
            "residual_rot_rad": r.residual_rot_rad,
            "reason": r.reason,
            "final_tcp_xyz_m": r.final_tcp.pos.tolist(),
            "final_tcp_quat_wxyz": r.final_tcp.quat.tolist(),
            "executed_lin_m": r.executed_lin_m,
            "executed_rot_rad": r.executed_rot_rad,
            "gripper_travel_end": r.gripper_travel_end,
        }
        for r in reports
    ]
    if any(r.arms for r in reports):
        for entry, r in zip(structured["execution"], reports, strict=True):
            if r.arms:
                entry["arms"] = r.arms
    if obs.arms:
        structured["arm_gripper_travel"] = {a: s.gripper.travel for a, s in obs.arms.items()}
        lines.append(
            "Grippers: "
            + "; ".join(f"{a} {describe_gripper(s)}" for a, s in obs.arms.items())
            + "."
        )
        return Feedback(lines=lines, structured=structured)
    lines.append(f"Gripper: {describe_gripper(obs)}.")
    return Feedback(lines=lines, structured=structured)


def describe_gripper(obs: Observation) -> str:
    """Open, closed on nothing, or holding.

    Finger travel alone supports "probably holding", not verified attachment: an
    obstructed close and a real grasp look the same. Verification belongs to the L2 grasp
    primitive, which lifts and checks.
    """
    travel = obs.gripper.travel
    if travel < 0.05:
        return "open"
    if travel > 0.95:
        return "closed on nothing, you are not holding anything"
    return f"closed at {travel * 100:.0f} percent travel, so probably holding something"


def _traceable(payload: dict[str, Any]) -> dict[str, Any]:
    """The decision payload as traced, with ``ModelUsage`` flattened to plain numbers."""
    import dataclasses

    return {
        k: (dataclasses.asdict(v) if isinstance(v, ModelUsage) else v) for k, v in payload.items()
    }


class _off_the_clock:
    """Exclude recording I/O from the embodiment's real-time clock.

    A deployment logs in the background, so disk writes are not charged as robot time.
    """

    def __init__(self, embodiment):
        self.exclude = getattr(embodiment, "exclude_wall", None)

    def __enter__(self):
        self.t0 = time.monotonic()

    def __exit__(self, *exc):
        if self.exclude is not None:
            self.exclude(time.monotonic() - self.t0)
        return False


def _record_turn(recorder, episode_id: str, policy, counter: list[int], embodiment=None) -> None:
    """Record what the model was shown: every new prompt and the catalog behind it.

    Without the catalog snapshot an episode can be watched but not replayed, so grounding
    errors cannot be traced to their source.
    """
    if recorder is None:
        return
    with _off_the_clock(embodiment):
        _write_turn(recorder, episode_id, policy, counter)


def _write_turn(recorder, episode_id: str, policy, counter: list[int]) -> None:
    prompts = getattr(policy, "prompts", None) or []
    for i in range(counter[0], len(prompts)):
        recorder.record_prompt(episode_id, i, prompts[i])
    counter[0] = len(prompts)
    snapshot = getattr(policy, "last_snapshot", None)
    if snapshot:
        recorder.record_json(episode_id, f"catalog_{max(0, len(prompts) - 1):04d}.json", snapshot)
