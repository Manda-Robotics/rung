"""The episode loop: budgets, feedback, termination and the trace it leaves behind.

Includes the baseline check: a policy completes an episode against the mock and the trace
records what happened.
"""

from __future__ import annotations

import json

import pytest

from rung.contracts.action import GripperAction, GripperCommand, TCPTarget
from rung.contracts.trace import TraceWriter, read_trace
from rung.loop import EndReason, Limits, describe_gripper, run_episode
from rung.mock import SCENES
from rung.mock.world import top_down_pose
from rung.policy import Decision, ModelUsage, Policy, Verdict
from rung.record import Recorder
from rung.reference import RandomReference, ScriptedReference


def test_scripted_reference_completes_and_succeeds():
    emb = SCENES["banana_in_bowl"]()
    result = run_episode(emb, ScriptedReference(emb), task="banana_in_bowl")
    assert result.success, result.grade_detail
    assert result.end_reason is EndReason.DONE
    assert result.decisions == 8  # seven phases plus the closing done
    assert result.sim_seconds > 0
    assert len(result.notes) == result.decisions


def test_scripted_reference_satisfies_the_policy_protocol():
    emb = SCENES["banana_in_bowl"]()
    assert isinstance(ScriptedReference(emb), Policy)
    assert isinstance(RandomReference(emb), Policy)


def test_full_trace_is_written(tmp_path):
    emb = SCENES["banana_in_bowl"]()
    trace = TraceWriter(tmp_path / "trace.jsonl")
    result = run_episode(emb, ScriptedReference(emb), task="banana_in_bowl", trace=trace)
    rows = read_trace(tmp_path / "trace.jsonl")
    kinds = [r["kind"] for r in rows]
    assert kinds[0] == "observation" and kinds[-1] == "episode_end"
    assert kinds.count("decision") == result.decisions
    assert kinds.count("execution") == result.servo_steps
    assert all(r["episode"] == result.episode_id for r in rows)
    assert [r["seq"] for r in rows] == list(range(1, len(rows) + 1))
    end = json.loads(json.dumps(rows[-1]["payload"]))
    assert end["success"] is True
    assert end["end_reason"] == "policy_done"


def test_every_decision_carries_a_note(tmp_path):
    emb = SCENES["banana_in_bowl"]()
    trace = TraceWriter(tmp_path / "t.jsonl")
    run_episode(emb, ScriptedReference(emb), task="t", trace=trace)
    notes = [e.payload["note"] for e in trace.events if e.kind == "decision"]
    assert all(isinstance(n, str) and len(n) > 10 for n in notes)


def test_recorder_saves_what_the_policy_saw(tmp_path):
    emb = SCENES["banana_in_bowl"]()
    rec = Recorder(tmp_path / "runs")
    result = run_episode(emb, ScriptedReference(emb), task="banana_in_bowl", recorder=rec)
    d = tmp_path / "runs" / result.episode_id
    assert (d / "obs_0000_scene.png").exists()
    assert (d / "obs_0000_wrist.png").exists()
    assert (d / "obs_0000_scene_depth.npy").exists()
    assert (d / "result.json").exists()
    meta = json.loads((d / "obs_0000.json").read_text())
    assert meta["cameras"]["scene"]["convention"] == "opencv"
    assert meta["cameras"]["scene"]["K"] is not None
    saved = json.loads((d / "result.json").read_text())
    assert saved["success"] is True


def test_recorder_images_can_be_switched_off(tmp_path):
    emb = SCENES["banana_in_bowl"]()
    rec = Recorder(tmp_path / "runs", images=False)
    result = run_episode(emb, ScriptedReference(emb), task="t", recorder=rec)
    d = tmp_path / "runs" / result.episode_id
    assert not list(d.glob("*.png"))
    assert (d / "obs_0000.json").exists()


class _Stubborn:
    """Never stops on its own; used to test the limits."""

    name = "stubborn"
    interface = "test"

    def __init__(self, dollars_per_call: float = 0.0):
        self.dollars = dollars_per_call

    def reset(self, instruction, capabilities):
        self.caps = capabilities

    def act(self, observation, feedback):
        usage = ModelUsage(calls=1, dollars=self.dollars)
        pos = observation.tcp.pos + [0.01, 0.0, 0.0]
        return Decision(
            kind="nudge",
            note="Nudging along x to burn budget.",
            targets=[TCPTarget(pose=top_down_pose(pos), timeout_s=5.0)],
            payload={"usage": usage},
        )


def test_decision_limit_ends_the_episode():
    emb = SCENES["banana_in_bowl"]()
    r = run_episode(emb, _Stubborn(), task="t", limits=Limits(max_decisions=5))
    assert r.end_reason is EndReason.DECISIONS
    assert r.decisions == 5


def test_dollar_cap_ends_the_episode():
    emb = SCENES["banana_in_bowl"]()
    r = run_episode(
        emb,
        _Stubborn(dollars_per_call=0.3),
        task="t",
        limits=Limits(max_decisions=100, max_dollars=1.0),
    )
    assert r.end_reason is EndReason.DOLLARS
    assert r.usage.dollars == pytest.approx(1.2, abs=1e-9)
    assert r.usage.calls == 4


def test_simulated_horizon_ends_the_episode():
    emb = SCENES["banana_in_bowl"]()
    r = run_episode(emb, _Stubborn(), task="t", limits=Limits(horizon_s=1.0, max_decisions=1000))
    assert r.end_reason is EndReason.HORIZON
    assert r.sim_seconds >= 1.0


class _Broken:
    name = "broken"
    interface = "test"

    def reset(self, instruction, capabilities):
        pass

    def act(self, observation, feedback):
        raise ValueError("could not parse the response")


def test_repeated_policy_errors_end_the_episode_and_are_fed_back():
    emb = SCENES["banana_in_bowl"]()
    r = run_episode(emb, _Broken(), task="t", limits=Limits(max_consecutive_errors=3))
    assert r.end_reason is EndReason.POLICY_ERROR
    assert "could not parse" in (r.error or "")
    assert r.decisions == 0


class _Quitter:
    name = "quitter"
    interface = "test"

    def reset(self, instruction, capabilities):
        pass

    def act(self, observation, feedback):
        return Decision(
            "give_up",
            "The object is not reachable from here.",
            verdict=Verdict.GIVE_UP,
            reason="unreachable",
        )


def test_give_up_is_recorded_as_a_distinct_ending():
    emb = SCENES["banana_in_bowl"]()
    r = run_episode(emb, _Quitter(), task="t")
    assert r.end_reason is EndReason.GAVE_UP
    assert not r.success


def test_done_does_not_imply_success():
    """The policy ends its own turn; the grader decides the outcome."""

    class _ClaimsVictory:
        name = "claims-victory"
        interface = "test"

        def reset(self, instruction, capabilities):
            pass

        def act(self, observation, feedback):
            return Decision("done", "I believe the task is complete.", verdict=Verdict.DONE)

    emb = SCENES["banana_in_bowl"]()
    r = run_episode(emb, _ClaimsVictory(), task="t")
    assert r.end_reason is EndReason.DONE
    assert not r.success


def test_feedback_reports_the_residual_after_a_capped_move():
    emb = SCENES["banana_in_bowl"]()

    class _OneBigMove:
        name = "one-big-move"
        interface = "test"
        seen: list[str] = []

        def reset(self, instruction, capabilities):
            pass

        def act(self, observation, feedback):
            _OneBigMove.seen.append(feedback.render())
            if len(_OneBigMove.seen) > 1:
                return Decision("done", "Stopping after one move.", verdict=Verdict.DONE)
            return Decision(
                kind="move_to",
                note="A single long slow move that cannot finish inside the timeout.",
                targets=[
                    TCPTarget(
                        pose=top_down_pose([0.70, 0.40, 0.50]),
                        max_lin_speed_mps=0.02,
                        timeout_s=1.0,
                    )
                ],
            )

    run_episode(emb, _OneBigMove(), task="t")
    assert "Task:" in _OneBigMove.seen[0]
    assert "largest remaining offset" in _OneBigMove.seen[1]
    assert "expired" in _OneBigMove.seen[1]


def test_gripper_description_separates_closed_empty_from_holding():
    emb = SCENES["banana_in_bowl"]()
    assert describe_gripper(emb.observe()) == "open"

    emb.execute(
        TCPTarget(
            pose=top_down_pose([0.30, 0.30, 0.06]),
            gripper=GripperCommand(GripperAction.CLOSE),
            timeout_s=10.0,
        )
    )
    assert "not holding anything" in describe_gripper(emb.observe())

    emb.reset()
    banana = emb.scene.body("banana")
    emb.execute(
        TCPTarget(
            pose=top_down_pose(banana.pos + [0, 0, 0.005], banana.yaw),
            gripper=GripperCommand(GripperAction.CLOSE),
            max_lin_speed_mps=0.3,
            timeout_s=10.0,
        )
    )
    said = describe_gripper(emb.observe())
    assert "probably holding" in said


def test_random_reference_rarely_succeeds_and_never_crashes():
    successes = 0
    for seed in range(6):
        emb = SCENES["banana_in_bowl"]()
        r = run_episode(
            emb,
            RandomReference(emb, seed=seed),
            task="t",
            limits=Limits(max_decisions=25, horizon_s=300.0),
        )
        successes += int(r.success)
    assert successes == 0


def test_scripted_reference_on_every_scene():
    """The scripted reference completes the banana and cluttered-bin scenes; the tall-bottle
    scene is run but its outcome is not asserted."""
    outcomes = {}
    for name, make in SCENES.items():
        emb = make()
        r = run_episode(emb, ScriptedReference(emb), task=name, limits=Limits(horizon_s=400.0))
        outcomes[name] = r.success
    assert outcomes["banana_in_bowl"] is True
    assert outcomes["cluttered_bin"] is True
    assert "tall_bottle_and_crate" in outcomes


def test_execution_trace_records_commanded_gripper_and_jaw_travel(tmp_path):
    from rung.contracts.trace import TraceWriter
    from rung.mock import SCENES
    from rung.reference import ScriptedReference

    emb = SCENES["banana_in_bowl"]()
    trace = TraceWriter(tmp_path / "trace.jsonl")
    run_episode(emb, ScriptedReference(emb), task="t", trace=trace)
    executions = [e.payload for e in trace.events if e.kind == "execution"]
    assert executions and all("gripper_travel_end" in p and "gripper" in p for p in executions)
    assert any(p["gripper"] == "close" for p in executions)
