"""L0: pose targets, the bottom rung.

The model does every piece of metric reasoning, so the tests are about what it is told
and what its answer turns into: a pose, in metres, in the root frame, with the jaw where
it asked for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rung.backends import MockBackend, ModelResponse, ToolCall, call
from rung.contracts.action import GripperAction
from rung.interfaces import L0Config, L0Policy
from rung.loop import EndReason, Limits, run_episode
from rung.mock import SCENES
from rung.policy import Policy


def make(scene: str = "banana_in_bowl", responses=None, config=None):
    emb = SCENES[scene]()
    backend = MockBackend(responses or [])
    return emb, backend, L0Policy(backend, config)


def test_l0_is_a_policy():
    _, _, policy = make()
    assert isinstance(policy, Policy)
    assert policy.interface == "L0"
    assert policy.name == "l0/mock-model"


def test_a_model_that_names_the_right_poses_solves_the_task():
    """L0 exposes pose targets only: no object catalog and no grasp planner."""
    emb = SCENES["banana_in_bowl"]()
    banana, bowl = emb.scene.body("banana"), emb.scene.body("bowl")
    yaw = float(np.degrees(banana.yaw))
    backend = MockBackend(
        [
            call(
                "move_to",
                x=banana.pos[0],
                y=banana.pos[1],
                z=banana.top_z + 0.12,
                yaw_deg=yaw,
                gripper="open",
                note="Above the banana, jaw across it.",
            ),
            call(
                "move_to",
                x=banana.pos[0],
                y=banana.pos[1],
                z=banana.top_z - 0.01,
                yaw_deg=yaw,
                gripper="close",
                note="Down onto it and close.",
            ),
            call(
                "move_to", x=banana.pos[0], y=banana.pos[1], z=0.25, yaw_deg=yaw, note="Lift clear."
            ),
            call(
                "move_to", x=bowl.pos[0], y=bowl.pos[1], z=0.22, yaw_deg=yaw, note="Over the bowl."
            ),
            call(
                "move_to",
                x=bowl.pos[0],
                y=bowl.pos[1],
                z=0.16,
                yaw_deg=yaw,
                gripper="open",
                note="Release into the bowl.",
            ),
            call("done", summary="The banana is in the bowl."),
        ]
    )
    result = run_episode(
        emb, L0Policy(backend), task="banana_in_bowl", limits=Limits(horizon_s=300.0)
    )
    assert result.success, result.grade_detail
    assert result.end_reason is EndReason.DONE


def test_the_prompt_states_the_frame_and_where_the_hand_is():
    emb, backend, policy = make(responses=[call("done", summary="stop")])
    run_episode(emb, policy, task="t")
    prompt = policy.prompts[0]
    assert "Fingertips are at x=" in prompt
    assert "Reachable: x" in prompt
    assert "jaw opens to 8.5 cm" in prompt
    assert "z = 0 is the table top" in prompt


def test_the_prompt_carries_the_residual_after_a_capped_move():
    """After a motion that does not finish, the next prompt states the largest remaining
    offset from the requested target."""
    emb, backend, policy = make(
        responses=[
            call("move_to", x=0.70, y=0.40, z=0.50, note="A long move that will not finish."),
            call("done", summary="stopping"),
        ],
        config=L0Config(max_speed_mps=0.02, timeout_s=1.0),
    )
    run_episode(emb, policy, task="t")
    assert "largest remaining offset" in policy.prompts[1]


def test_no_object_catalog_is_offered():
    """L0 measures unaided metric reasoning. Handing it measured objects would make it L2."""
    emb, backend, policy = make(responses=[call("done", summary="stop")])
    run_episode(emb, policy, task="t")
    prompt = policy.prompts[0]
    assert "cm across at grasp height" not in prompt
    assert "grasp candidate" not in prompt


def test_the_tool_surface_is_small_and_stable():
    from rung.interfaces.l0 import tool_specs

    assert [t.name for t in tool_specs(True)] == ["move_to", "observe", "done", "give_up"]
    move = tool_specs(True)[0]
    assert move.parameters["required"] == ["x", "y", "z", "note"]
    assert move.parameters["properties"]["gripper"]["enum"] == ["open", "close", "hold"]


def test_a_pose_becomes_an_absolute_top_down_target():
    emb, backend, policy = make(
        responses=[call("move_to", x=0.5, y=-0.1, z=0.3, yaw_deg=90, gripper="close", note="test")]
    )
    policy.reset(emb.instruction, emb.capabilities)
    decision = policy.act(emb.observe(), None)
    target = decision.targets[0]
    assert np.allclose(target.pose.pos, [0.5, -0.1, 0.3])
    assert target.gripper.action is GripperAction.CLOSE
    # pointing straight down: the approach axis is world -z
    approach = target.pose.rot @ np.array([0.0, 0.0, 1.0])
    assert approach[2] == pytest.approx(-1.0, abs=1e-9)


def test_yaw_turns_the_jaw_about_the_vertical_axis():
    emb, backend, policy = make(
        responses=[
            call("move_to", x=0.45, y=0.0, z=0.2, yaw_deg=0, note="a"),
            call("move_to", x=0.45, y=0.0, z=0.2, yaw_deg=90, note="b"),
        ]
    )
    policy.reset(emb.instruction, emb.capabilities)
    first = policy.act(emb.observe(), None).targets[0]
    second = policy.act(emb.observe(), None).targets[0]
    jaw1 = first.pose.rot[:, 1][:2]
    jaw2 = second.pose.rot[:, 1][:2]
    angle = abs(float(np.degrees(np.arctan2(jaw2[1], jaw2[0]) - np.arctan2(jaw1[1], jaw1[0]))))
    assert min(angle, 360 - angle) == pytest.approx(90.0, abs=1e-6)


def test_a_malformed_pose_is_retried_not_fatal():
    emb, backend, policy = make(
        responses=[
            ToolCall("move_to", {"x": 0.4, "y": 0.0, "note": "no z"}),
            call("done", summary="recovered"),
        ]
    )
    result = run_episode(emb, policy, task="t")
    assert result.end_reason is EndReason.DONE
    assert "needs x, y and z" in policy.prompts[1]


def test_a_non_numeric_pose_is_rejected():
    emb, backend, policy = make(
        responses=[ToolCall("move_to", {"x": "left a bit", "y": 0.0, "z": 0.2, "note": "n"})] * 3
    )
    result = run_episode(emb, policy, task="t", limits=Limits(max_consecutive_errors=3))
    assert result.end_reason is EndReason.POLICY_ERROR


def test_an_unknown_gripper_action_is_rejected():
    emb, backend, policy = make(
        responses=[
            ToolCall("move_to", {"x": 0.4, "y": 0.0, "z": 0.2, "gripper": "squeeze", "note": "n"})
        ]
        * 3
    )
    result = run_episode(emb, policy, task="t", limits=Limits(max_consecutive_errors=3))
    assert "open, close, hold" in (result.error or "")


def test_done_does_not_imply_success():
    emb, backend, policy = make(responses=[call("done", summary="I believe it is done.")])
    result = run_episode(emb, policy, task="t")
    assert result.end_reason is EndReason.DONE
    assert not result.success


def test_a_refusal_ends_the_episode():
    emb, backend, policy = make(responses=[ModelResponse(refusal="declined")])
    result = run_episode(emb, policy, task="t")
    assert result.end_reason is EndReason.GAVE_UP


def test_usage_is_drained_into_the_episode_total():
    emb, backend, policy = make(
        responses=[call("observe", note="look"), call("done", summary="stop")]
    )
    result = run_episode(emb, policy, task="t")
    assert result.usage.calls == 2


def test_the_runner_can_build_an_l0_policy(monkeypatch):
    """L0 is reachable from the command line, not just from the library."""
    from rung import cli
    from rung.interfaces import L0Policy

    monkeypatch.setattr(cli, "build_backend", lambda *a, **k: MockBackend([]))

    class Args:
        policy = "l0"
        model = "gemini-3.7-flash"
        vendor = None
        effort = "low"
        no_wrist = False
        seed = None
        grounding_model = None

    policy = cli.build_policy(Args(), SCENES["banana_in_bowl"]())
    assert isinstance(policy, L0Policy)
    assert policy.interface == "L0"
    assert policy.cfg.effort == "low"


def test_the_runner_refuses_l0_without_a_model():
    from rung.cli import build_policy

    class Args:
        policy = "l0"
        model = None
        vendor = None
        effort = None
        no_wrist = False
        seed = None
        grounding_model = None

    with pytest.raises(SystemExit, match="needs --model"):
        build_policy(Args(), SCENES["banana_in_bowl"]())
