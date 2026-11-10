"""Guarded action sequences and the depth probe on the direct interface.

Both are opt-in development candidates. A sequence must stop where the world diverged
from the plan; a depth probe must report a surface, never a fabricated table point.
"""

import numpy as np
import pytest

from rung.contracts.action import ExecutionOutcome, ExecutionReport
from rung.contracts.frames import Pose
from rung.contracts.observation import CameraFrame
from rung.interfaces.depth_probe import probe_depth_pixel
from rung.interfaces.direct import DirectConfig, DirectPolicy, tools
from rung.loop import Limits, run_episode
from rung.mock import SCENES
from rung.policy import Feedback
from tests.test_direct_session import reply, session


def policy(replies, **config):
    emb = SCENES["banana_in_bowl"]()
    s, fake = session(replies)
    p = DirectPolicy(s, DirectConfig(**config))
    p.reset("test task", emb.capabilities)
    return emb, p, fake


def sequence(steps):
    return reply("execute_sequence", {"steps": steps, "note": "approach, close, lift"})


# ---- tool surface ----------------------------------------------------------------


def test_sequence_and_probe_are_opt_in():
    names = {t.name for t in tools("both")}
    assert "execute_sequence" not in names and "probe_point" not in names
    names = {
        t.name
        for t in tools("both", action_sequence=True, depth_probe=True, depth_cameras=["scene"])
    }
    assert {"execute_sequence", "probe_point"} <= names
    assert "probe_point" not in {t.name for t in tools("both", depth_probe=True, depth_cameras=[])}


def test_sequence_step_tools_follow_the_motion_profile():
    spec = next(t for t in tools("relative", action_sequence=True) if t.name == "execute_sequence")
    step_tools = spec.parameters["properties"]["steps"]["items"]["properties"]["tool"]["enum"]
    assert "move_to" not in step_tools and "move_relative" in step_tools
    assert "move_relative_tcp" not in step_tools
    assert not {"observe", "done", "give_up", "project_table"} & set(step_tools)
    spec = next(
        t
        for t in tools("both", action_sequence=True, tcp_frame_motions=True, max_sequence_steps=4)
        if t.name == "execute_sequence"
    )
    assert spec.parameters["properties"]["steps"]["maxItems"] == 4
    assert (
        "move_relative_tcp"
        in spec.parameters["properties"]["steps"]["items"]["properties"]["tool"]["enum"]
    )


def test_config_bounds_sequence_length():
    with pytest.raises(ValueError):
        DirectConfig(max_sequence_steps=0)
    with pytest.raises(ValueError):
        DirectConfig(max_sequence_steps=True)


# ---- sequence planning -----------------------------------------------------------


def test_sequence_chains_relative_steps_from_predicted_pose():
    emb, p, _ = policy(
        [
            sequence(
                [
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0, 0, -0.03],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                    {"tool": "set_gripper", "action": "close"},
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0, 0, 0.05],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                ]
            )
        ],
        action_sequence=True,
    )
    obs = emb.observe()
    decision = p.act(obs, Feedback())
    assert decision.targets == []  # executed through execute_decision, not the loop
    targets = p._sequence_targets
    assert len(targets) == 3
    np.testing.assert_allclose(targets[0].pose.pos, obs.tcp.pos + [0, 0, -0.03])
    np.testing.assert_allclose(targets[1].pose.pos, obs.tcp.pos + [0, 0, -0.03])
    assert targets[1].gripper.action == "close"
    assert targets[1].meta["stop_if_closed_on_nothing"]
    np.testing.assert_allclose(targets[2].pose.pos, obs.tcp.pos + [0, 0, 0.02])
    assert targets[2].gripper.action == "hold"
    assert [s["tool"] for s in decision.payload["sequence"]["planned_steps"]] == [
        "move_relative",
        "set_gripper",
        "move_relative",
    ]


@pytest.mark.parametrize(
    "steps, message",
    [
        (
            [
                {
                    "tool": "move_relative",
                    "delta_xyz_m": [0, 0, 0.11],
                    "rotation_vector_rad": [0, 0, 0],
                }
            ],
            "out-of-bounds",
        ),
        (
            [{"tool": "move_to", "xyz_m": [0.5, 0, 0.2], "quat_wxyz": [1, 0, 0, 0]}],
            "not a sequence",
        ),
        ([{"tool": "observe"}], "not a sequence"),
        ([{"tool": "set_gripper", "action": "close", "seconds": 1}], "exactly"),
        ([{"tool": "wait", "seconds": 9}], "wait seconds"),
        ([], "non-empty"),
        ([{"tool": "wait", "seconds": 1}] * 7, "non-empty"),
    ],
)
def test_sequence_rejects_bad_steps_without_executing_any(steps, message):
    emb, p, _ = policy([sequence(steps)], action_sequence=True)
    with pytest.raises(ValueError, match=message):
        p.act(emb.observe(), Feedback())
    assert p._sequence_targets is None
    assert not p.session.pending
    assert p.drain_usage().calls == 1


def test_sequence_bounds_apply_per_step_not_to_the_total():
    emb, p, _ = policy(
        [
            sequence(
                [
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0.08, 0, 0],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0.08, 0, 0],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                ]
            )
        ],
        action_sequence=True,
    )
    decision = p.act(emb.observe(), Feedback())
    assert decision.kind == "execute_sequence" and len(p._sequence_targets) == 2


# ---- sequence execution ----------------------------------------------------------


def report(target, outcome, travel=0.0):
    return ExecutionReport(
        target_id=target.id,
        outcome=outcome,
        t_start=0,
        t_end=1,
        final_tcp=target.pose,
        residual_m=0,
        residual_axis="x",
        residual_rot_rad=0,
        executed_lin_m=0,
        executed_rot_rad=0,
        reason=None,
        gripper_travel_end=travel,
    )


def three_step_policy():
    return policy(
        [
            sequence(
                [
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0, 0, -0.03],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                    {"tool": "set_gripper", "action": "close"},
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0, 0, 0.05],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                ]
            ),
            reply(),
        ],
        action_sequence=True,
    )


def test_sequence_stops_at_blocked_step_and_reports_it():
    emb, p, fake = three_step_policy()
    decision = p.act(emb.observe(), Feedback())
    executed = []

    def execute(target):
        executed.append(target)
        return report(
            target, ExecutionOutcome.BLOCKED if len(executed) == 1 else ExecutionOutcome.REACHED
        )

    p.execute_decision(decision, execute)
    assert len(executed) == 1
    summary = {k: v for k, v in p._sequence_result.items() if k != "steps"}
    assert summary == {
        "planned": 3,
        "executed": 1,
        "stopped_after_step": 0,
        "stop_reason": "step 0 blocked",
        "interrupted": False,
    }
    p.act(emb.observe(), Feedback(structured={"outcomes": ["blocked"]}))
    receipt = fake.requests[1]["input"][-2]["result"][0]["text"]
    assert '"stopped_after_step": 0' in receipt and "step 0 blocked" in receipt


def test_sequence_stops_when_close_ends_on_nothing():
    emb, p, _ = three_step_policy()
    decision = p.act(emb.observe(), Feedback())
    executed = []

    def execute(target):
        executed.append(target)
        travel = 0.98 if target.gripper.action == "close" else 0.0
        return report(target, ExecutionOutcome.REACHED, travel)

    p.execute_decision(decision, execute)
    assert len(executed) == 2  # the lift after an empty close is not executed
    assert p._sequence_result["stop_reason"] == "step 1 closed on nothing"


def carry_policy(**config):
    return policy(
        [
            sequence(
                [
                    {"tool": "set_gripper", "action": "close"},
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0, 0, 0.05],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0.05, 0, 0],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0, 0.05, 0],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                ]
            ),
            reply(),
        ],
        action_sequence=True,
        **config,
    )


def test_sequence_grasp_guard_stops_when_the_jaw_closes_further_while_carrying():
    emb, p, _ = carry_policy(sequence_grasp_guard=True)
    decision = p.act(emb.observe(), Feedback())
    travels = iter([0.60, 0.61, 0.78, 0.0])  # close on something, carry, lose it

    def execute(target):
        return report(target, ExecutionOutcome.REACHED, next(travels))

    p.execute_decision(decision, execute)
    assert p._sequence_result["executed"] == 3 and p._sequence_result["stopped_after_step"] == 2
    assert p._sequence_result["stop_reason"].startswith(
        "step 2 grasp lost: jaw travel rose from 0.60 to 0.78"
    )


def test_sequence_grasp_guard_is_off_by_default_and_ignores_small_jitter():
    for config, travels, executed in (
        ({}, [0.60, 0.61, 0.78, 0.0], 4),  # default: no guard
        ({"sequence_grasp_guard": True}, [0.60, 0.65, 0.69, 0.68], 4),  # under the 0.10 rise
        ({"sequence_grasp_guard": True}, [0.98, 0.0, 0.0, 0.0], 1),  # empty close still stops
    ):
        emb, p, _ = carry_policy(**config)
        decision = p.act(emb.observe(), Feedback())
        it = iter(travels)
        p.execute_decision(
            decision, lambda target, it=it: report(target, ExecutionOutcome.REACHED, next(it))
        )
        assert p._sequence_result["executed"] == executed, (config, travels)


def test_sequence_grasp_guard_uses_the_held_travel_at_sequence_start():
    from dataclasses import replace

    emb, p, _ = policy(
        [
            sequence(
                [
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0, 0, 0.05],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0.05, 0, 0],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                ]
            ),
            reply(),
        ],
        action_sequence=True,
        sequence_grasp_guard=True,
    )
    obs = emb.observe()
    obs = replace(obs, gripper=replace(obs.gripper, travel=0.55))
    decision = p.act(obs, Feedback())
    travels = iter([0.56, 0.90])
    p.execute_decision(
        decision, lambda target: report(target, ExecutionOutcome.REACHED, next(travels))
    )
    assert p._sequence_result["executed"] == 2 and "grasp lost" in p._sequence_result["stop_reason"]


def test_sequence_runs_to_completion_when_every_step_reaches():
    emb, p, _ = three_step_policy()
    decision = p.act(emb.observe(), Feedback())
    p.execute_decision(
        decision,
        lambda t: report(t, ExecutionOutcome.REACHED, 0.5 if t.gripper.action == "close" else 0.0),
    )
    summary = {k: v for k, v in p._sequence_result.items() if k != "steps"}
    assert summary == {
        "planned": 3,
        "executed": 3,
        "stopped_after_step": None,
        "stop_reason": None,
        "interrupted": False,
    }
    assert p._sequence_targets is None


def test_sequence_records_interruption_by_a_loop_limit():
    emb, p, _ = three_step_policy()
    decision = p.act(emb.observe(), Feedback())

    def execute(target):
        raise RuntimeError("limit")

    with pytest.raises(RuntimeError):
        p.execute_decision(decision, execute)
    assert p._sequence_result["interrupted"] and p._sequence_result["executed"] == 0
    p.finalize(emb.observe(), Feedback(structured={}), "simulated_horizon")
    assert not p.session.pending


def test_single_tool_decisions_are_untouched_by_execute_decision():
    emb, p, _ = policy(
        [
            reply(
                "move_relative",
                {"delta_xyz_m": [0, 0, 0.01], "rotation_vector_rad": [0, 0, 0], "note": "one"},
            )
        ],
        action_sequence=True,
    )
    decision = p.act(emb.observe(), Feedback())
    assert len(decision.targets) == 1
    p.execute_decision(decision, lambda t: pytest.fail("sequence executor ran"))


def test_sequence_executes_through_the_real_loop_on_the_mock():
    emb, p, _ = policy(
        [
            sequence(
                [
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0, 0, 0.02],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                    {"tool": "wait", "seconds": 0.2},
                    {"tool": "set_gripper", "action": "close"},
                    {
                        "tool": "move_relative",
                        "delta_xyz_m": [0, 0, 0.02],
                        "rotation_vector_rad": [0, 0, 0],
                    },
                ]
            ),
            reply("done"),
        ],
        action_sequence=True,
    )
    result = run_episode(
        emb, p, task="t", limits=Limits(horizon_s=60, max_decisions=5, max_dollars=1)
    )
    assert result.decisions == 2 and result.end_reason == "policy_done"
    # The close happened in free air, so the sequence stopped before the final lift.
    assert result.servo_steps == 3
    assert p._sequence_result is None  # cleared for the next turn after the receipt


# ---- depth probe -----------------------------------------------------------------


def camera(depth, K=None, pos=(0, 0, 1)):
    h, w = depth.shape
    K = (
        np.array([[100, 0, (w - 1) / 2], [0, 100, (h - 1) / 2], [0, 0, 1]], float)
        if K is None
        else K
    )
    # Camera looks straight down: OpenCV +z forward maps to root -z.
    pose = Pose(np.array(pos, float), np.array([0, 1, 0, 0], float))
    return CameraFrame(
        "scene", np.zeros((h, w, 3), np.uint8), 2.5, depth=depth.astype(np.float32), K=K, pose=pose
    )


def test_probe_returns_surface_point_in_root_frame_with_table_height():
    depth = np.full((21, 21), 0.9, np.float32)
    frame = camera(depth)
    result = probe_depth_pixel(frame, 0.5, 0.5, 2.5, table_z=0.1)
    np.testing.assert_allclose(result["surface_point_xyz_m"], [0, 0, 0.1], atol=1e-6)
    assert result["height_above_table_m"] == pytest.approx(0.0, abs=1e-6)
    assert result["valid_depth_fraction"] == 1 and not result["on_depth_edge"]
    assert "not an object centre" in result["limitation"]


def test_probe_off_centre_pixel_uses_intrinsics():
    depth = np.full((21, 21), 1.0, np.float32)
    frame = camera(depth)
    result = probe_depth_pixel(frame, 1.0, 0.5, 2.5)
    # Column 20 is 10 px right of centre at f=100 and depth 1 m: 0.1 m in camera +x,
    # which the downward camera maps to root +x.
    np.testing.assert_allclose(result["surface_point_xyz_m"], [0.1, 0, 0], atol=1e-9)
    assert result["pixel"] == [20, 10]


def test_probe_flags_depth_edges_and_rejects_empty_windows():
    depth = np.full((21, 21), 1.0, np.float32)
    depth[:, 11:] = 0.8  # an object edge through the window
    assert probe_depth_pixel(camera(depth), 0.5, 0.5, 2.5)["on_depth_edge"]
    depth[:] = np.nan
    with pytest.raises(ValueError, match="no valid depth"):
        probe_depth_pixel(camera(depth), 0.5, 0.5, 2.5)
    depth[:] = 1.0
    depth[10, 10] = np.nan
    result = probe_depth_pixel(camera(depth), 0.5, 0.5, 2.5)
    assert result["valid_depth_fraction"] == pytest.approx(24 / 25)
    assert result["depth_m"] == pytest.approx(1.0)


def test_probe_rejects_stale_capture_missing_depth_and_shape_mismatch():
    depth = np.full((21, 21), 1.0, np.float32)
    frame = camera(depth)
    with pytest.raises(ValueError, match="stale"):
        probe_depth_pixel(frame, 0.5, 0.5, 2.0)
    frame.depth = None
    with pytest.raises(ValueError, match="no depth"):
        probe_depth_pixel(frame, 0.5, 0.5, 2.5)
    frame.depth = depth[:10]
    with pytest.raises(ValueError, match="resolution"):
        probe_depth_pixel(frame, 0.5, 0.5, 2.5)


def test_probe_tool_receipt_carries_surface_geometry_and_no_motion():
    emb, p, fake = policy(
        [
            reply(
                "probe_point",
                {
                    "camera": "scene",
                    "u": 0.5,
                    "v": 0.5,
                    "capture_time_s": 0.0,
                    "note": "banana top",
                },
            ),
            reply(),
        ],
        depth_probe=True,
        geometry_feedback=True,
    )
    obs = emb.observe()
    assert obs.scene.depth is not None
    decision = p.act(obs, Feedback())
    assert decision.targets == [] and decision.kind == "probe_point"
    geometry = decision.payload["geometry_result"]
    assert "surface_point_xyz_m" in geometry and "height_above_table_m" in geometry
    assert geometry["reprojection"]["views"]["scene"]["inside_image"]
    p.act(emb.observe(), Feedback())
    assert "surface_point_xyz_m" in fake.requests[1]["input"][-2]["result"][0]["text"]
    assert any(t["name"] == "probe_point" for t in fake.requests[0]["tools"])


def test_probe_tool_absent_without_depth_camera():
    emb, p, fake = policy([reply()], depth_probe=True)
    obs = emb.observe()
    obs.scene.depth = None
    p.act(obs, Feedback())
    assert not any(t["name"] == "probe_point" for t in fake.requests[0]["tools"])


def test_sequence_uses_achieved_pose_for_gripper_and_relative_lift():
    from dataclasses import replace

    emb, p, _ = three_step_policy()
    decision = p.act(emb.observe(), Feedback())
    executed = []
    actual = None

    def execute(target):
        nonlocal actual
        executed.append(target)
        if len(executed) == 1:
            actual = Pose(target.pose.pos + np.array([0.003, 0, 0.002]), target.pose.quat)
            return replace(report(target, ExecutionOutcome.REACHED), final_tcp=actual)
        return report(target, ExecutionOutcome.REACHED, travel=0.5)

    p.execute_decision(decision, execute)
    np.testing.assert_allclose(executed[1].pose.pos, actual.pos)
    np.testing.assert_allclose(executed[2].pose.pos, actual.pos + [0, 0, 0.05])
    assert len(p._sequence_result["steps"]) == 3


def test_sequence_accepts_null_step_fields_from_strict_transports():
    emb, p, _ = policy(
        [
            sequence(
                [
                    {
                        "tool": "set_gripper",
                        "action": "close",
                        "delta_xyz_m": None,
                        "seconds": None,
                        "xyz_m": None,
                        "quat_wxyz": None,
                        "rotation_vector_rad": None,
                    },
                    {"tool": "wait", "seconds": 0.5, "action": None},
                ]
            )
        ],
        action_sequence=True,
    )
    decision = p.act(emb.observe(), Feedback())
    assert decision.kind == "execute_sequence" and len(p._sequence_targets) == 2


def test_malformed_arguments_marker_becomes_a_readable_receipt():
    import json as _json

    emb, p, fake = policy(
        [reply("wait", {"malformed_arguments_json": "Expecting value: line 1 column 5"}), reply()],
        action_sequence=True,
    )
    with pytest.raises(ValueError, match="arguments were not valid JSON: Expecting value"):
        p.act(emb.observe(), Feedback())
    p.act(emb.observe(), Feedback(structured={"error": "arguments were not valid JSON"}))
    receipts = [
        _json.dumps(item)
        for item in fake.requests[1]["input"]
        if item.get("type") == "function_result"
    ]
    assert receipts and "arguments were not valid JSON" in receipts[0]
    assert "exactly the tool" in receipts[0]


def test_unviewed_images_marker_becomes_a_readable_receipt():
    import json as _json

    note = (
        "the proposal was made without viewing 7-scene_right.png; "
        "view every current image, then propose"
    )
    emb, p, fake = policy(
        [reply("wait", {"images_not_viewed": note}), reply()], action_sequence=True
    )
    with pytest.raises(
        ValueError,
        match="proposal rejected: the proposal was made without viewing 7-scene_right.png",
    ):
        p.act(emb.observe(), Feedback())
    p.act(emb.observe(), Feedback(structured={"error": "proposal rejected"}))
    receipts = [
        _json.dumps(item)
        for item in fake.requests[1]["input"]
        if item.get("type") == "function_result"
    ]
    assert receipts and "proposal rejected" in receipts[0]
