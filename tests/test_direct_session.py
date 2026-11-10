import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rung.backends.base import TextBlock
from rung.backends.gemini_session import GeminiSession, TokenPrices
from rung.interfaces.direct import DirectPolicy, tools
from rung.mock import SCENES
from rung.policy import Feedback


class Replies:
    def __init__(self, replies):
        self.replies, self.requests = iter(replies), []

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return next(self.replies)


def reply(name="observe", args=None, extra=None):
    return {
        "status": "completed",
        "steps": [
            {"type": "thought", "signature": "opaque-keep-exactly", "summary": []},
            {
                "type": "function_call",
                "id": "call1",
                "name": name,
                "arguments": args or {"note": "look"},
            },
            *(extra or []),
        ],
        "usage": {
            "total_input_tokens": 100,
            "total_output_tokens": 10,
            "total_thought_tokens": 20,
            "total_cached_tokens": 40,
        },
    }


def session(replies, tmp_path=None):
    fake = Replies(replies)
    return GeminiSession(
        "test",
        TokenPrices(1, 2, 0.1, "test"),
        client=SimpleNamespace(interactions=fake),
        trace_dir=tmp_path,
    ), fake


def test_native_transaction_preserves_steps_and_receipt(tmp_path):
    s, fake = session([reply(), reply()], tmp_path)
    r = s.complete("system", [TextBlock("first")], tools())
    assert r.usage.calls == 1
    assert r.usage.dollars == pytest.approx((60 + 4 + 60) / 1e6)
    with pytest.raises(ValueError, match="resolve"):
        s.complete("system", [TextBlock("illegal")], tools())
    with pytest.raises(ValueError, match="IDs"):
        s.resolve({"wrong": {}})
    s.resolve({"call1": {"executed": True}})
    s.complete("system", [TextBlock("second")], tools())
    assert fake.requests[1]["input"][1] == reply()["steps"][0]
    assert fake.requests[1]["input"][3]["call_id"] == "call1"
    assert len(list(tmp_path.glob("*request.json"))) == 2
    assert fake.requests[0]["generation_config"]["tool_choice"] == "any"
    s.reset()
    assert s.history == [] and s.pending == []


def make_policy(replies):
    emb = SCENES["banana_in_bowl"]()
    s, fake = session(replies)
    p = DirectPolicy(s)
    p.reset("test task", emb.capabilities)
    return emb, p, fake


def test_spatial_cue_is_opt_in_and_does_not_add_a_planner():
    from rung.interfaces.direct import SYSTEM, DirectConfig

    emb, p, fake = make_policy([reply()])
    p.act(emb.observe(), Feedback())
    assert fake.requests[0]["system_instruction"] == SYSTEM
    emb, p, fake = make_policy([reply()])
    p.cfg.spatial_frame_cues = True
    p.act(emb.observe(), Feedback())
    assert "Keep root x approximately aligned" in fake.requests[0]["system_instruction"]
    assert "native score is available" in fake.requests[0]["system_instruction"]
    with pytest.raises(ValueError, match="one rig image"):
        DirectConfig(public_overlay=True, tcp_marker=True)


def test_root_rotation_and_separate_gripper():
    emb, p, _ = make_policy(
        [
            reply(
                "move_relative",
                {"delta_xyz_m": [0, 0, 0.01], "rotation_vector_rad": [0.2, 0, 0], "note": "tilt"},
            )
        ]
    )
    obs = emb.observe()
    decision = p.act(obs, Feedback())
    assert decision.targets[0].gripper.action == "hold"
    assert not np.allclose(decision.targets[0].pose.quat, obs.tcp.quat)
    np.testing.assert_allclose(decision.targets[0].pose.pos, obs.tcp.pos + [0, 0, 0.01])


def test_tcp_relative_motion_uses_measured_axes_and_fingertip_pivot():
    from rung.contracts.frames import Pose, quat_from_axis_angle

    emb, p, _ = make_policy(
        [
            reply(
                "move_relative_tcp",
                {
                    "delta_xyz_m": [0.01, 0, 0],
                    "rotation_vector_rad": [0.2, 0, 0],
                    "note": "local adjustment",
                },
            )
        ]
    )
    p.cfg.tcp_frame_motions = True
    obs = emb.observe()
    obs.tcp = Pose(obs.tcp.pos, quat_from_axis_angle([0, 0, 1], np.pi / 2))
    decision = p.act(obs, Feedback())
    target = decision.targets[0]
    np.testing.assert_allclose(target.pose.pos, obs.tcp.pos + [0, 0.01, 0], atol=1e-12)
    increment = Pose([0, 0, 0], quat_from_axis_angle([1, 0, 0], 0.2))
    np.testing.assert_allclose(target.pose.rot, obs.tcp.rot @ increment.rot, atol=1e-12)
    assert target.gripper.action == "hold"


def test_tcp_rotation_does_not_translate_fingertip():
    emb, p, _ = make_policy(
        [
            reply(
                "move_relative_tcp",
                {"delta_xyz_m": [0, 0, 0], "rotation_vector_rad": [0, 0.2, 0], "note": "pivot"},
            )
        ]
    )
    p.cfg.tcp_frame_motions = True
    obs = emb.observe()
    np.testing.assert_allclose(p.act(obs, Feedback()).targets[0].pose.pos, obs.tcp.pos)


def test_tcp_motion_opt_in_and_same_magnitude_bound():
    assert "move_relative_tcp" not in {t.name for t in tools("both")}
    assert "move_relative_tcp" in {t.name for t in tools("both", tcp_frame_motions=True)}
    emb, p, _ = make_policy(
        [
            reply(
                "move_relative_tcp",
                {
                    "delta_xyz_m": [0.08, 0.08, 0],
                    "rotation_vector_rad": [0, 0, 0],
                    "note": "too far",
                },
            )
        ]
    )
    p.cfg.tcp_frame_motions = True
    with pytest.raises(ValueError, match="out-of-bounds"):
        p.act(emb.observe(), Feedback())
    assert p.drain_usage().calls == 1
    assert not p.session.pending


@pytest.mark.parametrize(
    "args",
    [
        {"delta_xyz_m": [0, 0, 0.11], "rotation_vector_rad": [0, 0, 0], "note": "large"},
        {"delta_xyz_m": [True, 0, 0], "rotation_vector_rad": [0, 0, 0], "note": "boolean"},
        {"delta_xyz_m": [float("nan"), 0, 0], "rotation_vector_rad": [0, 0, 0], "note": "nan"},
    ],
)
def test_reject_bad_motion_and_return_error_receipt(args):
    emb, p, _ = make_policy([reply("move_relative", args)])
    with pytest.raises(ValueError):
        p.act(emb.observe(), Feedback())
    assert not p.session.pending
    assert p.session.history[-1]["is_error"]
    assert p.drain_usage().calls == 1


def test_multiple_calls_execute_none_and_resolve_all():
    emb, p, _ = make_policy(
        [
            reply(
                extra=[
                    {
                        "type": "function_call",
                        "id": "call2",
                        "name": "set_gripper",
                        "arguments": {"action": "close", "note": "close"},
                    }
                ]
            )
        ]
    )
    with pytest.raises(ValueError, match="exactly one"):
        p.act(emb.observe(), Feedback())
    assert [s["call_id"] for s in p.session.history if s["type"] == "function_result"] == [
        "call1",
        "call2",
    ]


def test_next_turn_contains_measured_receipt_and_no_grasp_claim():
    emb, p, fake = make_policy(
        [reply("set_gripper", {"action": "close", "note": "close"}), reply("done")]
    )
    obs = emb.observe()
    first = p.act(obs, Feedback())
    assert first.targets[0].gripper.action == "close"
    p.act(obs, Feedback(structured={"outcomes": ["expired"], "residual_m": [0.02]}))
    receipt = fake.requests[1]["input"][3]
    assert "expired" in receipt["result"][0]["text"]
    assert "holding" not in receipt["result"][0]["text"]


def test_unknown_usage_is_not_zero_cost():
    r = reply()
    del r["usage"]
    s, _ = session([r])
    with pytest.raises(ValueError, match="usage missing"):
        s.complete("system", [TextBlock("first")], tools())


def test_late_response_cannot_repopulate_reset_session(tmp_path):
    s, _ = session([], tmp_path)

    def reset_during_call(**kwargs):
        s.reset()
        return reply()

    s.client.interactions.create = reset_during_call
    with pytest.raises(ValueError, match="late response"):
        s.complete("system", [TextBlock("first")], tools())
    assert s.history == [] and s.pending == []
    assert len(list(tmp_path.glob("*late*"))) == 1


def test_image_window_does_not_truncate_native_steps():
    from rung.backends.base import ImageBlock

    s, fake = session([reply(), reply(), reply()])
    for _ in range(3):
        s.complete("system", [ImageBlock(b"fixture")], tools())
        s.resolve({"call1": {"executed": True}})
    request = fake.requests[2]["input"]
    observations = [x for x in request if x["type"] == "user_input"]
    assert observations[0]["content"][0]["type"] == "text"
    assert observations[1]["content"][0]["type"] == "image"
    assert observations[2]["content"][0]["type"] == "image"
    assert len([x for x in request if x["type"] == "thought"]) == 2
    assert len([x for x in request if x["type"] == "function_result"]) == 2
    assert s.history[0]["content"][0]["type"] == "image"


def test_model_call_budget_includes_rejected_arguments():
    from rung.loop import EndReason, Limits, run_episode

    emb, p, _ = make_policy(
        [
            reply(
                "move_relative",
                {"delta_xyz_m": [0, 0, 1], "rotation_vector_rad": [0, 0, 0], "note": "invalid"},
            )
        ]
    )
    result = run_episode(emb, p, task="banana_in_bowl", limits=Limits(max_model_calls=1))
    assert result.end_reason == EndReason.MODEL_CALLS
    assert result.usage.calls == 1
    assert result.servo_steps == 0


def test_last_allowed_call_still_executes_its_motion():
    from rung.loop import EndReason, Limits, run_episode

    emb, p, _ = make_policy(
        [
            reply(
                "move_relative",
                {"delta_xyz_m": [0, 0, 0.01], "rotation_vector_rad": [0, 0, 0], "note": "lift"},
            )
        ]
    )
    result = run_episode(emb, p, task="banana_in_bowl", limits=Limits(max_model_calls=1))
    assert result.end_reason == EndReason.MODEL_CALLS
    assert result.servo_steps == 1


def test_unknown_billing_ends_episode_without_retrying_as_free():
    from rung.loop import EndReason, Limits, run_episode

    r = reply()
    del r["usage"]
    emb, p, _ = make_policy([r])
    result = run_episode(emb, p, task="banana_in_bowl", limits=Limits(max_model_calls=5))
    assert result.end_reason == EndReason.UNKNOWN_COST
    assert result.usage.unknown_cost_calls == 1
    assert result.usage.calls == 1 and result.servo_steps == 0


def test_final_tool_receipt_is_archived_without_another_api_call(tmp_path):
    from rung.loop import Limits, run_episode

    emb = SCENES["banana_in_bowl"]()
    s, _ = session([reply("done")], tmp_path)
    p = DirectPolicy(s)
    run_episode(emb, p, task="banana_in_bowl", limits=Limits(max_model_calls=1))
    assert s.pending == []
    assert len(list(tmp_path.glob("*receipts.json"))) == 1
    assert s.history[-1]["call_id"] == "call1"


def test_image_archive_deduplicates_and_reconstructs_exact_body(tmp_path):
    import json

    from rung.backends.base import ImageBlock
    from rung.backends.native_session import read_request_archive

    s, fake = session([reply(), reply()], tmp_path)
    for _ in range(2):
        s.complete("system", [ImageBlock(b"fixture")], tools())
        s.resolve({"call1": {"executed": True}})
    paths = sorted(tmp_path.glob("*request.json"))
    archive = json.loads(paths[-1].read_text())
    assert archive["archive_format"] == "rung-native-v1"
    assert len(list((tmp_path / "blobs").glob("*"))) == 1
    reconstructed = read_request_archive(paths[-1])["request"]
    expected = {k: v for k, v in fake.requests[-1].items() if k != "timeout"}
    assert reconstructed == expected
    blob = next((tmp_path / "blobs").glob("*"))
    blob.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_request_archive(paths[-1])


def test_absolute_tool_reaches_same_target_with_same_limits():
    from rung.interfaces.direct import DirectConfig

    emb = SCENES["banana_in_bowl"]()
    obs = emb.observe()
    xyz = (obs.tcp.pos + [0, 0, 0.01]).tolist()
    s, _ = session(
        [reply("move_to", {"xyz_m": xyz, "quat_wxyz": obs.tcp.quat.tolist(), "note": "lift"})]
    )
    p = DirectPolicy(s, DirectConfig(motion_tools="absolute"))
    p.reset("test", emb.capabilities)
    decision = p.act(obs, Feedback())
    np.testing.assert_allclose(decision.targets[0].pose.pos, xyz)
    assert decision.targets[0].gripper.action == "hold"
    assert [t.name for t in tools("absolute")][0] == "move_to"


def test_context_price_tier_applies_to_entire_request():
    from rung.backends.native_session import TokenPrices

    prices = TokenPrices(2, 12, 0.2, "official pricing", 200000, (4, 18, 0.4))
    assert prices.rates_for(200000) == (2, 12, 0.2)
    assert prices.rates_for(200001) == (4, 18, 0.4)
    with pytest.raises(ValueError):
        TokenPrices(2, 12, 0.2, "official pricing", 200000, None)


@pytest.mark.parametrize("geometry_feedback", [False, True])
def test_projection_receipt_does_not_move_or_reveal_object_state(geometry_feedback):
    from rung.interfaces.direct import DirectConfig

    emb, policy, fake = make_policy(
        [
            reply(
                "project_table",
                {
                    "camera": "scene",
                    "u": 0.5,
                    "v": 0.5,
                    "capture_time_s": 0.0,
                    "note": "locate table ray",
                },
            ),
            reply("observe", {"note": "read projection"}),
        ]
    )
    policy.cfg = DirectConfig(table_projection=True, geometry_feedback=geometry_feedback)
    obs = emb.observe()
    # Use a qualified synthetic rig, not the mock camera's scene geometry.
    from rung.contracts.frames import Pose
    from rung.contracts.observation import CameraFrame

    obs.scene = CameraFrame(
        "scene",
        np.zeros((101, 101, 3), np.uint8),
        0.0,
        K=np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]]),
        pose=Pose([0.5, 0, 1], [0, 1, 0, 0]),
    )
    decision = policy.act(obs, Feedback())
    assert not decision.targets
    assert "geometry_result" in decision.payload
    if geometry_feedback:
        view = decision.payload["geometry_result"]["reprojection"]["views"]["scene"]
        np.testing.assert_allclose(view["normalized_uv"], [0.5, 0.5])
        assert view["inside_image"]
    else:
        assert "reprojection" not in decision.payload["geometry_result"]
    policy.act(obs, Feedback())
    receipt = fake.requests[1]["input"][-2]["result"]
    # Provider receipt is serialized JSON; no simulator object lookup enters it.
    assert "table_intersection_xyz_m" in str(receipt)
    assert "object_pose" not in str(receipt)


def test_direct_observe_and_wait_request_real_simulated_dwell():
    from rung.contracts.action import GripperAction

    emb, policy, _ = make_policy(
        [
            reply("observe", {"note": "fresh frame"}),
            reply("wait", {"seconds": 0.5, "note": "settle"}),
        ]
    )
    observe = policy.act(emb.observe(), Feedback())
    assert observe.targets[0].min_dwell_s == emb.capabilities.control_period_s
    wait = policy.act(emb.observe(), Feedback())
    assert wait.targets[0].min_dwell_s == 0.5
    assert wait.targets[0].gripper.action == GripperAction.HOLD
    report = emb.execute(wait.targets[0])
    assert report.t_end - report.t_start >= 0.5 - 1e-9


def test_actor_knows_the_remaining_common_call_and_physics_budgets():
    emb, policy, fake = make_policy(
        [reply("observe", {"note": "look"}), reply("observe", {"note": "look again"})]
    )
    policy.budgets = {"horizon_s": 50.0, "max_model_calls": 2, "time_mode": "paused"}
    obs = emb.observe()
    policy.act(obs, Feedback())
    obs.t += 0.25
    policy.act(obs, Feedback())

    def budget(request):
        return json.loads(request["input"][-1]["content"][0]["text"])["budgets"]

    assert budget(fake.requests[0])["model_calls_remaining_including_this_call"] == 2
    assert budget(fake.requests[1])["model_calls_remaining_including_this_call"] == 1
    assert budget(fake.requests[1])["simulated_remaining_s"] == 49.75
