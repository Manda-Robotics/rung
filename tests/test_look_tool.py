"""The opt-in ``look`` tool on the direct interface (rung.interfaces.look).

Off by default: every existing profile must send the model exactly what it did before.
On: the planned wrist-camera viewpoint must look at the requested point, stay inside the
declared limits, stop like a guarded sequence when blocked, and show the model the view.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from rung.backends.gemini_session import GeminiSession, TokenPrices
from rung.contracts.action import ExecutionOutcome
from rung.contracts.frames import Pose, quat_angle_between, quat_from_axis_angle, quat_mul
from rung.contracts.observation import CameraFrame
from rung.interfaces import look as geometry
from rung.interfaces.direct import DirectConfig, DirectPolicy, tools
from rung.interfaces.table_projection import project_table_pixel
from rung.loop import Limits, run_episode
from rung.mock import SCENES
from rung.mock.embodiment import MockConfig
from rung.policy import Feedback
from tests.direct_regression_scenarios import FIXTURE, PROFILES, _reply, request_digest
from tests.test_direct_sequence_and_depth import report
from tests.test_direct_session import Replies

TARGET = [0.5, 0.1, 0.03]
BANANA = [0.42, -0.13, 0.018]


def look_args(**overrides):
    # 45 degrees: the mock's palm is an axis-aligned box that fills a steep wrist view.
    args = {
        "target_xyz_m": BANANA,
        "distance_m": 0.25,
        "elevation_deg": 45,
        "azimuth_deg": 180,
        "return_to_previous_pose": True,
        "note": "find the bowl",
    }
    args.update(overrides)
    return {k: v for k, v in args.items() if v is not ...}


def setup(replies, calibrated=True, scene="banana_in_bowl", **config):
    fake = Replies(
        [_reply(name, {"note": "n", **args}, i) for i, (name, args) in enumerate(replies)]
    )
    session = GeminiSession(
        "test", TokenPrices(1, 2, 0.1, "test"), client=SimpleNamespace(interactions=fake)
    )
    emb = SCENES[scene](MockConfig(wrist_calibration=calibrated))
    policy = DirectPolicy(session, DirectConfig(**{"look_tool": True, **config}))
    return emb, policy, fake


def run(replies, decisions=None, **kwargs):
    emb, policy, fake = setup(replies, **kwargs)
    start = emb.reset()
    result = run_episode(
        emb,
        policy,
        task="t",
        limits=Limits(horizon_s=60, max_decisions=decisions or len(replies), max_dollars=1),
    )
    return emb, policy, fake, result, start


def receipt(request, key="look_result"):
    """The function result the model got for its previous call, parsed."""
    for entry in reversed(request["input"]):
        if entry.get("type") == "function_result":
            result = json.loads(entry["result"][0]["text"])
            return result if key is None else result.get(key)
    raise AssertionError("no function result in request")


def new_turn(request):
    return request["input"][-1]["content"]


def tool_names(request):
    return {t.get("name") for t in request["tools"]}


# ---- default off: byte-identical -------------------------------------------------


def test_existing_profiles_send_byte_identical_requests():
    """Digests recorded at f27e8f8, before the look tool existed (frozen proof included)."""
    expected = json.loads(FIXTURE.read_text())["digests"]
    assert set(expected) == set(PROFILES)
    for name, fields in PROFILES.items():
        assert request_digest(fields) == expected[name], name
        assert request_digest({**fields, "look_tool": False}) == expected[name], name


def test_look_is_opt_in_in_tools_prompt_and_receipts():
    assert DirectConfig().look_tool is False
    assert "look" not in {t.name for t in tools("both", action_sequence=True)}
    off = setup([("observe", {}), ("done", {})], look_tool=False)
    run_episode(off[0], off[1], task="t", limits=Limits(horizon_s=60, max_decisions=2))
    on = setup([("observe", {}), ("done", {})])
    run_episode(on[0], on[1], task="t", limits=Limits(horizon_s=60, max_decisions=2))
    first_off, first_on = off[2].requests[0], on[2].requests[0]
    assert "look" not in tool_names(first_off) and "look" in tool_names(first_on)
    assert "look moves the wrist camera" not in first_off["system_instruction"]
    assert "look moves the wrist camera" in first_on["system_instruction"]
    assert "look_result" not in receipt(off[2].requests[1], None)
    assert receipt(on[2].requests[1]) is None  # observe: key present, nothing looked


def test_look_needs_a_wrist_camera_to_be_offered():
    fake = Replies([_reply("done", {"note": "n"}, 0)])
    session = GeminiSession(
        "test", TokenPrices(1, 2, 0.1, "test"), client=SimpleNamespace(interactions=fake)
    )
    emb = SCENES["banana_in_bowl"](MockConfig(with_wrist=False))
    run_episode(
        emb,
        DirectPolicy(session, DirectConfig(look_tool=True)),
        task="t",
        limits=Limits(horizon_s=60, max_decisions=1),
    )
    assert "look" not in tool_names(fake.requests[0])


def test_look_spec_offers_pixels_only_with_calibrated_cameras():
    spec = next(t for t in tools(look_tool=True, look_cameras=[]) if t.name == "look")
    assert "target_pixel" not in spec.parameters["properties"]
    assert spec.parameters["required"] == ["note"]
    spec = next(t for t in tools(look_tool=True, look_cameras=["scene"]) if t.name == "look")
    pixel = spec.parameters["properties"]["target_pixel"]
    assert pixel["properties"]["camera"]["enum"] == ["scene"]


def test_config_bounds_look_steps():
    with pytest.raises(ValueError):
        DirectConfig(look_max_steps=0)
    with pytest.raises(ValueError):
        DirectConfig(look_max_steps=True)


# ---- pose computation ------------------------------------------------------------


def workspace():
    return SCENES["banana_in_bowl"]().capabilities.workspace


DOWN = Pose(np.array([0.42, 0.0, 0.32]), np.array([0.0, 1.0, 0.0, 0.0]))  # approach -z


@pytest.mark.parametrize("elevation, azimuth", [(90, 0), (60, 180), (45, 90), (30, -135)])
def test_approach_axis_plan_points_the_tcp_at_the_target(elevation, azimuth):
    plan = geometry.plan_look(
        DOWN, None, TARGET, 0.25, elevation, azimuth, workspace(), 0.10, 0.50, 32
    )
    assert plan.view_model == geometry.APPROACH_AXIS and plan.notes
    eye = geometry.viewpoint(TARGET, 0.25, elevation, azimuth)
    np.testing.assert_allclose(plan.look_tcp.pos, eye, atol=1e-9)
    approach = plan.look_tcp.rot[:, 2]
    np.testing.assert_allclose(approach, (np.array(TARGET) - eye) / 0.25, atol=1e-9)
    np.testing.assert_allclose(plan.waypoints[-1].pos, plan.look_tcp.pos, atol=1e-12)
    assert quat_angle_between(plan.waypoints[-1].quat, plan.look_tcp.quat) < 1e-6


def test_straight_down_look_keeps_a_straight_down_hand_unrotated():
    plan = geometry.plan_look(DOWN, None, [0.5, 0.1, 0.03], 0.2, 90, 0, workspace(), 0.10, 0.50, 32)
    assert quat_angle_between(plan.look_tcp.quat, DOWN.quat) < 1e-9
    np.testing.assert_allclose(plan.look_tcp.pos, [0.5, 0.1, 0.23], atol=1e-9)


def test_calibrated_plan_puts_the_tilted_offset_camera_on_the_target():
    tilt = quat_from_axis_angle([1, 0, 0], 0.3)
    mount = Pose(np.array([0.02, -0.05, -0.12]), tilt)  # camera in the TCP frame
    frame = CameraFrame(
        "wrist",
        np.zeros((8, 8, 3), np.uint8),
        0.0,
        K=np.array([[4.0, 0, 3.5], [0, 4.0, 3.5], [0, 0, 1]]),
        pose=DOWN.compose(mount),
        state_at_capture=DOWN,
    )
    plan = geometry.plan_look(DOWN, frame, TARGET, 0.3, 50, 160, workspace(), 0.10, 0.50, 32)
    assert plan.view_model == geometry.CALIBRATED and not plan.notes
    camera = plan.look_tcp.compose(mount)
    eye = geometry.viewpoint(TARGET, 0.3, 50, 160)
    np.testing.assert_allclose(camera.pos, eye, atol=1e-9)
    np.testing.assert_allclose(camera.rot[:, 2], (np.array(TARGET) - eye) / 0.3, atol=1e-9)
    looked = CameraFrame("wrist", frame.rgb, 0.0, K=frame.K, pose=camera)
    assert geometry.expected_pixel(looked, TARGET)["normalized_uv"] == [0.5, 0.5]
    # The camera turned by the smallest rotation that moves its axis onto the target.
    before = DOWN.compose(mount).rot[:, 2]
    turned = quat_angle_between(DOWN.compose(mount).quat, camera.quat)
    assert turned == pytest.approx(math.acos(np.clip(before @ camera.rot[:, 2], -1, 1)))


def test_align_rotation_handles_opposite_axes():
    r = geometry.align_rotation([0, 0, 1], [0, 0, -1], fallback_axis=[1, 0, 0])
    np.testing.assert_allclose(r @ [0, 0, 1], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(geometry.align_rotation([1, 0, 0], [1, 0, 0]), np.eye(3))


def test_path_rises_before_translating_and_keeps_every_step_in_bounds():
    low = Pose(
        np.array([0.30, -0.25, 0.08]), quat_mul(quat_from_axis_angle([0, 0, 1], 0.4), DOWN.quat)
    )
    plan = geometry.plan_look(
        low, None, [0.55, 0.25, 0.03], 0.22, 30, 180, workspace(), 0.10, 0.50, 32
    )
    assert plan.transit_z == pytest.approx(workspace().table_z + geometry.TRANSIT_CLEARANCE_M)
    previous = low
    for pose in plan.waypoints:
        moved_xy = np.linalg.norm(pose.pos[:2] - previous.pos[:2]) > 1e-9
        if moved_xy:  # horizontal motion only at the transit height
            assert pose.pos[2] == pytest.approx(plan.transit_z)
            assert previous.pos[2] == pytest.approx(plan.transit_z)
        step_m = 0.10 - geometry.STEP_TRANSLATION_MARGIN_M
        step_rad = min(
            0.50 - geometry.STEP_ROTATION_MARGIN_RAD, geometry.LOOK_STEP_MAX_ROTATION_RAD
        )
        assert np.linalg.norm(pose.pos - previous.pos) <= step_m + 1e-9
        assert quat_angle_between(previous.quat, pose.quat) <= step_rad + 1e-9
        assert workspace().contains(pose.pos)
        previous = pose
    # The first step is a pure lift that keeps the orientation.
    first = plan.waypoints[0]
    np.testing.assert_allclose(first.pos[:2], low.pos[:2])
    assert quat_angle_between(first.quat, low.quat) < 1e-9


def _phases(start, waypoints):
    """Label each step lift, rotate, translate or descend; fail on any mixed step."""
    labels, previous = [], start
    for pose in waypoints:
        d = pose.pos - previous.pos
        turned = quat_angle_between(previous.quat, pose.quat) > 1e-9
        horizontal = np.linalg.norm(d[:2]) > 1e-9
        vertical = abs(d[2]) > 1e-9
        assert turned + horizontal + vertical == 1, (d, turned)
        labels.append(
            "rotate" if turned else "translate" if horizontal else "lift" if d[2] > 0 else "descend"
        )
        previous = pose
    return [k for i, k in enumerate(labels) if i == 0 or labels[i - 1] != k]


def test_legs_are_lift_rotate_translate_descend_and_never_mixed():
    low = Pose(np.array([0.30, -0.25, 0.08]), DOWN.quat)
    plan = geometry.plan_look(
        low, None, [0.55, 0.25, 0.03], 0.25, 25, 180, workspace(), 0.10, 0.50, 32
    )
    assert plan.look_tcp.pos[2] < plan.transit_z
    assert _phases(low, plan.waypoints) == ["lift", "rotate", "translate", "descend"]
    rotating = [p for p in plan.waypoints if quat_angle_between(p.quat, low.quat) > 1e-9]
    assert all(p.pos[2] == pytest.approx(plan.transit_z) for p in rotating[:-1])


def test_live_look_4_geometry_rotates_alone_in_capped_steps():
    """Live look #4: start [0.41, 0.11, 0.26], elevation 35 toward the base. Its first step
    combined about 0.17 m with 0.75 rad; now the turn is its own leg of small steps.
    Shifted 0.10 m in x to fit the mock's workspace."""
    start = Pose(np.array([0.51, 0.11, 0.26]), DOWN.quat)
    plan = geometry.plan_look(
        start, None, [0.50, 0.08, 0.09], 0.35, 35, 180, workspace(), 0.10, 1.57, 32
    )
    phases = _phases(start, plan.waypoints)
    assert phases.index("rotate") < phases.index("translate")
    total = quat_angle_between(start.quat, plan.look_tcp.quat)
    turns = [
        quat_angle_between(a.quat, b.quat)
        for a, b in zip([start, *plan.waypoints], plan.waypoints, strict=False)
    ]
    assert max(turns) <= geometry.LOOK_STEP_MAX_ROTATION_RAD + 1e-9
    assert sum(t > 1e-9 for t in turns) == math.ceil(total / geometry.LOOK_STEP_MAX_ROTATION_RAD)


def test_start_just_outside_the_workspace_moves_straight_in_first():
    ws = workspace()
    outside = Pose(np.array([0.42, 0.0, ws.max_xyz[2] + 0.04]), DOWN.quat)
    plan = geometry.plan_look(outside, None, TARGET, 0.25, 65, 180, ws, 0.10, 0.50, 32)
    entry = plan.waypoints[0]
    assert ws.contains(entry.pos)
    np.testing.assert_allclose(
        entry.pos, [0.42, 0.0, ws.max_xyz[2] - geometry.WORKSPACE_ENTRY_MARGIN_M]
    )
    assert quat_angle_between(entry.quat, outside.quat) < 1e-9
    assert all(ws.contains(p.pos) for p in plan.waypoints)
    assert any("outside the workspace" in n for n in plan.notes)
    assert plan.transit_z <= ws.max_xyz[2]


def test_start_far_outside_the_workspace_is_rejected_with_the_way_back():
    """Live look #6: the TCP was thrown to [-0.26, -0.75, 0.46]."""
    far = Pose(np.array([-0.26, -0.75, 0.46]), DOWN.quat)
    with pytest.raises(ValueError) as caught:
        geometry.plan_look(
            far, None, [0.422, 0.065, 0.09], 0.35, 65, 180, workspace(), 0.10, 0.50, 32
        )
    message = str(caught.value)
    assert "outside the declared workspace" in message
    assert "Nearest point inside: [0.16, -0.44, 0.46]" in message
    assert "nothing executed" in message and "no single call can bring" in message
    # Between one look step and one move_to: the receipt names the move_to that fixes it.
    ws = workspace()
    near = Pose(np.array([0.42, 0.0, ws.max_xyz[2] + 0.085]), DOWN.quat)
    with pytest.raises(
        ValueError,
        match=r"move_to \[0\.42, 0\.0, 0\.59\] with the "
        "current orientation first, then call look again",
    ):
        geometry.plan_look(near, None, TARGET, 0.25, 65, 180, ws, 0.10, 0.50, 32)


def test_out_of_workspace_start_through_the_policy():
    ws = workspace()
    emb, policy, _ = setup([("look", look_args(return_to_previous_pose=...))])
    policy.reset("t", emb.capabilities)
    start = emb.reset()
    start.tcp = Pose(np.array([0.42, 0.0, ws.max_xyz[2] + 0.04]), start.tcp.quat)
    decision = policy.act(start, Feedback())
    assert decision.kind == "look"
    first = policy._look_targets[0].pose
    assert ws.contains(first.pos) and first.pos[2] < ws.max_xyz[2]


# ---- defaults ----------------------------------------------------------------------


def test_spec_documents_defaults_and_side_views():
    spec = next(t for t in tools(look_tool=True, look_cameras=[]) if t.name == "look")
    assert "distance_m 0.3, elevation_deg 65, azimuth_deg 180" in spec.description
    assert "below 45 are side views" in spec.description
    assert "return_to_previous_pose true (default false)" in spec.description
    props = spec.parameters["properties"]
    assert "Default 0.3" in props["distance_m"]["description"]
    assert "side view" in props["elevation_deg"]["description"]


def test_omitted_viewpoint_fields_get_the_documented_defaults():
    emb, policy, _ = setup([("look", {"target_xyz_m": BANANA, "note": "n"})])
    policy.reset("t", emb.capabilities)
    decision = policy.act(emb.reset(), Feedback())
    look = decision.payload["look"]
    assert look["view"] == {"distance_m": 0.30, "elevation_deg": 65.0, "azimuth_deg": 180.0}
    assert look["defaults_used"] == [
        "azimuth_deg",
        "distance_m",
        "elevation_deg",
        "return_to_previous_pose",
    ]
    assert look["return_to_previous_pose"] is False and policy._look_return is False
    explicit = setup([("look", look_args())])
    explicit[1].reset("t", explicit[0].capabilities)
    assert "defaults_used" not in explicit[1].act(explicit[0].reset(), Feedback()).payload["look"]


def test_return_defaults_off_on_the_mock():
    emb, policy, fake, _, _ = run([("look", look_args(return_to_previous_pose=...)), ("done", {})])
    look = receipt(fake.requests[1])
    assert look["reached_look_pose"] and "returned" not in look
    assert look["defaults_used"] == ["return_to_previous_pose"]


# ---- limits ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(distance_m=0.05), "distance_m"),
        (dict(distance_m=0.9), "distance_m"),
        (dict(elevation_deg=5), "elevation_deg"),
        (dict(azimuth_deg=270), "azimuth_deg"),
        (dict(distance_m=0.6, elevation_deg=90), "outside the reachable workspace"),
        (dict(distance_m=0.10, elevation_deg=20, target=[0.5, 0.1, 0.0]), "of the table"),
        (dict(target=[2.0, 0, 0.03]), "target outside"),
        (dict(max_steps=2), "bounded steps"),
    ],
)
def test_plan_rejects_unsafe_or_unreachable_viewpoints(kwargs, message):
    args = dict(target=TARGET, distance_m=0.25, elevation_deg=60, azimuth_deg=180, max_steps=32)
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        geometry.plan_look(
            DOWN,
            None,
            args["target"],
            args["distance_m"],
            args["elevation_deg"],
            args["azimuth_deg"],
            workspace(),
            0.10,
            0.50,
            args["max_steps"],
        )


def test_calibrated_camera_behind_the_hand_keeps_fingers_off_the_target():
    mount = Pose(np.array([0.0, 0.0, -0.10]), np.array([1.0, 0, 0, 0]))
    frame = CameraFrame(
        "wrist",
        np.zeros((8, 8, 3), np.uint8),
        0.0,
        K=np.eye(3),
        pose=DOWN.compose(mount),
        state_at_capture=DOWN,
    )
    with pytest.raises(ValueError, match="of the target"):
        geometry.plan_look(DOWN, frame, [0.5, 0.1, 0.10], 0.12, 90, 0, workspace(), 0.10, 0.50, 32)


@pytest.mark.parametrize(
    "args, message",
    [
        (look_args(distance_m=0.05), "distance_m"),
        (look_args(target_xyz_m=...), "exactly one"),
        (
            look_args(target_pixel={"camera": "scene", "u": 0.5, "v": 0.5, "capture_time_s": 0.0}),
            "exactly one",
        ),
        (look_args(return_to_previous_pose="yes"), "true or false"),
        (look_args(target_xyz_m=[0.5, 0.1]), "3 numbers"),
    ],
)
def test_rejected_look_executes_nothing_and_resolves_the_call(args, message):
    emb, policy, fake = setup([("look", args)])
    policy.reset("t", emb.capabilities)
    with pytest.raises(ValueError, match=message):
        policy.act(emb.reset(), Feedback())
    assert policy._look_targets is None and not policy.session.pending


def test_strict_transport_nulls_are_ignored():
    emb, policy, _ = setup([("look", look_args(target_pixel=None))])
    policy.reset("t", emb.capabilities)
    decision = policy.act(emb.reset(), Feedback())
    assert decision.kind == "look" and decision.targets == []
    assert "target_pixel" not in decision.payload["arguments"]
    assert len(policy._look_targets) == decision.payload["look"]["planned_steps"]


def test_look_steps_carry_the_speed_limits_and_a_settle_dwell():
    emb, policy, _ = setup([("look", look_args())], max_speed_mps=0.2, max_ang_speed_rps=0.8)
    policy.reset("t", emb.capabilities)
    policy.act(emb.reset(), Feedback())
    targets = policy._look_targets
    assert all(t.max_lin_speed_mps == 0.2 and t.max_ang_speed_rps == 0.8 for t in targets)
    assert all(t.gripper.action == "hold" for t in targets)
    assert targets[-1].min_dwell_s == geometry.SETTLE_S
    assert all(t.min_dwell_s == 0 for t in targets[:-1])


# ---- execution, blocked looks and the receipt ------------------------------------


def test_look_receipt_and_return_on_the_mock():
    emb, policy, fake, result, start = run([("look", look_args()), ("done", {})])
    assert result.end_reason == "policy_done"
    look = receipt(fake.requests[1])
    assert look["view_model"] == geometry.CALIBRATED
    assert look["reached_look_pose"] and look["executed_steps"] == look["planned_steps"]
    np.testing.assert_allclose(
        look["achieved_tcp_xyz_m"], look["planned_look_tcp_xyz_m"], atol=1e-3
    )
    assert look["target_in_look_view"] == {
        "normalized_uv": [0.5, 0.5],
        "in_view": True,
        "camera_to_target_m": 0.25,
    }
    assert look["returned"] and look["return_stop_reason"] is None
    assert look["view_span_at_target_m"][0] > 0
    final = receipt(fake.requests[1], None)["measured_tcp_xyz_m"]
    np.testing.assert_allclose(final, start.tcp.pos, atol=1e-6)


def test_model_sees_the_look_view_before_the_current_frames():
    emb, policy, fake, _, start = run([("look", look_args()), ("done", {})])
    turn = new_turn(fake.requests[1])
    texts = [p["text"] for p in turn if p["type"] == "text"]
    images = [p for p in turn if p["type"] == "image"]
    assert any(t.startswith("LOOK VIEW: camera wrist") for t in texts)
    assert len(images) == 3  # look view, current scene, current wrist
    first_turn_images = [p for p in new_turn(fake.requests[0]) if p["type"] == "image"]
    assert len(first_turn_images) == 2
    look_view, wrist_now = images[0]["data"], images[2]["data"]
    assert look_view != wrist_now  # the arm went back; the look view is a different picture
    assert wrist_now == first_turn_images[1]["data"]  # and back where it started
    # Shown once: the following turn has no look view.
    emb, policy, fake, _, _ = run([("look", look_args()), ("observe", {}), ("done", {})])
    assert not any(
        p["type"] == "text" and p["text"].startswith("LOOK VIEW")
        for p in new_turn(fake.requests[2])
    )


def test_look_without_return_leaves_the_arm_at_the_viewpoint():
    emb, policy, fake, _, start = run(
        [("look", look_args(return_to_previous_pose=False)), ("done", {})]
    )
    look = receipt(fake.requests[1])
    assert "returned" not in look and look["frames"].startswith("the current observation")
    measured = receipt(fake.requests[1], None)["measured_tcp_xyz_m"]
    np.testing.assert_allclose(measured, look["planned_look_tcp_xyz_m"], atol=1e-3)
    turn = new_turn(fake.requests[1])
    assert not any(p["type"] == "text" and p["text"].startswith("LOOK VIEW") for p in turn)
    wrist_now = [p for p in turn if p["type"] == "image"][1]["data"]
    wrist_before = [p for p in new_turn(fake.requests[0]) if p["type"] == "image"][1]["data"]
    assert wrist_now != wrist_before


def test_blocked_look_stops_captures_where_it_stopped_and_retraces():
    emb, policy, _ = setup([("look", look_args(target_xyz_m=[0.55, 0.25, 0.03]))])
    policy.reset("t", emb.capabilities)
    start = emb.reset()
    decision = policy.act(start, Feedback())
    planned = list(policy._look_targets)
    assert len(planned) >= 3
    executed = []

    def execute(target):
        executed.append(target)
        blocked = len(executed) == 2
        outcome = ExecutionOutcome.BLOCKED if blocked else ExecutionOutcome.REACHED
        result = report(target, outcome)
        if blocked:  # stopped halfway along the step
            result.final_tcp = Pose(
                (executed[0].pose.pos + target.pose.pos) / 2, executed[0].pose.quat
            )
            result.reason = "blocked by bowl"
        return result

    observed = []
    policy.execute_decision(decision, execute, observe=lambda: observed.append(1) or start)
    look = policy._look_result
    assert look["executed_steps"] == 2 and not look["reached_look_pose"]
    assert look["stop_reason"] == "look step 1 blocked: blocked by bowl"
    assert observed == [1] and policy._look_snapshot is not None
    # Retrace: back to the first waypoint, then to the start pose.
    back = executed[2:]
    assert len(back) == 2 and look["return_steps"] == 2 and look["returned"]
    np.testing.assert_allclose(back[0].pose.pos, planned[0].pose.pos)
    np.testing.assert_allclose(back[1].pose.pos, start.tcp.pos)
    assert all(t.meta["phase"] == "look_return" for t in back)


def test_blocked_return_is_reported_not_hidden():
    emb, policy, _ = setup([("look", look_args())])
    policy.reset("t", emb.capabilities)
    start = emb.reset()
    decision = policy.act(start, Feedback())
    n = len(policy._look_targets)
    calls = []

    def execute(target):
        calls.append(target)
        outcome = ExecutionOutcome.EXPIRED if len(calls) == n + 1 else ExecutionOutcome.REACHED
        return report(target, outcome)

    policy.execute_decision(decision, execute, observe=lambda: start)
    look = policy._look_result
    assert look["reached_look_pose"] and look["returned"] is False
    assert look["return_stop_reason"].startswith("return step 0 expired")
    assert len(calls) == n + 1


def test_look_records_interruption_by_a_loop_limit():
    emb, policy, _ = setup([("look", look_args())])
    policy.reset("t", emb.capabilities)
    decision = policy.act(emb.reset(), Feedback())

    def execute(target):
        raise RuntimeError("horizon")

    with pytest.raises(RuntimeError):
        policy.execute_decision(decision, execute, observe=emb.observe)
    assert policy._look_result["interrupted"] is True


def test_real_mock_block_stops_the_look_and_reports_it():
    """A straight-down look onto a tall bottle: the descent runs the palm into it."""
    emb, policy, fake, result, _ = run(
        [
            (
                "look",
                look_args(
                    target_xyz_m=[0.40, -0.10, 0.0],
                    distance_m=0.12,
                    elevation_deg=90,
                    azimuth_deg=0,
                    return_to_previous_pose=True,
                ),
            ),
            ("done", {}),
        ],
        calibrated=False,
        scene="tall_bottle_and_crate",
    )
    look = receipt(fake.requests[1])
    assert look["view_model"] == geometry.APPROACH_AXIS
    assert not look["reached_look_pose"]
    assert "blocked" in look["stop_reason"] and "bottle" in look["stop_reason"]
    assert look["stop_reason"].startswith(f"look step {look['executed_steps'] - 1} blocked")
    assert look["achieved_tcp_xyz_m"][2] > 0.12  # stopped above the planned viewpoint
    assert look["returned"]  # retraced the way it came
    execution = receipt(fake.requests[1], None)["execution_feedback"]["outcomes"]
    assert "blocked" in execution


def test_pixel_target_resolves_through_the_table_plane():
    emb = SCENES["banana_in_bowl"](MockConfig(wrist_calibration=True))
    first = emb.reset()
    pixel = {"camera": "scene", "u": 0.55, "v": 0.6, "capture_time_s": first.t}
    emb, policy, fake, _, _ = run(
        [("look", look_args(target_xyz_m=..., target_pixel=pixel)), ("done", {})]
    )
    look = receipt(fake.requests[1])
    expected = project_table_pixel(first.scene, 0.55, 0.6, first.t, 0.0)
    np.testing.assert_allclose(
        look["target_xyz_m"], expected["table_intersection_xyz_m"], atol=1e-4
    )
    assert look["target"]["source"].startswith("table plane under scene pixel")


def test_finalize_carries_the_look_result():
    emb, policy, fake, result, _ = run([("look", look_args())], decisions=1)
    assert result.end_reason == "decision_limit"
    assert policy._look_result["returned"] is True
