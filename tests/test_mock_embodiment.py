"""The mock embodiment: rendering geometry, execution semantics, grasping and grading.

If these pass, the harness can be developed and regression tested with no simulator.
"""

from __future__ import annotations

import numpy as np
import pytest

from rung.contracts.action import ExecutionOutcome, GripperAction, GripperCommand, TCPTarget
from rung.contracts.frames import deproject, project
from rung.contracts.validate import validate_observation
from rung.embodiment import Embodiment, Gradeable
from rung.mock import SCENES, MockConfig, MockEmbodiment
from rung.mock.embodiment import banana_in_bowl, cluttered_bin, tall_bottle_and_crate
from rung.mock.world import grasp_check, top_down_pose


def _reach(emb: MockEmbodiment, pos, yaw=0.0, grip=GripperAction.HOLD, timeout=8.0):
    return emb.execute(
        TCPTarget(
            pose=top_down_pose(pos, yaw),
            gripper=GripperCommand(grip),
            max_lin_speed_mps=0.3,
            timeout_s=timeout,
        )
    )


def test_mock_satisfies_the_protocols():
    emb = banana_in_bowl()
    assert isinstance(emb, Embodiment)
    assert isinstance(emb, Gradeable)


def test_reset_is_deterministic_without_a_seed():
    a, b = banana_in_bowl(), banana_in_bowl()
    assert np.allclose(a.observe().scene.rgb, b.observe().scene.rgb)


def test_seeded_reset_perturbs_only_graspable_bodies():
    emb = banana_in_bowl()
    bowl_before = emb.scene.body("bowl").pos.copy()
    emb.reset(episode_seed=7)
    assert np.allclose(emb.scene.body("bowl").pos, bowl_before)
    assert not np.allclose(emb.scene.body("banana").pos, [0.42, -0.13, 0.018])


def test_observation_passes_its_own_contract():
    for make in SCENES.values():
        obs = make().observe()
        validate_observation(obs, make().capabilities)


def test_render_depth_matches_projected_geometry():
    """Deprojecting the pixel at a body's centre must land on that body's surface."""
    emb = banana_in_bowl()
    obs = emb.observe()
    banana = emb.scene.body("banana")
    top_centre = banana.pos + np.array([0.0, 0.0, banana.size[2] / 2])
    uv, z = project(top_centre[None], obs.scene.K, obs.scene.pose)
    u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
    d = obs.scene.depth[v, u]
    assert np.isfinite(d)
    point = deproject(np.array([u, v], dtype=float), float(d), obs.scene.K, obs.scene.pose)
    assert np.linalg.norm(point - top_centre) < 0.02


def test_depth_and_rgb_agree_on_occlusion():
    """Where the hand is in front of the table, depth is nearer than the table behind it."""
    emb = banana_in_bowl()
    _reach(emb, [0.42, -0.13, 0.10])
    obs = emb.observe()
    finite = np.isfinite(obs.scene.depth)
    assert finite.mean() > 0.5
    assert float(np.nanmin(obs.scene.depth)) < 0.9


def test_execute_reaches_and_reports_residual():
    emb = banana_in_bowl()
    rep = _reach(emb, [0.45, 0.05, 0.25])
    assert rep.outcome is ExecutionOutcome.REACHED
    assert rep.residual_m < 1e-6
    assert rep.duration_s > 0
    assert np.allclose(emb.observe().tcp.pos, [0.45, 0.05, 0.25], atol=1e-9)


def test_execute_expires_when_the_move_is_too_long_for_the_timeout():
    emb = banana_in_bowl()
    rep = emb.execute(
        TCPTarget(pose=top_down_pose([0.70, 0.40, 0.55]), max_lin_speed_mps=0.02, timeout_s=1.0)
    )
    assert rep.outcome is ExecutionOutcome.EXPIRED
    assert "split the move" in (rep.reason or "")
    assert rep.residual_m > 0.05


def test_execute_clamps_outside_the_workspace():
    emb = banana_in_bowl()
    rep = _reach(emb, [1.6, 0.0, 0.3])
    assert rep.outcome is ExecutionOutcome.CLAMPED
    assert rep.clamped_to is not None
    assert emb.observe().tcp.pos[0] <= emb.capabilities.workspace.max_xyz[0] + 1e-9


def test_execute_clamps_at_the_table():
    emb = banana_in_bowl()
    rep = _reach(emb, [0.30, 0.30, -0.05])
    assert rep.outcome is ExecutionOutcome.CLAMPED
    assert emb.observe().tcp.pos[2] >= emb.capabilities.workspace.table_z


def test_simulated_time_advances_only_with_execution():
    emb = banana_in_bowl()
    t0 = emb.observe().t
    emb.observe()
    assert emb.observe().t == t0
    _reach(emb, [0.45, 0.0, 0.30])
    assert emb.observe().t > t0


def test_grasp_requires_the_jaw_to_straddle_the_object():
    emb = banana_in_bowl()
    banana = emb.scene.body("banana")
    # aligned but hovering 15 cm up: the fingers are above the object
    high = top_down_pose(banana.pos + np.array([0, 0, 0.15]), yaw=banana.yaw)
    holds, why = grasp_check(high, 1.0, banana, emb.gripper_model)
    assert not holds and "above" in why
    # at the object, aligned with its short axis: grasp
    at = top_down_pose(banana.pos + np.array([0, 0, 0.005]), yaw=banana.yaw)
    holds, why = grasp_check(at, 1.0, banana, emb.gripper_model)
    assert holds, why


def test_grasp_yaw_must_align_across_the_long_axis():
    """A 17 cm banana yawed 0.35 rad measures 10 cm across a jaw held at yaw 0. Only a jaw
    turned to the body's own yaw fits."""
    emb = banana_in_bowl()
    banana = emb.scene.body("banana")
    misaligned = top_down_pose(banana.pos + np.array([0, 0, 0.005]), yaw=0.0)
    holds, why = grasp_check(misaligned, 1.0, banana, emb.gripper_model)
    assert not holds and "too wide" in why


def test_grasp_rejects_a_body_wider_than_the_jaw():
    emb = tall_bottle_and_crate()
    box = emb.scene.body("box")  # 14 x 11 cm, wider than the 8.5 cm jaw either way
    pose = top_down_pose(box.pos + np.array([0, 0, 0.01]))
    holds, why = grasp_check(pose, 1.0, box, emb.gripper_model)
    assert not holds and "too wide" in why


def test_closing_on_nothing_reports_fully_closed():
    emb = banana_in_bowl()
    _reach(emb, [0.30, 0.30, 0.06], grip=GripperAction.CLOSE)
    obs = emb.observe()
    assert obs.gripper.travel == pytest.approx(1.0)
    assert emb.robot.held is None
    assert emb.events[-1]["event"] == "closed_empty"


def test_held_object_follows_the_hand_and_falls_on_release():
    emb = banana_in_bowl()
    banana = emb.scene.body("banana")
    _reach(emb, banana.pos + np.array([0, 0, 0.12]), yaw=banana.yaw)
    _reach(emb, banana.pos + np.array([0, 0, 0.005]), yaw=banana.yaw, grip=GripperAction.CLOSE)
    assert emb.robot.held == "banana"
    assert 0.0 < emb.observe().gripper.travel < 1.0

    _reach(emb, [0.42, -0.13, 0.30])
    assert emb.scene.body("banana").pos[2] > 0.2

    _reach(emb, [0.30, -0.35, 0.30], grip=GripperAction.OPEN)
    assert emb.robot.held is None
    assert emb.scene.body("banana").pos[2] == pytest.approx(banana.size[2] / 2, abs=1e-6)


def test_scripted_pick_and_place_succeeds_and_grades():
    """The end-to-end path a scripted reference policy takes."""
    emb = banana_in_bowl()
    ok, _ = emb.grade()
    assert not ok

    banana = emb.scene.body("banana")
    bowl = emb.scene.body("bowl")
    _reach(emb, banana.pos + np.array([0, 0, 0.12]), yaw=banana.yaw)
    _reach(emb, banana.pos + np.array([0, 0, 0.005]), yaw=banana.yaw, grip=GripperAction.CLOSE)
    _reach(emb, banana.pos + np.array([0, 0, 0.20]), yaw=banana.yaw)
    _reach(emb, [bowl.pos[0], bowl.pos[1], 0.22], yaw=banana.yaw)
    _reach(emb, [bowl.pos[0], bowl.pos[1], 0.16], yaw=banana.yaw, grip=GripperAction.OPEN)

    ok, detail = emb.grade()
    assert ok, detail
    assert detail["inside_footprint"] and detail["below_rim"] and not detail["still_held"]


def test_grade_is_false_while_the_object_is_still_held():
    emb = banana_in_bowl()
    banana, bowl = emb.scene.body("banana"), emb.scene.body("bowl")
    _reach(emb, banana.pos + np.array([0, 0, 0.005]), yaw=banana.yaw, grip=GripperAction.CLOSE)
    _reach(emb, [bowl.pos[0], bowl.pos[1], 0.14], yaw=banana.yaw)
    ok, detail = emb.grade()
    assert not ok and detail["still_held"]


def test_tall_bottle_scene_defeats_a_naive_top_down_centre_grasp():
    """A 22 cm bottle grasped at its centre drives the fingertips through the body; only a
    grasp near the top is reachable."""
    emb = tall_bottle_and_crate()
    bottle = emb.scene.body("bottle")
    at_centre = top_down_pose(bottle.pos)
    holds, _ = grasp_check(at_centre, 1.0, bottle, emb.gripper_model)
    assert holds  # the naive pose does close on it in the mock...
    near_top = top_down_pose(np.array([bottle.pos[0], bottle.pos[1], bottle.top_z - 0.02]))
    holds_top, _ = grasp_check(near_top, 1.0, bottle, emb.gripper_model)
    assert holds_top
    # ...but reaching the centre pose means driving the fingertips through the body
    rep = _reach(emb, bottle.pos)
    assert rep.outcome is ExecutionOutcome.BLOCKED


def test_cluttered_scene_has_neighbours_within_flood_fill_distance():
    emb = cluttered_bin()
    apple, lime = emb.scene.body("apple"), emb.scene.body("lime")
    assert np.linalg.norm(apple.pos[:2] - lime.pos[:2]) < 0.10


def test_wrist_camera_is_optional():
    emb = MockEmbodiment(
        bodies=cluttered_bin().scene.bodies,
        instruction="x",
        goal=("apple", "bin"),
        config=MockConfig(with_wrist=False),
    )
    obs = emb.observe()
    assert obs.wrist is None
    assert "wrist" not in {c.name for c in emb.capabilities.cameras}


def test_rotation_applies_to_the_held_body():
    emb = banana_in_bowl()
    banana = emb.scene.body("banana")
    _reach(emb, banana.pos + np.array([0, 0, 0.005]), yaw=banana.yaw, grip=GripperAction.CLOSE)
    yaw_before = emb.scene.body("banana").yaw
    here = emb.observe().tcp.pos
    emb.execute(
        TCPTarget(
            pose=top_down_pose(here, banana.yaw + 0.6),
            max_lin_speed_mps=0.3,
            timeout_s=8.0,
        )
    )
    assert emb.scene.body("banana").yaw == pytest.approx(yaw_before + 0.6, abs=1e-6)


def test_gripper_actuation_shares_the_target_deadline():
    from rung.contracts.action import ExecutionOutcome, GripperAction, GripperCommand, TCPTarget
    from rung.mock import SCENES

    emb = SCENES["banana_in_bowl"]()
    report = emb.execute(
        TCPTarget(emb.observe().tcp, GripperCommand(GripperAction.CLOSE), timeout_s=0.05)
    )
    assert report.outcome is ExecutionOutcome.EXPIRED
    assert report.duration_s <= 0.05 + 1e-9
    assert emb.observe().gripper.travel == 0
