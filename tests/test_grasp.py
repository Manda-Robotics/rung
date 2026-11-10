"""The grasp planner and its checks.

Most tests cover rejections: an infeasible candidate is rejected with a stated reason
before execution rather than offered and failed on the arm.
"""

from __future__ import annotations

import numpy as np
import pytest

from rung.contracts.frames import Pose, quat_to_rot
from rung.grasp import AnalyticGraspPlanner, obstacles_for
from rung.grasp.analytic import PlannerConfig, top_down_pose
from rung.grasp.checks import (
    check_finger_collision,
    check_opening,
    check_reach,
    check_table,
    finger_volumes,
)
from rung.grasp.types import closing_yaw
from rung.mock import SCENES
from rung.mock.world import grasp_check
from rung.perception import OracleGrounding, PerceptionService


def setup(scene: str):
    emb = SCENES[scene]()
    cat = PerceptionService(OracleGrounding(emb)).build(emb.observe(), emb.instruction)
    planner = AnalyticGraspPlanner(emb.capabilities)
    return emb, cat, planner


def plan_for(emb, cat, planner, label: str):
    obj = next(o for o in cat.objects if o.label == label)
    table_z = emb.capabilities.workspace.table_z
    return obj, planner.plan(obj, obstacles_for(obj.object_id, cat, table_z))


def test_graspable_objects_get_feasible_candidates():
    for scene, label in [
        ("banana_in_bowl", "banana"),
        ("cluttered_bin", "apple"),
        ("tall_bottle_and_crate", "bottle"),
    ]:
        emb, cat, planner = setup(scene)
        _, plan = plan_for(emb, cat, planner, label)
        assert plan.feasible, (scene, label, plan.render())


def test_containers_are_rejected_as_too_wide_with_a_readable_reason():
    for scene, label in [
        ("banana_in_bowl", "bowl"),
        ("cluttered_bin", "bin"),
        ("tall_bottle_and_crate", "crate"),
    ]:
        emb, cat, planner = setup(scene)
        _, plan = plan_for(emb, cat, planner, label)
        assert not plan.feasible
        assert any("too wide for the jaw" in (c.rejected or "") for c in plan.candidates)
        assert "cm" in plan.render()


def test_every_feasible_candidate_measures_the_true_width():
    """A candidate's stated width is what the jaw will actually have to span."""
    for scene in SCENES:
        emb, cat, planner = setup(scene)
        for obj in cat.objects:
            plan = planner.plan(obj, obstacles_for(obj.object_id, cat))
            truth = emb.scene.body(obj.label).footprint[1]
            for cand in plan.feasible:
                assert cand.width_m == pytest.approx(truth, abs=0.003), (scene, cand.candidate_id)


def test_a_feasible_candidate_actually_grasps_in_the_world():
    """The planner's promise, checked against the embodiment's own grasp rule."""
    for scene, label in [
        ("banana_in_bowl", "banana"),
        ("cluttered_bin", "apple"),
        ("tall_bottle_and_crate", "bottle"),
    ]:
        emb, cat, planner = setup(scene)
        _, plan = plan_for(emb, cat, planner, label)
        best = plan.best()
        holds, why = grasp_check(best.pose, 1.0, emb.scene.body(label), emb.gripper_model)
        assert holds, (scene, label, best.describe(), why)


def test_mid_body_grasps_are_rejected_as_unmeasurable_from_one_camera():
    """A band that does not reach the top face contains only the visible side walls, so
    its width is an underestimate. The planner must say so rather than act on it."""
    emb, cat, planner = setup("tall_bottle_and_crate")
    _, plan = plan_for(emb, cat, planner, "bottle")
    low = [c for c in plan.candidates if c.height_m < 0.15]
    assert low, "expected the planner to try low grasps on a 22 cm bottle"
    assert all("cannot measure the width" in (c.rejected or "") for c in low)


def test_clearance_is_reported_and_small_in_clutter():
    emb, cat, planner = setup("cluttered_bin")
    _, apple = plan_for(emb, cat, planner, "apple")
    _, lime = plan_for(emb, cat, planner, "lime")
    assert np.isfinite(apple.best().clearance_m)
    assert lime.best().clearance_m < 0.05


def test_an_obstacle_in_the_finger_volume_rejects_the_candidate():
    """Points where a finger has to go must reject that grasp angle. The planner may turn
    the jaw to another angle that clears them, and must then say so; it may never offer a
    grasp whose hand still contains them."""
    from rung.grasp.checks import check_finger_collision

    emb, cat, planner = setup("cluttered_bin")
    obj = next(o for o in cat.objects if o.label == "apple")
    clean = planner.plan(obj, np.zeros((0, 3)))
    assert clean.feasible
    blocked_points = np.concatenate(
        [
            np.asarray(centre)[None]
            for c in clean.feasible
            for centre, _, _ in finger_volumes(c.pose, c.opening_m, emb.capabilities)
        ]
    )
    plan = planner.plan(obj, blocked_points)
    for c in plan.candidates:
        if c.feasible:
            assert c.approach == "side" or any("turned" in n for n in c.notes), c.describe()
            assert (
                check_finger_collision(c.pose, c.opening_m, emb.capabilities, blocked_points)[0]
                is None
            )
        elif c.approach == "top":
            assert "collide" in c.rejected or "blocked" in c.rejected, c.rejected


def test_a_wide_object_gets_a_side_candidate_rejected_for_its_width():
    """A bowl too wide for the jaw from every direction is rejected with its width in
    centimetres, not reported as an unimplemented approach."""
    emb, cat, planner = setup("banana_in_bowl")
    _, plan = plan_for(emb, cat, planner, "bowl")
    side = [c for c in plan.candidates if c.approach == "side"]
    assert side and "too wide" in side[0].rejected and "not implemented" not in side[0].rejected


def test_side_pose_lays_the_approach_flat_and_stays_right_handed():
    from rung.contracts.frames import quat_to_rot
    from rung.grasp.analytic import side_pose

    pose = side_pose([0.4, 0.0, 0.1], approach=[1.0, 0.0, 0.0], closing=[0.0, 1.0, 0.0])
    rot = quat_to_rot(pose.quat)
    assert np.allclose(rot[:, 2], [1.0, 0.0, 0.0])
    assert np.allclose(rot[:, 1], [0.0, 1.0, 0.0])
    assert np.allclose(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2])
    assert np.linalg.det(rot) == pytest.approx(1.0)


def test_a_tall_bottle_can_be_taken_from_the_side():
    """A horizontal approach executes end to end on the mock: standoff, approach, close and
    lift, with the bottle held."""
    from rung.contracts.action import ExecutionOutcome, GripperAction, GripperCommand, TCPTarget
    from rung.perception.cloud import view_direction_xy

    emb, cat, planner = setup("tall_bottle_and_crate")
    obj, _ = plan_for(emb, cat, planner, "bottle")
    table_z = emb.capabilities.workspace.table_z
    # The executor always tells the planner where the camera is. Without it the square
    # bottle's mid-body slab, two visible faces from one view, measures 4.7 cm across
    # its diagonal and the jaw would be aimed where the bottle is 9.9 cm wide.
    view = view_direction_xy(emb.observe().scene.pose.pos, obj.grasp_centre)
    cand = planner._side_candidate(
        obj, obstacles_for(obj.object_id, cat, table_z), 0, view_dir=view
    )
    assert cand is not None and cand.feasible, cand and cand.rejected
    assert cand.pose.pos[2] > table_z + 0.02
    approach = np.array(cand.pose.pos) - np.array(cand.pre_pose.pos)
    assert abs(approach[2]) < 1e-6, "a side approach is horizontal"
    assert np.linalg.norm(cand.pre_pose.pos[:2]) < np.linalg.norm(cand.pose.pos[:2]), (
        "the standoff is on the robot's side of the object"
    )

    def go(pose, action):
        return emb.execute(
            TCPTarget(
                pose=pose,
                gripper=GripperCommand(action),
                max_lin_speed_mps=0.2,
                max_ang_speed_rps=1.0,
                timeout_s=20.0,
            )
        )

    assert go(cand.pre_pose, GripperAction.OPEN).outcome is ExecutionOutcome.REACHED
    r = go(cand.pose, GripperAction.HOLD)
    assert r.outcome is ExecutionOutcome.REACHED, r.reason
    go(emb.observe().tcp, GripperAction.CLOSE)
    assert emb.robot.held == "bottle", emb.events[-3:]
    lifted = Pose(cand.pose.pos + np.array([0.0, 0.0, 0.10]), cand.pose.quat)
    assert go(lifted, GripperAction.HOLD).outcome is ExecutionOutcome.REACHED
    assert emb.scene.body("bottle").bottom_z > table_z + 0.05


def test_reach_and_table_checks():
    emb = SCENES["banana_in_bowl"]()
    caps = emb.capabilities
    assert check_reach(top_down_pose([3.0, 0.0, 0.2]), caps) is not None
    assert check_reach(top_down_pose([0.45, 0.0, 0.2]), caps) is None
    assert check_table(top_down_pose([0.45, 0.0, -0.01]), caps) is not None


def test_opening_check_names_the_numbers():
    emb = SCENES["banana_in_bowl"]()
    reason = check_opening(0.20, emb.capabilities)
    assert "20.0 cm" in reason and "8.5 cm" in reason
    assert check_opening(0.04, emb.capabilities) is None


def test_finger_volumes_straddle_the_tcp():
    """Two fingers on opposite sides of the TCP, extending back above the fingertips."""
    emb = SCENES["banana_in_bowl"]()
    pose = top_down_pose([0.45, 0.0, 0.10])
    vols = finger_volumes(pose, 0.06, emb.capabilities)
    assert len(vols) == 2
    offsets = [float(v[0][1] - pose.pos[1]) for v in vols]
    assert offsets[0] * offsets[1] < 0, offsets
    assert min(abs(o) for o in offsets) > 0.03
    assert all(v[0][2] > pose.pos[2] for v in vols)


def test_finger_collision_detects_points_between_the_jaws():
    emb = SCENES["banana_in_bowl"]()
    pose = top_down_pose([0.45, 0.0, 0.10])
    blocker = np.array([[0.45, 0.045, 0.13]])
    reason, clearance = check_finger_collision(pose, 0.06, emb.capabilities, blocker)
    assert reason is not None and clearance == 0.0
    far = np.array([[0.45, 0.30, 0.13]])
    reason2, clearance2 = check_finger_collision(pose, 0.06, emb.capabilities, far)
    assert reason2 is None and clearance2 > 0.1


def test_closing_yaw_puts_the_jaw_across_the_given_direction():
    for angle in (0.0, 0.4, -1.1, 2.0):
        d = np.array([np.cos(angle), np.sin(angle)])
        pose = top_down_pose([0.45, 0.0, 0.1], closing_yaw(d))
        jaw = pose.rot[:, 1][:2]
        assert abs(abs(float(np.dot(jaw / np.linalg.norm(jaw), d))) - 1.0) < 1e-6


def test_planner_returns_an_empty_plan_for_an_object_without_points():
    emb, cat, planner = setup("banana_in_bowl")
    obj = cat.objects[0]
    obj.points = None
    assert planner.plan(obj).candidates == []


def test_scoring_prefers_more_clearance_all_else_equal():
    """The ranking has to be explainable, because the controlled track compares a model's
    choice of candidate against it."""
    emb, cat, planner = setup("cluttered_bin")
    obj = next(o for o in cat.objects if o.label == "apple")
    roomy = planner.plan(obj, np.zeros((0, 3))).best()
    # a point just outside the finger, close enough to matter but not to collide
    centre, half, rot = finger_volumes(roomy.pose, roomy.opening_m, emb.capabilities)[1]
    near = np.asarray(centre) + rot @ np.array([0.0, float(half[1]) + 0.012, 0.0])
    tight = planner.plan(obj, near[None]).best()
    assert tight is not None and tight.clearance_m < 0.05
    assert roomy.clearance_m > tight.clearance_m
    assert roomy.score > tight.score


def test_single_height_config_still_works():
    emb, cat, planner = setup("banana_in_bowl")
    planner = AnalyticGraspPlanner(emb.capabilities, PlannerConfig(heights=1))
    _, plan = plan_for(emb, cat, planner, "banana")
    assert len([c for c in plan.candidates if c.approach == "top"]) == 1 and plan.feasible


def test_the_gripper_body_hits_a_tall_rim_the_fingers_clear():
    """The fingers are 5.5 cm; the body above them is wider. A pail rim 20 cm up beside a
    short bottle is clear of the fingers and not of the hand."""
    from rung.grasp.checks import check_finger_collision

    emb = SCENES["banana_in_bowl"]()
    caps = emb.capabilities
    pose = top_down_pose([0.49, 0.0, 0.03], yaw=0.0)  # closes along y
    closing = quat_to_rot(pose.quat)[:, 1]
    # A wall of points 6.5 cm to one side along the closing axis, from 15 to 23 cm up:
    # beyond the open finger (5.35 cm + pad) but inside the wider body above it.
    wall = np.array(
        [
            [0.49 + dx, 0.0, z]
            for dx in np.linspace(-0.02, 0.02, 5)
            for z in np.linspace(0.15, 0.23, 9)
        ]
    ) + np.append(closing[:2] * 0.065, 0)
    reason, _ = check_finger_collision(pose, caps.gripper.max_opening_m, caps, wall)
    assert reason is not None and "gripper body" in reason, reason


def test_a_colliding_jaw_angle_is_turned_away_from_the_neighbour():
    from rung.grasp.checks import check_finger_collision

    emb, cat, planner = setup("cluttered_bin")
    obj = next(o for o in cat.objects if o.label == "apple")
    first = planner.plan(obj, np.zeros((0, 3))).best()
    closing = quat_to_rot(first.pose.quat)[:, 1]
    centre = first.pose.pos
    wall = np.array(
        [
            centre + closing * 0.066 + np.array([0, 0, z]) + perp
            for z in np.linspace(0.0, 0.2, 11)
            for perp in (np.cross([0, 0, 1], closing) * s for s in (-0.01, 0, 0.01))
        ]
    )
    plan = planner.plan(obj, wall)
    turned = [c for c in plan.feasible if c.approach == "top"]
    assert turned, plan.render()
    for c in turned:
        assert check_finger_collision(c.pose, c.opening_m, emb.capabilities, wall)[0] is None


def test_an_open_mug_wider_than_the_jaw_gets_a_rim_pinch():
    """A mug wider than the jaw is still graspable by a rim pinch on its wall, one finger
    inside and one outside."""
    from rung.perception.types import ObjectDescriptor

    emb = SCENES["banana_in_bowl"]()
    planner = AnalyticGraspPlanner(emb.capabilities)
    rng = np.random.default_rng(7)
    c = np.array([0.50, 0.10])
    theta = rng.uniform(0, 2 * np.pi, 4000)
    r = rng.uniform(0.042, 0.047, 4000)  # 9.4 cm across the outside, 5 mm wall
    z = rng.uniform(0.0, 0.095, 4000)
    wall = np.column_stack([c[0] + r * np.cos(theta), c[1] + r * np.sin(theta), z])
    floor = np.column_stack([c + rng.uniform(-0.03, 0.03, (500, 2)), np.full(500, 0.01)])
    pts = np.vstack([wall, floor])
    mug = ObjectDescriptor(
        object_id="mug_0",
        label="mug",
        centroid=np.array([c[0], c[1], 0.05]),
        top_z=0.095,
        bottom_z=0.0,
        length=0.094,
        width=0.094,
        height=0.095,
        axis=np.array([1.0, 0.0]),
        grasp_length=0.094,
        grasp_width=0.094,
        grasp_centre=np.array([c[0], c[1], 0.095]),
        n_points=len(pts),
        visible_fraction=1.0,
        points=pts,
    )
    plan = planner.plan(mug, np.zeros((0, 3)))
    rim = [x for x in plan.candidates if x.approach == "rim"]
    assert rim and rim[0].feasible, plan.render()
    cand = rim[0]
    assert cand.width_m < 0.02
    to_base = -c / np.linalg.norm(c)
    assert np.linalg.norm(cand.pose.pos[:2] - (c + to_base * 0.0445)) < 0.01, "on the near wall"
    assert abs(float(quat_to_rot(cand.pose.quat)[:2, 1] @ to_base)) > 0.95, "closes across it"
    assert cand.pose.pos[2] == pytest.approx(0.075, abs=0.003)


def test_a_solid_object_wider_than_the_jaw_gets_no_rim_pinch():
    from rung.perception.types import ObjectDescriptor

    emb = SCENES["banana_in_bowl"]()
    planner = AnalyticGraspPlanner(emb.capabilities)
    rng = np.random.default_rng(8)
    pts = np.column_stack(
        [rng.uniform(0.45, 0.55, 4000), rng.uniform(0.05, 0.15, 4000), np.full(4000, 0.09)]
    )
    box = ObjectDescriptor(
        object_id="box_0",
        label="box",
        centroid=np.array([0.5, 0.1, 0.045]),
        top_z=0.09,
        bottom_z=0.0,
        length=0.1,
        width=0.1,
        height=0.09,
        axis=np.array([1.0, 0.0]),
        grasp_length=0.1,
        grasp_width=0.1,
        grasp_centre=np.array([0.5, 0.1, 0.09]),
        n_points=4000,
        visible_fraction=1.0,
        points=pts,
    )
    assert not [x for x in planner.plan(box, np.zeros((0, 3))).candidates if x.approach == "rim"]


def test_a_mug_handle_facing_the_robot_does_not_move_the_rim_pinch_onto_it():
    """A handle facing the robot does not inflate the rim estimate: the rim pinch aims at
    the wall, away from the handle."""
    from rung.perception.types import ObjectDescriptor

    emb = SCENES["banana_in_bowl"]()
    planner = AnalyticGraspPlanner(emb.capabilities)
    rng = np.random.default_rng(11)
    c = np.array([0.40, 0.30])
    to_base = -c / np.linalg.norm(c)
    theta = rng.uniform(0, 2 * np.pi, 5000)
    r = rng.uniform(0.041, 0.045, 5000)
    wall = np.column_stack(
        [c[0] + r * np.cos(theta), c[1] + r * np.sin(theta), rng.uniform(0.0, 0.09, 5000)]
    )
    along = rng.uniform(0.045, 0.07, 600)  # a handle toward the robot
    handle = np.column_stack(
        [
            c[0] + to_base[0] * along + rng.uniform(-0.006, 0.006, 600),
            c[1] + to_base[1] * along + rng.uniform(-0.006, 0.006, 600),
            rng.uniform(0.02, 0.09, 600),
        ]
    )
    floor = np.column_stack([c + rng.uniform(-0.03, 0.03, (500, 2)), np.full(500, 0.008)])
    pts = np.vstack([wall, handle, floor])
    mug = ObjectDescriptor(
        object_id="mug_0",
        label="mug",
        centroid=np.array([c[0], c[1], 0.045]),
        top_z=0.09,
        bottom_z=0.0,
        length=0.115,
        width=0.09,
        height=0.09,
        axis=np.array([1.0, 0.0]),
        grasp_length=0.115,
        grasp_width=0.09,
        grasp_centre=np.array([c[0], c[1], 0.09]),
        n_points=len(pts),
        visible_fraction=1.0,
        points=pts,
    )
    rim = [x for x in planner.plan(mug, np.zeros((0, 3))).candidates if x.approach == "rim"]
    assert rim and rim[0].feasible, rim and rim[0].rejected
    aim = rim[0].pose.pos[:2]
    assert abs(np.linalg.norm(aim - c) - 0.043) < 0.004, np.linalg.norm(aim - c)  # on the wall
    side = (aim - c) / np.linalg.norm(aim - c)
    assert float(side @ to_base) < np.cos(np.radians(35)), "must not pinch at the handle"
