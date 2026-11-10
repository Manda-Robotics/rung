"""The L2 primitives end to end against the mock.

The central property: a grasp that did not take the object never reports ``grasped``. The
rest of the harness trusts that result.
"""

from __future__ import annotations

import numpy as np
import pytest

from rung.contracts.action import GripperAction, GripperCommand, TCPTarget
from rung.executor import Executor, Outcome
from rung.grasp import AnalyticGraspPlanner
from rung.grasp.analytic import top_down_pose
from rung.mock import SCENES
from rung.perception import ObjectMemory, OracleGrounding, PerceptionService


def make(scene: str = "banana_in_bowl"):
    emb = SCENES[scene]()
    ex = Executor(
        emb,
        PerceptionService(OracleGrounding(emb)),
        AnalyticGraspPlanner(emb.capabilities),
        ObjectMemory(),
        emb.instruction,
    )
    ex.refresh()
    return emb, ex


def goal_ids(emb, ex):
    obj_label, target_label = emb.goal
    cat = ex.catalog_or_refresh()
    return (
        next(o.object_id for o in cat.objects if o.label == obj_label),
        next(o.object_id for o in cat.objects if o.label == target_label),
    )


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_grasp_then_place_solves_every_development_scene(scene):
    """Given the right object choices, the executor completes the task. Model policies are
    evaluated on top of this baseline."""
    emb, ex = make(scene)
    obj_id, target_id = goal_ids(emb, ex)

    grasped = ex.grasp(obj_id)
    assert grasped.outcome is Outcome.GRASPED, grasped.render()
    assert ex.memory.held == obj_id

    placed = ex.place(target_id)
    assert placed.outcome is Outcome.PLACED, placed.render()
    assert ex.memory.held is None

    success, detail = emb.grade()
    assert success, detail


def test_grasp_verification_reports_the_measured_drop():
    emb, ex = make("tall_bottle_and_crate")
    obj_id, _ = goal_ids(emb, ex)
    result = ex.grasp(obj_id)
    assert result.outcome is Outcome.GRASPED
    assert result.detail["dropped_m"] == pytest.approx(0.22, abs=0.02)
    assert "came with the hand" in result.reason


def test_a_grasp_that_does_not_take_the_object_is_not_reported_as_grasped():
    """A close that does not capture the object never reports ``grasped``.

    The gripper closes on nothing while the object stays put, which finger travel alone
    cannot distinguish from an obstructed close.
    """
    emb, ex = make("banana_in_bowl")
    obj_id, _ = goal_ids(emb, ex)
    emb.scene.body("banana").graspable = False  # the fingers pass around it and meet

    result = ex.grasp(obj_id)
    assert result.outcome is Outcome.MISSED
    assert ex.memory.held is None
    assert "nothing between them" in result.reason
    assert emb.grade()[0] is False


def test_closure_obstructed_when_the_fingers_stop_but_nothing_moves():
    """Travel says "holding", the world says otherwise. The primitive must believe the world."""
    emb, ex = make("banana_in_bowl")
    obj_id, _ = goal_ids(emb, ex)

    real_close = emb._close

    def close_but_do_not_attach(target):
        real_close(target)
        emb.robot.held = None  # jaws stopped on it, but it never came along
        emb.robot.held_offset = None
        emb.robot.travel = 0.5

    emb._close = close_but_do_not_attach
    result = ex.grasp(obj_id)
    assert result.outcome is Outcome.CLOSURE_OBSTRUCTED
    assert ex.memory.held is None
    assert "stayed where it was" in result.reason


def test_grasping_a_container_is_rejected_with_the_measurement():
    emb, ex = make("cluttered_bin")
    _, bin_id = goal_ids(emb, ex)
    result = ex.grasp(bin_id)
    assert result.outcome is Outcome.REJECTED
    assert "too wide for the jaw" in result.reason


def test_grasping_an_unknown_object_lists_what_is_there():
    emb, ex = make("banana_in_bowl")
    result = ex.grasp("pineapple_9")
    assert result.outcome is Outcome.REJECTED
    assert "no object called pineapple_9" in result.reason
    assert "banana_0" in result.reason


def test_grasping_while_already_holding_is_rejected():
    emb, ex = make("cluttered_bin")
    apple, _ = goal_ids(emb, ex)
    assert ex.grasp(apple).outcome is Outcome.GRASPED
    second = ex.grasp("lime_0")
    assert second.outcome is Outcome.REJECTED
    assert "already holding apple_0" in second.reason


def test_placing_with_an_empty_hand_is_rejected():
    emb, ex = make("banana_in_bowl")
    _, bowl = goal_ids(emb, ex)
    result = ex.place(bowl)
    assert result.outcome is Outcome.NOT_HOLDING
    assert "grasp something first" in result.reason


def test_place_releases_above_the_rim_not_at_the_centroid():
    """Releasing at a container's centroid height drives the held object into its floor."""
    emb, ex = make("cluttered_bin")
    apple, bin_id = goal_ids(emb, ex)
    ex.grasp(apple)
    result = ex.place(bin_id)
    assert result.outcome is Outcome.PLACED
    assert result.detail["release_z"] > emb.scene.body("bin").top_z


def test_inspect_reports_size_and_grasps_without_moving():
    emb, ex = make("tall_bottle_and_crate")
    before = emb.observe().tcp.pos.copy()
    result = ex.inspect("bottle_0")
    assert result.outcome is Outcome.OBSERVED
    assert "cm" in result.detail["description"]
    assert "top grasp" in result.detail["grasps"]
    assert np.allclose(emb.observe().tcp.pos, before)


def test_propose_grasps_lists_candidate_ids():
    emb, ex = make("banana_in_bowl")
    result = ex.propose_grasps("banana_0")
    assert result.detail["candidates"]
    assert all(c.startswith("banana_0#") for c in result.detail["candidates"])


def test_move_to_reports_a_short_move_without_calling_it_a_failure():
    emb, ex = make("banana_in_bowl")
    result = ex.move_to(top_down_pose([0.50, 0.10, 0.28]))
    assert result.outcome is Outcome.MOVED
    assert result.detail["residual_m"] < 0.01


def test_memory_records_every_primitive_attempt():
    emb, ex = make("banana_in_bowl")
    obj_id, target_id = goal_ids(emb, ex)
    ex.grasp(obj_id)
    ex.place(target_id)
    rendered = ex.memory.render()
    assert "grasp(banana_0) -> grasped" in rendered
    assert "place(banana_0) -> placed" in rendered


def test_release_drops_what_is_held_and_clears_memory():
    emb, ex = make("cluttered_bin")
    apple, _ = goal_ids(emb, ex)
    ex.grasp(apple)
    assert ex.memory.held == apple
    result = ex.release()
    assert ex.memory.held is None
    assert result.object_id == apple
    assert emb.robot.held is None


def test_the_executor_never_reads_the_grader():
    """A primitive result must be derivable from observations alone.

    Guarded structurally: the executor holds an embodiment, and grading is a separate
    protocol. If a primitive ever called it, this test is where that would be caught.
    """
    import inspect as _inspect

    from rung.executor import primitives

    source = _inspect.getsource(primitives)
    assert "grade(" not in source
    assert ".goal" not in source


def test_a_placed_object_keeps_its_identity():
    """An object the harness placed keeps its id.

    A placed object can drop out of sight behind the container wall. Memory then keeps the
    id and the object's new location, and no other detection is issued that name.
    """
    emb, ex = make("banana_in_bowl")
    obj_id, target_id = goal_ids(emb, ex)
    ex.grasp(obj_id)
    ex.place(target_id)
    catalog = ex.refresh()
    bananas = [i for i in catalog.ids() if i.startswith("banana")]
    assert bananas in ([], [obj_id]), catalog.ids()
    assert obj_id in ex.memory.tracked
    assert ex.memory.held is None
    assert ex.memory.tracked[obj_id].centroid[2] > emb.capabilities.workspace.table_z + 0.02


def test_a_container_keeps_its_identity_when_something_lands_in_it():
    """A crate keeps its id when its visible centroid shifts because an object is dropped
    into it; the match distance scales with object size."""
    emb, ex = make("tall_bottle_and_crate")
    bottle, crate = goal_ids(emb, ex)
    ex.grasp(bottle)
    ex.place(crate)
    ids = ex.refresh().ids()
    assert crate in ids, ids
    assert len([i for i in ids if i.startswith("crate_")]) == 1, ids


def test_missing_depth_at_old_location_is_unknown(monkeypatch):
    emb, ex = make()
    import rung.executor.primitives as primitives

    monkeypatch.setattr(
        primitives, "depth_image_to_points", lambda *a, **k: (np.array([[0.7, 0.4, 0.0]]), None)
    )
    assert ex._surface_dropped(np.array([0.4, 0]), 0.1) is None


def test_a_held_object_the_camera_cannot_see_is_still_a_grasp(monkeypatch):
    """Occlusion by the hand does not veto a grasp.

    From the scene camera a held object is behind the hand, so the surface match fails
    when the grasp worked. The surface matches grade confidence; the discriminator is the
    jaw, which cannot be occluded.
    """
    _, ex = make()
    monkeypatch.setattr(ex, "_attachment_match", lambda *a: False)
    result = ex.grasp("banana_0")
    assert result.outcome is Outcome.GRASPED, result.reason
    assert "inferred" in result.detail["confidence"]
    assert ex.memory.held == "banana_0"


def test_an_object_knocked_out_of_its_place_is_not_a_grasp(monkeypatch):
    """An object swept aside is not reported as held.

    If the fingers push the object off its place, the jaw meets nothing afterwards and runs
    shut; a jaw with something in it cannot.
    """
    _, ex = make()
    monkeypatch.setattr(ex, "_attachment_match", lambda *a: False)
    real_observe = ex.emb.observe
    state = {"closed_after_lift": False}

    def observe():
        obs = real_observe()
        if state["closed_after_lift"]:
            from dataclasses import replace

            return replace(obs, gripper=replace(obs.gripper, travel=1.0))
        return obs

    monkeypatch.setattr(ex.emb, "observe", observe)
    original_go = ex._go

    def go(pose, gripper, meta=None, **kw):
        report = original_go(pose, gripper, meta, **kw)
        if (meta or {}).get("phase") == "step_aside":
            state["closed_after_lift"] = True
        return report

    monkeypatch.setattr(ex, "_go", go)
    result = ex.grasp("banana_0")
    assert result.outcome is Outcome.MISSED, result.reason
    assert "knocked aside" in result.reason
    assert ex.memory.held is None
    assert not ex.unverified_hold


def test_incomplete_descent_does_not_close_the_gripper(monkeypatch):
    """An approach that never gets there is retried, then reported, and the fingers
    never close: closing on nothing wastes the model's turn and, worse, closing at the
    wrong height can knock the object away."""
    from dataclasses import replace

    from rung.contracts.action import ExecutionOutcome

    emb, ex = make()
    execute = emb.execute
    phases = []

    def short(target):
        phases.append(target.meta["phase"])
        result = execute(target)
        return replace(
            result,
            outcome=ExecutionOutcome.EXPIRED,
            residual_m=0.05,
            residual_axis="z",
            executed_lin_m=0.05,
        )

    monkeypatch.setattr(emb, "execute", short)
    result = ex.grasp("banana_0")
    assert result.outcome is Outcome.BLOCKED
    assert "approach" in result.reason
    assert phases == ["pre_grasp"] * 3, phases
    assert emb.observe().gripper.travel < 0.5


def test_attachment_check_distinguishes_a_real_hold_from_air():
    """The attachment check, tested directly rather than stubbed.

    It must say yes when the object is in the hand and no when the fingers closed on
    nothing, and it must not be fooled by the table underneath.
    """
    emb, ex = make("cluttered_bin")
    apple, _ = goal_ids(emb, ex)
    obj = ex.object(apple)
    pose = ex.plan_for(apple).best().pose

    # nothing in the hand: the gripper reads fully closed
    emb.execute(
        TCPTarget(
            pose=top_down_pose([0.30, 0.32, 0.10]),
            gripper=GripperCommand(GripperAction.CLOSE),
            max_lin_speed_mps=0.3,
            timeout_s=10.0,
        )
    )
    assert ex._attachment_match(obj, pose) is False

    # a real hold, lifted clear of the table
    emb, ex = make("cluttered_bin")
    apple, _ = goal_ids(emb, ex)
    obj = ex.object(apple)
    pose = ex.plan_for(apple).best().pose
    assert ex.grasp(apple).outcome is Outcome.GRASPED
    assert ex._attachment_match(ex.object(apple) or obj, pose) is not False


def test_attachment_check_is_not_fooled_by_the_table():
    """A short object lifted 5 cm still predicts surfaces close to the table. If table
    returns counted as evidence, an empty gripper would verify as a hold."""
    emb, ex = make("banana_in_bowl")
    banana, _ = goal_ids(emb, ex)
    obj = ex.object(banana)
    pose = ex.plan_for(banana).best().pose

    # hover over bare table with the jaws part-closed, holding nothing
    emb.execute(
        TCPTarget(
            pose=top_down_pose([0.30, 0.30, 0.06]),
            gripper=GripperCommand(GripperAction.CLOSE),
            max_lin_speed_mps=0.3,
            timeout_s=10.0,
        )
    )
    emb.robot.travel = 0.5  # reads as "probably holding" from travel alone
    assert emb.robot.held is None
    assert ex._attachment_match(obj, pose) in (False, None)


def test_release_height_follows_what_is_being_held():
    """A tall held object is released higher than a short one over the same container.

    The release height puts the object's underside above the rim, using the overhang below
    the fingertips measured at the grasp.
    """
    _, ex = make()
    target = next(o for o in ex.catalog_or_refresh().objects if "bowl" in o.object_id)

    ex.held_overhang_m = None
    baseline, _ = ex._release_pose(target, "in")
    ex.held_overhang_m = 0.02
    shallow, _ = ex._release_pose(target, "in")
    ex.held_overhang_m = 0.08
    deep, _ = ex._release_pose(target, "in")

    assert deep > shallow, "a deeper hold has to be released higher"
    assert deep - shallow == pytest.approx(0.06, abs=1e-9), "one to one with the overhang"
    assert baseline > 0


def test_a_converging_motion_is_reissued_rather_than_abandoned():
    """An expired motion that is still making progress is re-issued, not reported blocked.

    The approach lives inside one ``grasp`` call, so the model cannot ask for it to
    continue; the executor has to.
    """
    from dataclasses import replace

    from rung.contracts.action import ExecutionOutcome
    from rung.contracts.frames import Pose

    _, ex = make()
    calls = []
    real_issue = ex._issue

    def issue(target, meta):
        report = real_issue(target, meta)
        calls.append(target.id)
        # The first two attempts expire while still travelling; the third arrives.
        if len(calls) < 3:
            return replace(report, outcome=ExecutionOutcome.EXPIRED, executed_lin_m=0.05)
        return report

    ex._issue = issue
    ex._go(
        Pose(np.array([0.45, 0.0, 0.25]), ex.emb.observe().tcp.quat),
        GripperCommand(GripperAction.HOLD),
    )
    assert len(calls) == 3, f"expected two retries then arrival, got {len(calls)} attempts"


def test_a_stuck_motion_is_not_reissued():
    """No progress means blocked, and retrying only spends the episode's clock."""
    from dataclasses import replace

    from rung.contracts.action import ExecutionOutcome
    from rung.contracts.frames import Pose
    from rung.executor.primitives import MotionIncomplete

    _, ex = make()
    calls = []
    real_issue = ex._issue

    def issue(target, meta):
        report = real_issue(target, meta)
        calls.append(target.id)
        return replace(report, outcome=ExecutionOutcome.EXPIRED, executed_lin_m=0.0001)

    ex._issue = issue
    with pytest.raises(MotionIncomplete):
        ex._go(
            Pose(np.array([0.45, 0.0, 0.25]), ex.emb.observe().tcp.quat),
            GripperCommand(GripperAction.HOLD),
        )
    assert len(calls) == 1, "a motion making no progress must not be retried"


# ---- motions under the episode loop's handler ---------------------------------------


def test_retries_also_happen_through_the_loops_motion_handler():
    """Under the episode loop every motion goes through the loop's handler, and motion
    retries apply on that path as well."""
    from dataclasses import replace

    from rung.contracts.action import ExecutionOutcome
    from rung.contracts.frames import Pose

    emb, ex = make()
    calls = []

    def handler(target, *, servo=False):
        report = emb.execute(target)
        calls.append(target.id)
        if len(calls) < 3:
            return replace(
                report, outcome=ExecutionOutcome.EXPIRED, executed_lin_m=0.05, residual_m=0.03
            )
        return report

    ex.motion_handler = handler
    ex._go(
        Pose(np.array([0.45, 0.0, 0.25]), ex.emb.observe().tcp.quat),
        GripperCommand(GripperAction.HOLD),
    )
    assert len(calls) == 3


def test_a_waypoint_that_stopped_a_little_short_is_close_enough():
    from dataclasses import replace

    from rung.contracts.action import ExecutionOutcome
    from rung.contracts.frames import Pose
    from rung.executor.primitives import MotionIncomplete

    emb, ex = make()

    def handler(target, *, servo=False):
        report = emb.execute(target)
        return replace(
            report, outcome=ExecutionOutcome.EXPIRED, executed_lin_m=0.0, residual_m=0.012
        )

    ex.motion_handler = handler
    pose = Pose(np.array([0.45, 0.0, 0.25]), ex.emb.observe().tcp.quat)
    ex._go(pose, GripperCommand(GripperAction.HOLD), tolerance_m=0.02)
    with pytest.raises(MotionIncomplete):
        ex._go(pose, GripperCommand(GripperAction.HOLD), tolerance_m=0.01)


def test_a_blocked_descent_backs_off_and_says_the_fingers_hit_something(monkeypatch):
    """When the fingertips stop above the grasp height they are resting on the object
    or a neighbour. The hand goes back up so the next look is clean, and the report
    says what happened rather than 'blocked'."""
    from dataclasses import replace

    from rung.contracts.action import ExecutionOutcome

    emb, ex = make()
    execute = emb.execute
    phases = []

    def blocked_descent(target):
        phases.append(target.meta["phase"])
        if target.meta["phase"] == "descend":
            short = replace(
                target, pose=replace(target.pose, pos=target.pose.pos + np.array([0, 0, 0.03]))
            )
            report = execute(short)
            return replace(
                report,
                outcome=ExecutionOutcome.BLOCKED,
                residual_m=0.03,
                residual_axis="z",
                reason="the arm stopped 3.0 cm short on z",
            )
        return execute(target)

    monkeypatch.setattr(emb, "execute", blocked_descent)
    result = ex.grasp("banana_0")
    assert result.outcome is Outcome.BLOCKED
    assert "came down on" in result.reason and "3.0 cm" in result.reason
    assert phases[-1] == "retreat_after_block"
    assert emb.observe().tcp.pos[2] > 0.10, "the hand is back up"
    assert emb.observe().gripper.travel < 0.5, "and still open"
    assert ex.memory.attempts[-1].result == "blocked"


def test_an_empty_grounding_keeps_the_last_view():
    """An empty grounding result mid-episode is a failed call, not an empty scene: the
    previous catalog is kept and its objects stay addressable."""

    emb, ex = make()
    before = ex.refresh()
    assert before.objects

    class Nothing:
        name = "nothing"
        failures = ["the grounding model returned no usable boxes: ''"]

        def detect(self, observation, instruction=""):
            return []

    ex.perception.grounding = Nothing()
    after = ex.refresh()
    assert after.ids() == before.ids()
    assert after.failures and "previous look" in after.failures[0]
    assert all(t.missing_steps == 0 for t in ex.memory.tracked.values())
    assert ex.grasp("banana_0").outcome is Outcome.GRASPED


def test_an_object_that_falls_out_during_the_carry_is_not_reported_placed(monkeypatch):
    """The jaw is read before release, so an object that slips out during the carry is
    reported dropped, not placed."""
    emb, ex = make("banana_in_bowl")
    obj_id, target_id = goal_ids(emb, ex)
    assert ex.grasp(obj_id).outcome is Outcome.GRASPED
    execute = emb.execute

    def slip(target):
        report = execute(target)
        if target.meta.get("phase") == "carry" and emb.robot.held is not None:
            emb.robot.held = None
            emb.robot.held_offset = None
            emb.robot.travel = 1.0
        return report

    monkeypatch.setattr(emb, "execute", slip)
    result = ex.place(target_id)
    assert result.outcome is Outcome.DROPPED, result.render()
    assert "fell out of the hand" in result.reason
    assert ex.memory.held is None
    assert not emb.grade()[0]


def test_the_step_aside_never_goes_outward_or_too_near_the_base():
    """The step aside never goes outward, which puts the forearm over the vacated spot, and
    never ends nearer the base than the arm can reach with the hand pointing down."""
    from rung.executor.primitives import MIN_COMFORTABLE_REACH_M

    _, ex = make("banana_in_bowl")
    for pos in ([0.33, -0.04, 0.08], [0.49, -0.10, 0.08], [0.60, 0.20, 0.08]):
        pos = np.array(pos)
        d = ex._aside_offset(pos, 0.17)
        assert np.linalg.norm(d) > 0, pos
        after = pos + d
        radial = pos[:2] / np.linalg.norm(pos[:2])
        assert float(d[:2] @ radial) <= 1e-9, f"stepped outward from {pos}"
        assert np.linalg.norm(after[:2]) >= MIN_COMFORTABLE_REACH_M


def test_the_robot_over_the_vacated_spot_is_not_the_object(monkeypatch):
    """After the step aside the gripper body and wrist can still reach over the spot the
    object left. Points far above the object's old top cannot be the object and are not
    read as it."""
    from rung.executor import primitives

    _, ex = make("banana_in_bowl")
    xy = np.array([0.60, -0.30])
    rng = np.random.default_rng(6)
    table = np.column_stack([xy + rng.uniform(-0.02, 0.02, (200, 2)), np.zeros(200)])
    arm = np.column_stack([xy + rng.uniform(-0.02, 0.02, (300, 2)), np.full(300, 0.17)])
    monkeypatch.setattr(
        primitives, "depth_image_to_points", lambda *a, **k: (np.vstack([table, arm]), None)
    )
    left = ex._surface_dropped(xy, before_top=0.06, hand_left=True)
    assert left == pytest.approx(0.06, abs=0.005)
    still_over = ex._surface_dropped(xy, before_top=0.06, hand_left=False)
    assert still_over < 0, "straight after the lift the column reading stays inconclusive"


def test_a_lift_that_is_held_down_releases_and_says_so(monkeypatch):
    """A jaw closed on a bowl's rim cannot lift. That is not a hold, probable or otherwise."""
    from dataclasses import replace

    from rung.contracts.action import ExecutionOutcome

    emb, ex = make("banana_in_bowl")
    obj_id, _ = goal_ids(emb, ex)
    execute = emb.execute
    phases = []

    def anchored(target):
        phases.append(target.meta.get("phase"))
        if target.meta.get("phase") == "lift_check":
            report = execute(replace(target, pose=emb.observe().tcp))
            return replace(
                report,
                outcome=ExecutionOutcome.BLOCKED,
                residual_m=0.05,
                residual_axis="z",
                reason="the arm stopped 5.0 cm short on z",
            )
        return execute(target)

    monkeypatch.setattr(emb, "execute", anchored)
    result = ex.grasp(obj_id)
    assert result.outcome is Outcome.CLOSURE_OBSTRUCTED, result.render()
    assert "would not lift" in result.reason
    assert "release_anchored" in phases and "step_aside" not in phases
    assert emb.observe().gripper.travel < 0.5
    assert ex.memory.held is None and not ex.unverified_hold


def test_the_carry_rises_straight_up_and_clears_what_it_passes(monkeypatch):
    """A carry first rises in place above tall neighbours, then moves across, instead of
    travelling diagonally from the lift height."""
    emb, ex = make("banana_in_bowl")
    obj_id, target_id = goal_ids(emb, ex)
    assert ex.grasp(obj_id).outcome is Outcome.GRASPED
    start = emb.observe().tcp.pos.copy()
    target = ex.catalog.get(target_id)
    mid = (start[:2] + target.centroid[:2]) / 2
    tall = np.array(
        [[mid[0] + dx, mid[1] + dy, 0.28] for dx in (-0.02, 0, 0.02) for dy in (-0.02, 0, 0.02)]
    )
    monkeypatch.setattr(ex, "_neighbour_points", lambda exclude=None: tall)
    attempts = []
    execute = emb.execute
    monkeypatch.setattr(
        emb,
        "execute",
        lambda t: (attempts.append((t.meta.get("phase"), t.pose.pos.copy())), execute(t))[1],
    )
    result = ex.place(target_id)
    phases = [a[0] for a in attempts]
    assert phases[:2] == ["rise", "carry"], phases
    rise = attempts[0][1]
    assert np.allclose(rise[:2], start[:2], atol=1e-6), "rises in place first"
    ceiling = emb.capabilities.workspace.max_xyz[2] - 0.02
    assert rise[2] >= min(0.28 + 0.05, ceiling) - 1e-6, "clears the tall neighbour"
    assert result.outcome in (Outcome.PLACED, Outcome.BLOCKED), result.render()


def test_the_step_aside_takes_the_side_away_from_a_neighbour(monkeypatch):
    _, ex = make("banana_in_bowl")
    pos = np.array([0.49, 0.0, 0.08])
    radial = pos[:2] / np.linalg.norm(pos[:2])
    lateral = np.array([-radial[1], radial[0]])
    wall_side = pos[:2] + lateral * 0.08
    wall = np.array(
        [
            [wall_side[0] + dx, wall_side[1] + dy, 0.15]
            for dx in (-0.02, 0, 0.02)
            for dy in (-0.02, 0, 0.02)
        ]
    )
    monkeypatch.setattr(ex, "_neighbour_points", lambda exclude=None: wall)
    d = ex._aside_offset(pos, 0.05, object_id="banana_0")
    assert float(d[:2] @ lateral) < 0, "stepped toward the neighbour"


@pytest.mark.parametrize(
    "relation,axis,sign",
    [("left_of", 1, 1), ("right_of", 1, -1), ("in_front_of", 0, -1), ("behind", 0, 1)],
)
def test_place_beside_a_target_sets_the_object_down_on_that_side(relation, axis, sign):
    """``place`` supports left_of, right_of, in_front_of and behind as well as in and on."""
    emb, ex = make("banana_in_bowl")
    obj_id, target_id = goal_ids(emb, ex)
    assert ex.grasp(obj_id).outcome is Outcome.GRASPED
    result = ex.place(target_id, relation)
    if result.outcome is Outcome.REJECTED:
        assert "no free spot" in result.reason  # e.g. behind the bowl is out of reach
        return
    assert result.outcome is Outcome.PLACED, result.render()
    banana, bowl = emb.scene.body("banana"), emb.scene.body("bowl")
    gap = (banana.pos[axis] - bowl.pos[axis]) * sign
    assert gap > bowl.size[axis] / 2, (relation, banana.pos, bowl.pos)
    assert banana.bottom_z < emb.capabilities.workspace.table_z + 0.01, "it is on the table"


def test_a_thin_wall_hold_is_judged_by_the_scene_not_the_jaw(monkeypatch):
    """On a thin-wall pinch the jaw closes fully, as on air, so jaw travel is not read as a
    miss or a drop; the scene decides."""
    emb, ex = make("banana_in_bowl")
    obj_id, target_id = goal_ids(emb, ex)
    plan = ex.plan_for(obj_id)
    for c in plan.candidates:
        c.width_m = 0.005  # pretend it is a thin-wall pinch
    real_close = emb._close

    def close_fully(target):
        real_close(target)
        emb.robot.travel = 1.0  # held, but the jaw reads fully shut

    monkeypatch.setattr(emb, "_close", close_fully)
    result = ex.grasp(obj_id)
    assert result.outcome is Outcome.GRASPED, result.render()
    assert emb.robot.held == "banana"
    placed = ex.place(target_id)
    assert placed.outcome is Outcome.PLACED, placed.render()  # no false "fell out"


def test_a_thin_wall_pinch_on_nothing_is_still_a_miss(monkeypatch):
    emb, ex = make("banana_in_bowl")
    obj_id, _ = goal_ids(emb, ex)
    plan = ex.plan_for(obj_id)
    for c in plan.candidates:
        c.width_m = 0.005
    emb.scene.body("banana").graspable = False
    result = ex.grasp(obj_id)
    assert result.outcome is not Outcome.GRASPED, result.render()
    assert ex.memory.held is None


def test_a_destination_missed_by_one_look_is_placed_into_from_memory(monkeypatch):
    """A static destination missing from the latest look is placed into from its last
    remembered measurement."""
    emb, ex = make("banana_in_bowl")
    obj_id, target_id = goal_ids(emb, ex)
    assert ex.grasp(obj_id).outcome is Outcome.GRASPED
    build = ex.perception.build

    def forget_target(*args, **kwargs):
        catalog = build(*args, **kwargs)
        catalog.objects = [o for o in catalog.objects if o.label != emb.goal[1]]
        return catalog

    monkeypatch.setattr(ex.perception, "build", forget_target)
    ex.refresh()
    assert ex.catalog.get(target_id) is None
    result = ex.place(target_id)
    assert result.outcome is Outcome.PLACED, result.render()
    assert result.detail.get("target_from_memory") is True
    assert emb.grade()[0]


def test_an_object_never_seen_is_still_unknown_to_place():
    emb, ex = make("banana_in_bowl")
    obj_id, _ = goal_ids(emb, ex)
    assert ex.grasp(obj_id).outcome is Outcome.GRASPED
    assert ex.place("crate_9").outcome is Outcome.REJECTED


class _CountingBoxes:
    """Oracle boxes with the live grounding's contract: it remembers its last detections
    and counts model calls."""

    def __init__(self, emb):
        self.inner = OracleGrounding(emb)
        self.calls = 0
        self.last_detections = []
        self.name = "counting-oracle"

    def detect(self, observation, instruction=""):
        self.calls += 1
        self.last_detections = self.inner.detect(observation, instruction)
        return self.last_detections


def test_a_verified_grasp_re_measures_the_last_boxes_instead_of_grounding_again():
    """After a verified grasp only the held object has moved, so the last boxes are
    re-measured in fresh depth instead of calling the grounding model again."""
    emb = SCENES["banana_in_bowl"]()
    boxes = _CountingBoxes(emb)
    ex = Executor(
        emb,
        PerceptionService(boxes),
        AnalyticGraspPlanner(emb.capabilities),
        ObjectMemory(),
        emb.instruction,
    )
    ex.refresh()
    obj_id, target_id = goal_ids(emb, ex)
    before = boxes.calls
    assert ex.grasp(obj_id).outcome is Outcome.GRASPED
    assert boxes.calls == before
    assert ex.catalog.get(target_id) is not None
    assert not any("no usable depth" in f for f in ex.catalog.failures)
    assert ex.place(target_id).outcome is Outcome.PLACED
    assert boxes.calls == before + 1
    assert emb.grade()[0]
