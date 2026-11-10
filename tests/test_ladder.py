"""L0 and L2 on one task.

Not a result. Checks that one scene, embodiment and grader can be driven through two
interfaces that ask the model for different things, and that the difference is visible in
the trace.

Scripted answers stand in for a model, chosen as the shortest competent play at each
interface: L0 spends decisions on metric approach, L2 on choosing objects.
"""

from __future__ import annotations

import numpy as np

from rung.backends import MockBackend, call
from rung.contracts.trace import TraceWriter
from rung.executor import Executor
from rung.grasp import AnalyticGraspPlanner
from rung.interfaces import L0Policy, L2Policy
from rung.loop import EndReason, Limits, run_episode
from rung.mock import SCENES
from rung.perception import ObjectMemory, OracleGrounding, PerceptionService

LIMITS = Limits(horizon_s=400.0, max_decisions=30)


def l0_run():
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
                note="Above the banana.",
            ),
            call(
                "move_to",
                x=banana.pos[0],
                y=banana.pos[1],
                z=banana.top_z - 0.01,
                yaw_deg=yaw,
                gripper="close",
                note="Down and close.",
            ),
            call("move_to", x=banana.pos[0], y=banana.pos[1], z=0.25, yaw_deg=yaw, note="Lift."),
            call("move_to", x=bowl.pos[0], y=bowl.pos[1], z=0.22, yaw_deg=yaw, note="Carry."),
            call(
                "move_to",
                x=bowl.pos[0],
                y=bowl.pos[1],
                z=0.16,
                yaw_deg=yaw,
                gripper="open",
                note="Release.",
            ),
            call("done", summary="In the bowl."),
        ]
    )
    trace = TraceWriter(None)
    result = run_episode(emb, L0Policy(backend), task="banana_in_bowl", limits=LIMITS, trace=trace)
    return result, trace


def l2_run():
    emb = SCENES["banana_in_bowl"]()
    ex = Executor(
        emb,
        PerceptionService(OracleGrounding(emb)),
        AnalyticGraspPlanner(emb.capabilities),
        ObjectMemory(),
    )
    ex.refresh()
    backend = MockBackend(
        [
            call("grasp", object_id="banana_0", note="Picking up the banana."),
            call("place", target_id="bowl_0", note="Into the bowl."),
            call("done", summary="In the bowl."),
        ]
    )
    trace = TraceWriter(None)
    result = run_episode(
        emb, L2Policy(backend, ex), task="banana_in_bowl", limits=LIMITS, trace=trace
    )
    return result, trace


def test_both_rungs_solve_the_same_task_on_the_same_embodiment():
    l0, _ = l0_run()
    l2, _ = l2_run()
    assert l0.success and l2.success, (l0.grade_detail, l2.grade_detail)
    assert l0.end_reason is EndReason.DONE and l2.end_reason is EndReason.DONE
    assert l0.task == l2.task


def test_the_rungs_are_labelled_so_a_result_row_says_which_one_it_is():
    l0, _ = l0_run()
    l2, _ = l2_run()
    assert l0.interface == "L0" and l2.interface == "L2"
    assert l0.policy.startswith("l0/") and l2.policy.startswith("l2/")


def test_decisions_and_motions_are_counted_separately_at_both_rungs():
    """Decision counts count model calls, not motions: an L2 primitive that issues many
    targets is still one decision."""
    l0, l0_trace = l0_run()
    l2, l2_trace = l2_run()

    assert l0.decisions == l0_trace.count("decision")
    assert l2.decisions == l2_trace.count("decision")
    assert l2.decisions < l0.decisions, "L2 should need fewer model calls for this task"
    assert l2.servo_steps > l2.decisions, "and more motions than calls, run by the harness"


def test_only_the_upper_rung_produces_primitive_events():
    """The trace says which rung ran without being told: L2 emits primitives with
    outcomes, L0 emits only attempts and executions."""
    _, l0_trace = l0_run()
    _, l2_trace = l2_run()
    assert l0_trace.count("primitive") == 0
    assert l2_trace.count("primitive") > 0


def test_both_rungs_leave_a_trace_that_ends_with_the_grader():
    for _, trace in (l0_run(), l2_run()):
        kinds = [e.kind for e in trace.events]
        assert kinds[0] == "observation" and kinds[-1] == "episode_end"
        assert trace.events[-1].payload["success"] is True
