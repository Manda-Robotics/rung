"""L2: the prompt, the tool surface, dispatch, and what the model is and is not told.

Most tests cover the prompt, which is easy to change unnoticed; the controlled track
requires every model to receive the same text.
"""

from __future__ import annotations

import numpy as np
import pytest

from rung.backends import MockBackend, ModelResponse, ToolCall, call
from rung.executor import Executor, Outcome
from rung.grasp import AnalyticGraspPlanner
from rung.interfaces import L2Config, L2Policy, mark_objects
from rung.interfaces.l2 import tool_specs
from rung.loop import EndReason, Limits, run_episode
from rung.mock import SCENES
from rung.perception import ObjectMemory, OracleGrounding, PerceptionService
from rung.policy import Policy


def make(scene: str = "banana_in_bowl", responses=None, config: L2Config | None = None):
    emb = SCENES[scene]()
    ex = Executor(
        emb,
        PerceptionService(OracleGrounding(emb)),
        AnalyticGraspPlanner(emb.capabilities),
        ObjectMemory(),
    )
    backend = MockBackend(responses or [])
    return emb, ex, backend, L2Policy(backend, ex, config)


SOLVE_BANANA = [
    call("inspect", object_id="banana_0", note="Checking the banana's width first."),
    call("grasp", object_id="banana_0", note="It fits the jaw, picking it up."),
    call("place", target_id="bowl_0", relation="in", note="Releasing over the bowl."),
    call("done", summary="The banana is in the bowl."),
]


def test_l2_is_a_policy():
    _, _, _, policy = make()
    assert isinstance(policy, Policy)
    assert policy.interface == "L2"
    assert policy.name == "l2/mock-model"


def test_a_scripted_model_solves_the_task_through_the_primitives():
    emb, _, backend, policy = make(responses=SOLVE_BANANA)
    result = run_episode(emb, policy, task="banana_in_bowl")
    assert result.success, result.grade_detail
    assert result.end_reason is EndReason.DONE
    assert result.decisions == 4
    assert backend.calls == 4


def test_usage_and_dollars_are_accumulated():
    emb, _, _, policy = make(responses=SOLVE_BANANA)
    result = run_episode(emb, policy, task="t")
    assert result.usage.calls == 4
    assert result.usage.input_tokens == 4 * 900
    assert result.usage.dollars == 0.0  # the mock model has no price, and is not invented


def test_the_prompt_carries_the_catalog_with_measurements():
    emb, _, backend, policy = make(responses=SOLVE_BANANA)
    run_episode(emb, policy, task="t")
    first = policy.prompts[0]
    assert "Task: Pick up the banana and place it in the bowl" in first
    assert "banana_0 (banana)" in first
    assert "cm across at grasp height" in first
    assert "jaw opens to 8.5 cm" in first


def test_the_prompt_carries_memory_and_the_last_result():
    emb, _, _, policy = make(responses=SOLVE_BANANA)
    run_episode(emb, policy, task="t")
    assert "not holding anything" in policy.prompts[0]
    third = policy.prompts[2]
    assert "holding banana_0" in third
    assert "grasp(banana_0) -> grasped" in third
    assert "Result of your last call" in third


def test_both_the_raw_and_the_marked_image_are_sent():
    """A mark can cover the object it names, so the unmarked view goes too."""
    emb, _, backend, policy = make(responses=SOLVE_BANANA)
    run_episode(emb, policy, task="t")
    labels = [img.label for img in backend.last_images()]
    assert "scene" in labels and "scene_marked" in labels and "wrist" in labels


def test_images_can_be_trimmed_by_configuration():
    emb, _, backend, policy = make(
        responses=SOLVE_BANANA, config=L2Config(send_marked_image=False, send_wrist=False)
    )
    run_episode(emb, policy, task="t")
    assert [img.label for img in backend.last_images()] == ["scene"]


def test_the_tool_surface_is_stable():
    """These names end up in every trace; renaming one invalidates earlier runs."""
    assert [t.name for t in tool_specs(has_wrist=True)] == [
        "inspect",
        "propose_grasps",
        "grasp",
        "place",
        "move_to",
        "release",
        "observe",
        "done",
        "give_up",
    ]


def test_every_acting_tool_requires_a_note():
    for spec in tool_specs(True):
        if spec.name in ("done", "give_up"):
            continue
        assert "note" in spec.parameters["required"], spec.name


def test_the_wrist_camera_is_only_offered_when_the_rig_has_one():
    with_wrist = next(t for t in tool_specs(True) if t.name == "observe")
    without = next(t for t in tool_specs(False) if t.name == "observe")
    assert with_wrist.parameters["properties"]["camera"]["enum"] == ["scene", "wrist"]
    assert without.parameters["properties"]["camera"]["enum"] == ["scene"]


def test_a_rejected_grasp_reaches_the_model_with_its_reason():
    emb, _, _, policy = make(
        "cluttered_bin",
        responses=[
            call("grasp", object_id="bin_0", note="Trying to pick up the bin."),
            call("give_up", reason="The bin is furniture, not an object to move."),
        ],
    )
    result = run_episode(emb, policy, task="t")
    assert result.end_reason is EndReason.GAVE_UP
    assert "too wide for the jaw" in policy.prompts[1]


def test_the_unmeasurable_grasp_reason_reaches_the_model():
    emb, _, _, policy = make(
        "tall_bottle_and_crate",
        responses=[
            call("propose_grasps", object_id="bottle_0", note="What can I do with the bottle?"),
            call("done", summary="Stopping here."),
        ],
    )
    run_episode(emb, policy, task="t")
    assert "cannot measure the width" in policy.prompts[1]


def test_done_is_not_success():
    emb, _, _, policy = make(responses=[call("done", summary="I think it is finished.")])
    result = run_episode(emb, policy, task="t")
    assert result.end_reason is EndReason.DONE
    assert not result.success


def test_a_refusal_ends_the_episode_without_being_treated_as_an_error():
    emb, _, _, policy = make(responses=[ModelResponse(refusal="declined for safety")])
    result = run_episode(emb, policy, task="t")
    assert result.end_reason is EndReason.GAVE_UP
    assert "declined for safety" in result.notes[0]


def test_a_reply_with_no_tool_call_is_fed_back_and_retried():
    emb, _, _, policy = make(
        responses=[
            ModelResponse(text="I would like to pick up the banana."),
            call("grasp", object_id="banana_0", note="Now as a tool call."),
            call("place", target_id="bowl_0", note="Into the bowl."),
            call("done", summary="Done."),
        ]
    )
    result = run_episode(emb, policy, task="t")
    assert result.success
    assert "could not be used" in policy.prompts[1]


def test_an_unknown_tool_is_reported_as_a_policy_error():
    emb, _, _, policy = make(responses=[ToolCall("teleport", {"note": "worth a try"})] * 3)
    result = run_episode(emb, policy, task="t", limits=Limits(max_consecutive_errors=3))
    assert result.end_reason is EndReason.POLICY_ERROR
    assert "unknown tool 'teleport'" in (result.error or "")


def test_move_to_takes_metres_and_degrees():
    emb, ex, _, policy = make(
        responses=[call("move_to", x=0.5, y=0.1, z=0.3, yaw_deg=90, note="Getting a clear view.")]
    )
    policy.reset(emb.instruction, emb.capabilities)
    decision = policy.act(emb.observe(), None)
    assert decision.kind == "move_to"
    assert np.allclose(emb.observe().tcp.pos, emb.cfg.home)
    policy.execute_decision(decision, lambda target, **kwargs: emb.execute(target))
    assert decision.payload["outcome"] == Outcome.MOVED.value
    assert np.allclose(emb.observe().tcp.pos, [0.5, 0.1, 0.3], atol=1e-6)


def test_dispatch_maps_every_tool_to_a_primitive():
    emb, ex, _, policy = make("cluttered_bin")
    policy.reset(emb.instruction, emb.capabilities)
    ex.refresh()
    for tool, args in [
        ("inspect", {"object_id": "apple_0"}),
        ("propose_grasps", {"object_id": "apple_0"}),
        ("observe", {}),
        ("grasp", {"object_id": "apple_0"}),
        ("place", {"target_id": "bin_0"}),
        ("release", {}),
    ]:
        result = policy._run(ToolCall(tool, {**args, "note": "n"}))
        assert result.primitive in (tool, "observe", "release"), (tool, result.primitive)


def test_marked_image_differs_from_the_raw_one_and_keeps_its_shape():
    emb, ex, _, _ = make("cluttered_bin")
    catalog = ex.refresh()
    obs = emb.observe()
    marked = mark_objects(obs, catalog)
    assert marked.shape == obs.scene.rgb.shape
    assert marked.dtype == np.uint8
    assert not np.array_equal(marked, obs.scene.rgb)


def test_prompts_are_recorded_verbatim_for_the_transcript():
    emb, _, _, policy = make(responses=SOLVE_BANANA)
    run_episode(emb, policy, task="t")
    assert len(policy.prompts) == 4
    assert policy.prompts[0].startswith("[system]")
    assert "<image scene" in policy.prompts[0]
    assert "[tools] inspect, propose_grasps" in policy.prompts[0]


def test_reset_clears_memory_between_episodes():
    emb, ex, backend, policy = make(responses=SOLVE_BANANA + SOLVE_BANANA)
    run_episode(emb, policy, task="t")
    assert ex.memory.attempts
    policy.reset(emb.instruction, emb.capabilities)
    assert not ex.memory.attempts
    assert ex.memory.held is None
    assert policy.turn == 0


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_a_model_that_names_the_right_objects_solves_every_scene(scene):
    emb = SCENES[scene]()
    obj_label, target_label = emb.goal
    ex = Executor(
        emb,
        PerceptionService(OracleGrounding(emb)),
        AnalyticGraspPlanner(emb.capabilities),
        ObjectMemory(),
    )
    ids = {o.label: o.object_id for o in ex.refresh().objects}
    backend = MockBackend(
        [
            call("grasp", object_id=ids[obj_label], note="Picking it up."),
            call("place", target_id=ids[target_label], note="Placing it."),
            call("done", summary="Finished."),
        ]
    )
    result = run_episode(emb, L2Policy(backend, ex), task=scene)
    assert result.success, result.grade_detail


def test_grounding_calls_are_counted_in_the_episode_cost():
    """A turn that grounds and then acts counts and bills two model calls."""
    from rung.backends import ModelResponse
    from rung.perception import BoxGrounding
    from rung.policy import ModelUsage

    emb = SCENES["banana_in_bowl"]()
    obs = emb.observe()
    boxes = []
    for body in emb.scene.bodies:
        from rung.contracts.frames import project

        uv, _ = project(np.asarray(body.pos)[None], obs.scene.K, obs.scene.pose)
        u, v = float(uv[0, 0]), float(uv[0, 1])
        boxes.append(
            {
                "label": body.name,
                "box_2d": [
                    (v - 25) / obs.scene.height * 1000,
                    (u - 25) / obs.scene.width * 1000,
                    (v + 25) / obs.scene.height * 1000,
                    (u + 25) / obs.scene.width * 1000,
                ],
            }
        )

    import json as _json

    grounding_backend = MockBackend(
        [
            ModelResponse(
                text=_json.dumps(boxes),
                usage=ModelUsage(input_tokens=1500, output_tokens=80, dollars=0.004, calls=1),
            )
        ]
        * 4
    )
    ex = Executor(
        emb,
        PerceptionService(BoxGrounding(grounding_backend)),
        AnalyticGraspPlanner(emb.capabilities),
        ObjectMemory(),
    )
    acting = MockBackend(
        [
            ModelResponse(
                tool_calls=[call("done", summary="Stopping.").__class__("done", {"summary": "s"})],
                usage=ModelUsage(input_tokens=900, output_tokens=40, dollars=0.01, calls=1),
            )
        ]
    )
    result = run_episode(emb, L2Policy(acting, ex), task="t")
    assert result.usage.calls == 2, "one acting call plus one grounding call"
    assert result.usage.dollars == pytest.approx(0.014)


def test_full_quaternion_is_exposed_and_executed():
    from rung.contracts.frames import quat_angle_between, quat_from_axis_angle

    q = quat_from_axis_angle([0, 1, 0], 0.7)
    emb, _, _, policy = make(
        responses=[
            call("move_to", x=0.5, y=0.1, z=0.4, quat_wxyz=q.tolist(), note="Tilt."),
            call("done", summary="Finished."),
        ]
    )
    assert (
        "quat_wxyz"
        in next(t for t in tool_specs(True) if t.name == "move_to").parameters["properties"]
    )
    run_episode(emb, policy, task="t")
    assert quat_angle_between(emb.observe().tcp.quat, q) < 1e-6


def test_l2_decision_is_traced_before_motion_and_substeps_are_counted(tmp_path):
    from rung.contracts.trace import TraceWriter

    trace = TraceWriter(tmp_path / "trace.jsonl")
    emb, _, _, policy = make(responses=SOLVE_BANANA)
    result = run_episode(emb, policy, task="t", trace=trace)
    kinds = [e.kind for e in trace.events]
    assert kinds.index("decision") < kinds.index("attempt")
    assert result.servo_steps > 0
    assert result.servo_steps == kinds.count("execution")


def test_l2_embodiment_failure_is_not_retried_as_a_model_error():
    emb, _, backend, policy = make(responses=SOLVE_BANANA[1:])

    def broken(target):
        raise RuntimeError("robot disconnected")

    emb.execute = broken
    result = run_episode(emb, policy, task="t")
    assert result.end_reason is EndReason.EMBODIMENT_ERROR
    assert backend.calls == 1
    assert result.usage.calls == 1
    assert "robot disconnected" in result.error


def test_l2_horizon_interrupts_inside_a_primitive():
    emb, _, _, policy = make(responses=SOLVE_BANANA[1:])
    result = run_episode(emb, policy, task="t", limits=Limits(horizon_s=0.1))
    assert result.end_reason is EndReason.HORIZON
    assert result.sim_seconds <= 0.1 + 1e-8
    assert result.servo_steps == 1
    assert not any(e["event"] == "grasped" for e in emb.events)


def test_invalid_model_responses_are_still_billed():
    emb, _, _, policy = make(responses=[ModelResponse(text="no tool")] * 3)
    result = run_episode(emb, policy, task="t")
    assert result.end_reason is EndReason.POLICY_ERROR
    assert result.usage.calls == 3


def test_grounding_during_primitive_execution_is_billed():
    from rung.policy import ModelUsage

    emb, ex, _, policy = make(responses=[SOLVE_BANANA[1], SOLVE_BANANA[-1]])
    build = ex.perception.build
    calls = []

    def counted(*args, **kwargs):
        catalog = build(*args, **kwargs)
        calls.append(1)
        catalog.extra["grounding_usage"] = ModelUsage(calls=1, dollars=0.01)
        return catalog

    ex.perception.build = counted
    result = run_episode(emb, policy, task="t")
    # the first view, and the look that ends the grasp; the second turn reuses that look
    assert len(calls) == 2
    assert result.usage.calls == 2 + len(calls)
    assert result.usage.dollars == pytest.approx(0.01 * len(calls))


def test_exhausting_dollar_budget_prevents_the_selected_motion():
    from rung.policy import ModelUsage

    response = ModelResponse(
        tool_calls=[call("grasp", object_id="banana_0", note="Pick up.")],
        usage=ModelUsage(calls=1, dollars=0.02),
    )
    emb, _, _, policy = make(responses=[response])
    result = run_episode(emb, policy, task="t", limits=Limits(max_dollars=0.01))
    assert result.end_reason is EndReason.DOLLARS
    assert result.usage.dollars == pytest.approx(0.02)
    assert result.servo_steps == 0
    assert result.sim_seconds == 0


@pytest.mark.parametrize(
    "args",
    [
        {"quat_wxyz": [0, 0, 0, 0]},
        {"quat_wxyz": [1, 0, 0, 0], "yaw_deg": 0},
        {"yaw_deg": float("nan")},
    ],
)
def test_invalid_or_ambiguous_orientation_is_rejected_before_execution(args):
    from rung.interfaces.l2 import _pose_from

    with pytest.raises(ValueError):
        _pose_from({"x": 0.4, "y": 0.0, "z": 0.3, **args})


def test_a_missing_tool_argument_is_retried_not_fatal():
    """A tool call missing a required argument is rejected inside ``act``, so the turn is
    retried instead of the episode ending."""
    emb, _, _, policy = make(
        responses=[
            ToolCall("grasp", {"note": "picking it up"}),  # no object_id
            call("grasp", object_id="banana_0", note="now with the id"),
            call("place", target_id="bowl_0", note="into the bowl"),
            call("done", summary="done"),
        ]
    )
    result = run_episode(emb, policy, task="t")
    assert result.success, result.grade_detail
    assert result.end_reason is EndReason.DONE
    assert "needs object_id" in policy.prompts[1]


def test_a_primitive_error_is_reported_to_the_model_not_fatal():
    emb, ex, _, policy = make(
        responses=[
            call("grasp", object_id="banana_0", note="picking it up"),
            call("give_up", reason="stopping"),
        ]
    )

    def explode(*_a, **_k):
        raise ValueError("the perception service fell over")

    ex.grasp = explode
    result = run_episode(emb, policy, task="t")
    assert result.end_reason is EndReason.GAVE_UP
    assert result.error is None
    assert "could not be carried out" in policy.prompts[1]


def test_the_recorder_keeps_every_prompt_and_the_catalog_behind_it(tmp_path):
    """An episode must be replayable, not only watchable: the exact prompt text and the
    catalog with its detection boxes, for every model call."""
    import json

    from rung.record import Recorder

    emb, _, _, policy = make(responses=SOLVE_BANANA)
    result = run_episode(emb, policy, task="t", recorder=Recorder(tmp_path), episode_id="ep")
    folder = tmp_path / "ep"
    prompts = sorted(folder.glob("prompt_*.txt"))
    catalogs = sorted(folder.glob("catalog_*.json"))
    assert len(prompts) == len(policy.prompts) == 4, result
    assert prompts[0].read_text().startswith("[system]")
    first = json.loads(catalogs[0].read_text())
    assert first["objects"] and first["objects"][0]["box_px"] is not None
    assert {"object_id", "top_z", "grasp_width"} <= set(first["objects"][0])


def test_the_prompt_does_not_tell_the_model_to_give_up_on_wide_objects():
    """The system prompt does not call objects wider than the jaw out of reach; it points
    the model at ``propose_grasps`` and rim pinches instead."""
    from rung.interfaces.l2 import SYSTEM

    assert "genuinely out of reach" not in SYSTEM
    assert "propose_grasps" in SYSTEM and "rim" in SYSTEM


def test_the_trace_shows_every_harness_step_between_model_calls(tmp_path):
    """The transcript must show what happened, not only what the model chose: the model
    request and response, the perception catalog, the grasp plan and the checks."""
    import json

    from rung.contracts.trace import TraceWriter

    emb, _, _, policy = make(responses=SOLVE_BANANA)
    trace = TraceWriter(tmp_path / "trace.jsonl")
    run_episode(emb, policy, task="t", trace=trace, episode_id="ep")
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    kinds = [e["kind"] for e in events]
    for kind in (
        "model_request",
        "model_response",
        "grounding",
        "perception",
        "plan",
        "check",
        "decision",
        "primitive",
    ):
        assert kind in kinds, (kind, sorted(set(kinds)))
    first_request = kinds.index("model_request")
    assert kinds.index("perception") < first_request < kinds.index("decision")
    checks = [e["payload"] for e in events if e["kind"] == "check"]
    assert any(c["check"] == "jaw after close" for c in checks)


def test_a_turn_reuses_the_last_look_when_no_time_has_passed(tmp_path):
    """A primitive ends with a fresh look, so the next turn at the same simulated instant
    reuses that catalog instead of grounding again."""
    import json

    from rung.contracts.trace import TraceWriter

    emb, ex, _, policy = make(responses=SOLVE_BANANA)
    calls = []
    build = ex.perception.build
    ex.perception.build = lambda *a, **k: (calls.append(1), build(*a, **k))[1]
    trace = TraceWriter(tmp_path / "trace.jsonl")
    run_episode(emb, policy, task="t", trace=trace, episode_id="ep")
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    reused = [e for e in events if e["kind"] == "perception" and e["payload"].get("reused")]
    assert reused, "turns after a primitive should reuse its last look"
    # inspect does not move; grasp and place end with a look at their final instant
    assert len(calls) <= 4, len(calls)


def test_images_go_to_the_model_as_jpeg():
    from rung.interfaces.l2 import _image_block

    block = _image_block(np.zeros((72, 128, 3), dtype=np.uint8), "scene")
    assert block.mime == "image/jpeg" and block.data[:2] == b"\xff\xd8"
