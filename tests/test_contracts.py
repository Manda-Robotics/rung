"""Contract validation: every rule in validate.py has a test that trips it."""

from __future__ import annotations

import numpy as np
import pytest

from rung.contracts.action import (
    ExecutionOutcome,
    GripperAction,
    GripperCommand,
    TCPTarget,
    residual_of,
)
from rung.contracts.capabilities import (
    CameraSpec,
    EmbodimentCapabilities,
    GripperGeometry,
    Workspace,
)
from rung.contracts.frames import Pose, quat_from_axis_angle
from rung.contracts.observation import CameraFrame, GripperState, Observation
from rung.contracts.trace import TraceEvent, TraceWriter, read_trace
from rung.contracts.validate import ContractError, validate_observation, validate_target


def _caps(**kw) -> EmbodimentCapabilities:
    base = dict(
        name="test",
        cameras=(CameraSpec("scene", 32, 24, has_depth=True),),
        gripper=GripperGeometry(0.085, 0.055, 0.022),
        workspace=Workspace(np.array([0.0, -0.5, 0.0]), np.array([0.9, 0.5, 0.7]), table_z=0.0),
        control_period_s=0.05,
        tcp_from_ee=Pose.identity(),
        max_lin_speed_mps=0.35,
    )
    base.update(kw)
    return EmbodimentCapabilities(**base)


def _frame(**kw) -> CameraFrame:
    base = dict(
        name="scene",
        rgb=np.zeros((24, 32, 3), dtype=np.uint8),
        t=0.0,
        depth=np.full((24, 32), 1.0, dtype=np.float32),
        K=np.array([[40.0, 0, 15.5], [0, 40.0, 11.5], [0, 0, 1.0]]),
        pose=Pose(np.array([0.0, -0.4, 0.5]), np.array([1.0, 0, 0, 0])),
    )
    base.update(kw)
    return CameraFrame(**base)


def _obs(**kw) -> Observation:
    base = dict(
        t=0.0,
        scene=_frame(),
        tcp=Pose(np.array([0.4, 0.0, 0.3]), np.array([1.0, 0, 0, 0])),
        gripper=GripperState(travel=0.0),
    )
    base.update(kw)
    return Observation(**base)


def test_valid_observation_passes():
    validate_observation(_obs(), _caps())


def test_rgb_must_be_uint8_hw3():
    with pytest.raises(ContractError, match="uint8"):
        validate_observation(_obs(scene=_frame(rgb=np.zeros((24, 32, 3), dtype=np.float32))))


def test_depth_must_be_float32_metres():
    with pytest.raises(ContractError, match="float32"):
        validate_observation(_obs(scene=_frame(depth=np.ones((24, 32), dtype=np.float64))))


def test_depth_shape_must_match_rgb():
    with pytest.raises(ContractError, match="shape"):
        validate_observation(_obs(scene=_frame(depth=np.ones((12, 16), dtype=np.float32))))


def test_depth_without_geometry_is_rejected():
    with pytest.raises(ContractError, match="geometry is unusable"):
        validate_observation(_obs(scene=_frame(K=None)))


def test_opengl_convention_must_be_converted_by_the_adapter():
    with pytest.raises(ContractError, match="opencv"):
        validate_observation(_obs(scene=_frame(convention="opengl")))


def test_principal_point_must_match_delivered_resolution():
    """An image resized without rescaling ``K`` fails validation."""
    K = np.array([[40.0, 0, 160.0], [0, 40.0, 120.0], [0, 0, 1.0]])
    with pytest.raises(ContractError, match="principal point"):
        validate_observation(_obs(scene=_frame(K=K)))


def test_camera_resolution_must_match_capabilities():
    caps = _caps(cameras=(CameraSpec("scene", 64, 48, has_depth=True),))
    with pytest.raises(ContractError, match="capabilities declare"):
        validate_observation(_obs(), caps)


def test_declared_depth_must_be_delivered():
    with pytest.raises(ContractError, match="declares depth"):
        validate_observation(_obs(scene=_frame(depth=None, K=None, pose=None)), _caps())


def test_gripper_travel_range():
    with pytest.raises(ContractError, match=r"travel"):
        validate_observation(_obs(gripper=GripperState(travel=1.4)))


def test_tcp_far_outside_workspace_is_a_contract_error():
    """Catches a wrong root frame or a missing TCP offset, not a near-boundary pose."""
    far = Pose(np.array([2.0, 0.0, 0.3]), np.array([1.0, 0, 0, 0]))
    with pytest.raises(ContractError, match="workspace"):
        validate_observation(_obs(tcp=far), _caps())


def test_capture_time_cannot_be_in_the_future():
    with pytest.raises(ContractError, match="capture time"):
        validate_observation(_obs(t=0.0, scene=_frame(t=0.5)), _caps())


def test_target_speed_capped_by_embodiment():
    p = Pose(np.array([0.4, 0, 0.2]), np.array([1.0, 0, 0, 0]))
    t = TCPTarget(pose=p, max_lin_speed_mps=1.0)
    with pytest.raises(ContractError, match="exceeds embodiment cap"):
        validate_target(t, _caps())


def test_target_angular_speed_capped_by_embodiment():
    t = TCPTarget(
        pose=Pose(np.array([0.4, 0, 0.2]), np.array([1.0, 0, 0, 0])), max_ang_speed_rps=9.0
    )
    with pytest.raises(ContractError, match="max_ang_speed_rps"):
        validate_target(t, _caps())


def test_target_timeout_must_be_positive():
    t = TCPTarget(pose=Pose(np.array([0.4, 0, 0.2]), np.array([1.0, 0, 0, 0])), timeout_s=0.0)
    with pytest.raises(ContractError, match="timeout"):
        validate_target(t, _caps())


def test_gripper_opening_within_geometry():
    t = TCPTarget(
        pose=Pose(np.array([0.4, 0, 0.2]), np.array([1.0, 0, 0, 0])),
        gripper=GripperCommand(GripperAction.OPEN, opening_m=0.2),
    )
    with pytest.raises(ContractError, match="opening"):
        validate_target(t, _caps())


def test_residual_of_reports_largest_axis():
    target = TCPTarget(pose=Pose(np.array([0.4, 0.1, 0.2]), np.array([1.0, 0, 0, 0])))
    actual = Pose(np.array([0.39, 0.1, 0.26]), quat_from_axis_angle([0, 0, 1], 0.2))
    mag, axis, rot = residual_of(target, actual)
    assert axis == "z"
    assert mag == pytest.approx(0.06, abs=1e-9)
    assert rot == pytest.approx(0.2, abs=1e-9)


def test_target_ids_are_unique():
    p = Pose(np.array([0.4, 0, 0.2]), np.array([1.0, 0, 0, 0]))
    assert TCPTarget(pose=p).id != TCPTarget(pose=p).id


def test_trace_writer_round_trip(tmp_path):
    w = TraceWriter(tmp_path / "trace.jsonl")
    w.emit("observation", 0.0, episode="ep0", prompt_hash="abc")
    w.emit("attempt", 0.1, episode="ep0", servo=True, target=np.array([1.0, 2.0]))
    w.emit("model_call", 0.1, episode="ep0", usage={"in": 10, "out": 3})
    rows = read_trace(tmp_path / "trace.jsonl")
    assert [r["kind"] for r in rows] == ["observation", "attempt", "model_call"]
    assert rows[1]["payload"]["target"] == [1.0, 2.0]
    assert [r["seq"] for r in rows] == [1, 2, 3]


def test_servo_rows_do_not_count_as_model_calls(tmp_path):
    w = TraceWriter(tmp_path / "t.jsonl")
    w.emit("attempt", 0.0, episode="e", servo=True)
    w.emit("attempt", 0.1, episode="e", servo=False)
    w.emit("model_call", 0.1, episode="e")
    assert w.count("attempt", servo=False) == 1
    assert w.count("model_call") == 1


def test_unknown_event_kind_rejected():
    with pytest.raises(ValueError):
        TraceEvent(kind="chatter", t_sim=0.0, payload={}, episode="e")


def test_execution_outcome_values_are_stable():
    """These strings end up in saved traces; changing one invalidates old runs."""
    assert {o.value for o in ExecutionOutcome} == {
        "reached",
        "expired",
        "superseded",
        "clamped",
        "rejected",
        "blocked",
    }


def test_to_jsonable_handles_tensor_like_objects():
    """The trace encoder serialises torch-like tensors without importing torch.

    The fake stands in for a CUDA tensor that requires grad, which rejects a direct
    ``numpy()``; the encoder detaches, moves to host, then calls ``tolist``.
    """

    class FakeTensor:
        def __init__(self, values, *, detached=False, on_host=False):
            self.values, self.detached, self.on_host = values, detached, on_host

        def detach(self):
            return FakeTensor(self.values, detached=True, on_host=self.on_host)

        def cpu(self):
            return FakeTensor(self.values, detached=self.detached, on_host=True)

        def tolist(self):
            if not (self.detached and self.on_host):
                raise RuntimeError("read a live device tensor")
            return list(self.values)

    from rung.contracts.trace import to_jsonable

    assert to_jsonable({"pos": FakeTensor([1.0, 2.0])}) == {"pos": [1.0, 2.0]}
