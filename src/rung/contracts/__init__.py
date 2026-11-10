"""Typed contracts between the harness core, embodiment adapters and model backends."""

from rung.contracts.action import (
    ExecutionOutcome,
    ExecutionReport,
    GripperAction,
    GripperCommand,
    TCPTarget,
    residual_of,
)
from rung.contracts.capabilities import (
    ArmSpec,
    CameraSpec,
    EmbodimentCapabilities,
    GripperGeometry,
    Workspace,
)
from rung.contracts.frames import Pose
from rung.contracts.observation import ArmState, CameraFrame, GripperState, JointState, Observation
from rung.contracts.trace import RunProvenance, TraceEvent, TraceWriter, read_trace
from rung.contracts.validate import (
    ContractError,
    validate_observation,
    validate_target,
)

__all__ = [
    "ArmSpec",
    "ArmState",
    "CameraFrame",
    "CameraSpec",
    "ContractError",
    "EmbodimentCapabilities",
    "ExecutionOutcome",
    "ExecutionReport",
    "GripperAction",
    "GripperCommand",
    "GripperGeometry",
    "GripperState",
    "JointState",
    "Observation",
    "Pose",
    "RunProvenance",
    "TCPTarget",
    "TraceEvent",
    "TraceWriter",
    "Workspace",
    "read_trace",
    "residual_of",
    "validate_observation",
    "validate_target",
]
