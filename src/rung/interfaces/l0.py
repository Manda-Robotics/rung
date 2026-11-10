"""L0: pose targets.

The model sees the scene (and optionally wrist) image, the TCP position and yaw, the
gripper state and the reachable box, and returns an absolute fingertip position in the
root frame (metres) with a yaw and a gripper action. All metric reasoning is the model's;
the harness only interpolates and clamps. L0 measures unaided metric spatial reasoning
from images.

The interface follows RoboCurve's inspect-robots agent: absolute targets rather than
deltas, a largest-remaining-offset line after every motion, a mandatory note per call,
and explicit ``done`` and ``give_up``. Scores are interface-aligned with RoboCurve's but
not numerically comparable, since the simulator, rig, cameras and grader differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rung.backends.base import (
    Message,
    ModelBackend,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolSpec,
)
from rung.contracts.action import GripperAction, GripperCommand, TCPTarget
from rung.contracts.capabilities import EmbodimentCapabilities
from rung.contracts.frames import Pose, quat_from_axis_angle, quat_mul
from rung.contracts.observation import Observation
from rung.interfaces.l2 import _image_block, _render_prompt
from rung.policy import Decision, Feedback, ModelUsage, Verdict

SYSTEM = """You are operating a single robot arm with a parallel-jaw gripper on a tabletop.

You move the arm by naming where its fingertips should go, as an absolute position in the
robot's own frame, in metres. Nothing measures the scene for you: you decide from the
pictures where things are and how far to move.

The frame: x runs away from the robot's base across the table, y runs to the left, z runs
up, and z = 0 is the table top. The gripper points straight down. `yaw_deg` turns the jaw
about the vertical axis, so it decides which way the fingers close across an object.

Each turn you are told where the fingertips are now, what the gripper is doing, and, after
a move, how far the arm actually got. A move is played out at a safe speed and may not
finish inside its time; if so you are told the largest remaining offset and can simply ask
again.

Work in small, deliberate steps and look after each one. Every call needs a short `note`
saying what you see and why you chose it; a person reads those to follow your reasoning.

Call exactly one tool per turn."""

NOTE_SCHEMA = {
    "type": "string",
    "description": (
        "One or two plain sentences: what you observe right now and why you chose this. "
        "Written for a person watching, not for yourself."
    ),
}


def tool_specs(has_wrist: bool) -> list[ToolSpec]:
    cameras = ["scene"] + (["wrist"] if has_wrist else [])
    return [
        ToolSpec(
            "move_to",
            "Move the fingertips to an absolute position in the robot frame, in metres, "
            "and optionally open or close the gripper when they arrive.",
            {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "metres, away from the base"},
                    "y": {"type": "number", "description": "metres, to the left"},
                    "z": {"type": "number", "description": "metres, up from the table"},
                    "yaw_deg": {
                        "type": "number",
                        "description": "rotation of the jaw about the vertical axis",
                    },
                    "gripper": {
                        "type": "string",
                        "enum": ["open", "close", "hold"],
                        "description": "what to do with the fingers on arrival",
                    },
                    "note": NOTE_SCHEMA,
                },
                "required": ["x", "y", "z", "note"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "observe",
            "Take a fresh look without moving.",
            {
                "type": "object",
                "properties": {
                    "camera": {"type": "string", "enum": cameras, "default": "scene"},
                    "note": NOTE_SCHEMA,
                },
                "required": ["note"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "done",
            "The task is complete. The trial ends and a separate grader decides the outcome.",
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "give_up",
            "The task cannot be done with this rig. Say why. The trial ends.",
            {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        ),
    ]


@dataclass
class L0Config:
    effort: str | None = None
    max_tokens: int = 2048
    send_wrist: bool = True
    image_scale: float = 1.0
    max_speed_mps: float = 0.10
    timeout_s: float = 10.0


class L0Policy:
    """A model naming poses, with the harness doing nothing but carrying them out."""

    interface = "L0"

    def __init__(self, backend: ModelBackend, config: L0Config | None = None):
        self.backend = backend
        self.cfg = config or L0Config()
        self.instruction = ""
        self.caps: EmbodimentCapabilities | None = None
        self.turn = 0
        self.prompts: list[str] = []
        self.last_error: str | None = None
        self.pending_usage = ModelUsage()
        self._feedback: Feedback | None = None

    @property
    def name(self) -> str:
        return f"l0/{self.backend.model}"

    def reset(self, instruction: str, capabilities: EmbodimentCapabilities) -> None:
        self.instruction = instruction
        self.caps = capabilities
        self.turn = 0
        self.prompts = []
        self.last_error = None
        self.pending_usage = ModelUsage()
        self._feedback = None

    def drain_usage(self) -> ModelUsage:
        usage = self.pending_usage
        self.pending_usage = ModelUsage()
        return usage

    def act(self, observation: Observation, feedback: Feedback) -> Decision:
        self.turn += 1
        self._feedback = feedback
        request = self.build_request(observation, feedback)
        self.prompts.append(_render_prompt(request))
        response = self.backend.complete(request)
        self.pending_usage.add(response.usage)

        if response.refusal:
            return Decision(
                "refusal",
                f"The model declined: {response.refusal}",
                verdict=Verdict.GIVE_UP,
                reason=response.refusal,
            )
        call = response.call
        if call is None:
            self.last_error = "no tool call in the response. Reply with exactly one tool call."
            raise ValueError(self.last_error)
        try:
            return self.dispatch(call, response)
        except ValueError as exc:
            self.last_error = str(exc)
            raise

    def build_request(self, observation: Observation, feedback: Feedback | None) -> ModelRequest:
        blocks: list[Any] = [TextBlock(f"Task: {self.instruction}")]
        blocks.append(TextBlock("Camera view:"))
        blocks.append(_image_block(observation.scene.rgb, "scene", self.cfg.image_scale))
        if self.cfg.send_wrist and observation.wrist is not None:
            blocks.append(TextBlock("Wrist view:"))
            blocks.append(_image_block(observation.wrist.rgb, "wrist", self.cfg.image_scale))

        blocks.append(TextBlock(self._state_lines(observation)))
        if self.last_error:
            blocks.append(TextBlock(f"Your last reply could not be used: {self.last_error}"))
            self.last_error = None
        elif feedback is not None and feedback.lines:
            blocks.append(TextBlock(feedback.render()))
        blocks.append(TextBlock("Choose one tool."))

        return ModelRequest(
            system=SYSTEM,
            messages=[Message.user(*blocks)],
            tools=tool_specs(has_wrist=observation.wrist is not None),
            effort=self.cfg.effort,
            max_tokens=self.cfg.max_tokens,
        )

    def _state_lines(self, observation: Observation) -> str:
        """TCP and gripper state, in the same frame and units the model answers in.

        The reachable box is included so the model does not have to discover it by
        being clamped.
        """
        from rung.contracts.frames import yaw_of
        from rung.loop import describe_gripper

        pos = observation.tcp.pos
        lines = [
            f"Fingertips are at x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f} metres, "
            f"jaw at {np.degrees(yaw_of(observation.tcp.quat)):.0f} degrees.",
            f"Gripper: {describe_gripper(observation)}.",
        ]
        if self.caps is not None:
            ws = self.caps.workspace
            lines.append(
                f"Reachable: x {ws.min_xyz[0]:.2f} to {ws.max_xyz[0]:.2f}, "
                f"y {ws.min_xyz[1]:.2f} to {ws.max_xyz[1]:.2f}, "
                f"z {ws.table_z:.2f} to {ws.max_xyz[2]:.2f}. "
                f"The jaw opens to {self.caps.gripper.max_opening_m * 100:.1f} cm."
            )
        return "\n".join(lines)

    def dispatch(self, call: ToolCall, response: ModelResponse) -> Decision:
        args = call.arguments
        note = str(args.get("note") or args.get("summary") or args.get("reason") or "")
        payload: dict[str, Any] = {"tool": call.name, "arguments": args}
        if response.thoughts:
            payload["thoughts"] = response.thoughts
        if response.text:
            payload["response_text"] = response.text

        if call.name == "done":
            return Decision("done", note or "Task complete.", verdict=Verdict.DONE, payload=payload)
        if call.name == "give_up":
            return Decision(
                "give_up",
                note or "Cannot continue.",
                verdict=Verdict.GIVE_UP,
                reason=str(args.get("reason", "")),
                payload=payload,
            )
        if call.name == "observe":
            return Decision("observe", note or "Looking again.", payload=payload)
        if call.name != "move_to":
            known = ", ".join(t.name for t in tool_specs(True))
            raise ValueError(f"unknown tool {call.name!r}; available: {known}")

        target = self._target_from(args)
        payload["pose"] = target.pose.pos
        return Decision(kind="move_to", note=note, targets=[target], payload=payload)

    def _target_from(self, args: dict[str, Any]) -> TCPTarget:
        for key in ("x", "y", "z"):
            if args.get(key) is None:
                raise ValueError(f"move_to needs x, y and z; {key} was missing")
        try:
            position = np.array([float(args["x"]), float(args["y"]), float(args["z"])])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"move_to needs numbers for x, y and z: {exc}") from exc
        if not np.all(np.isfinite(position)):
            raise ValueError("move_to needs finite numbers for x, y and z")

        yaw = float(np.radians(float(args.get("yaw_deg") or 0.0)))
        action = {
            "open": GripperAction.OPEN,
            "close": GripperAction.CLOSE,
            "hold": GripperAction.HOLD,
        }.get(str(args.get("gripper", "hold")).lower())
        if action is None:
            raise ValueError("gripper must be one of open, close, hold")

        caps = self.caps
        return TCPTarget(
            pose=_top_down(position, yaw),
            gripper=GripperCommand(action),
            max_lin_speed_mps=min(self.cfg.max_speed_mps, caps.max_lin_speed_mps if caps else 0.25),
            max_ang_speed_rps=min(1.0, caps.max_ang_speed_rps if caps else 1.0),
            timeout_s=self.cfg.timeout_s,
            meta={"phase": "move_to", "policy": self.name},
        )


def _top_down(position, yaw: float) -> Pose:
    down = quat_from_axis_angle([1.0, 0.0, 0.0], np.pi)
    about_z = quat_from_axis_angle([0.0, 0.0, 1.0], yaw)
    return Pose(np.asarray(position, dtype=np.float64).reshape(3), quat_mul(about_z, down))
