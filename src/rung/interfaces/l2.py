"""L2: semantic primitives.

The model chooses what to act on and which primitive to use. The harness supplies the
geometry: object sizes, grasp candidates with their opening and clearance, and closed-loop
execution with verification. Measurements the harness could not make are reported to the
model rather than filled in.

Each turn the model sees the raw scene image, the same image with catalogued objects
marked by id, the object catalog, its memory (held object, attempts, objects out of view),
the robot time used when the horizon is known, and the result of its last call.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from rung.backends.base import (
    ImageBlock,
    Message,
    ModelBackend,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolSpec,
)
from rung.contracts.capabilities import EmbodimentCapabilities
from rung.contracts.frames import Pose, project
from rung.contracts.observation import Observation
from rung.executor.primitives import Executor, MotionIncomplete, Outcome, PrimitiveResult
from rung.perception.types import Catalog
from rung.policy import Decision, Feedback, ModelUsage, Verdict

SYSTEM = """You are operating a single robot arm with a parallel-jaw gripper on a tabletop.

You work through tools. You do not command joint angles or estimate distances: the harness
measures the scene and executes your choices, and tells you what happened.

What you are given each turn:
- The camera view, and the same view with each catalogued object marked by its id.
- The object catalog: every object the harness can see, with its measured size in
  centimetres and its height. Sizes come from one depth camera, so an object flagged as
  uncertain has been measured from a partial view.
- Your memory: what you are holding, what you have tried, what has gone out of view.
- The result of your last tool call.

How to work:
- Refer to objects by their catalog id, never by pixel position.
- `grasp` picks the best candidate the harness found. Call `propose_grasps` first if you
  want to choose one yourself, or if a grasp was rejected and you want to know why.
- A grasp is only reported as successful after the harness lifts the object and confirms
  it came with the hand. Trust that result over what the picture looks like.
- If a primitive is rejected, read the reason. It usually names a measurement, and the
  answer is a different object, a different candidate, or a different approach, not the
  same call again.
- Before deciding an object cannot be picked up, call `propose_grasps` for it. A width
  in the catalog is across the object's body, and the harness has other ways to hold
  things: turning the jaw, approaching from the side, or pinching the rim of an open
  container such as a mug. Use `give_up` only when `propose_grasps` finds no feasible
  grasp for every object the task needs.

Call exactly one tool per turn. Every call needs a short `note` saying what you see and
why you chose it; a person reads those notes to follow your reasoning."""

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
            "inspect",
            "Look closely at one catalogued object: its measured size, how much of it the "
            "camera can see, and the grasp candidates for it. Does not move the robot.",
            {
                "type": "object",
                "properties": {"object_id": {"type": "string"}, "note": NOTE_SCHEMA},
                "required": ["object_id", "note"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "propose_grasps",
            "List the grasp candidates for one object, each with the jaw opening it needs, "
            "its clearance from other objects, and the reason for any that were rejected.",
            {
                "type": "object",
                "properties": {"object_id": {"type": "string"}, "note": NOTE_SCHEMA},
                "required": ["object_id", "note"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "grasp",
            "Pick up an object. The harness approaches, closes, lifts and verifies that the "
            "object came with the hand. Omit candidate_id to use the best candidate.",
            {
                "type": "object",
                "properties": {
                    "object_id": {"type": "string"},
                    "candidate_id": {
                        "type": "string",
                        "description": "From propose_grasps. Omit to let the harness choose.",
                    },
                    "note": NOTE_SCHEMA,
                },
                "required": ["object_id", "note"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "place",
            "Put what you are holding into, onto or beside a target object. For in and on it "
            "releases above the target; for left_of, right_of, in_front_of and behind the "
            "harness finds a free spot on the table on that side and sets the object down. "
            "Directions are the robot's: left is its left, in front of is nearer the robot.",
            {
                "type": "object",
                "properties": {
                    "target_id": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["in", "on", "left_of", "right_of", "in_front_of", "behind"],
                        "default": "in",
                    },
                    "note": NOTE_SCHEMA,
                },
                "required": ["target_id", "note"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "move_to",
            "Move the gripper to an absolute position in the robot frame, in metres. Use it "
            "to get a different view or reorient. Supply quat_wxyz for a full orientation; "
            "otherwise yaw_deg selects a top-down pose.",
            {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "quat_wxyz": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Unit TCP quaternion w,x,y,z in the robot root frame. "
                        "Do not combine with yaw_deg.",
                    },
                    "yaw_deg": {
                        "type": "number",
                        "description": "Rotation of the jaw about the vertical axis.",
                    },
                    "note": NOTE_SCHEMA,
                },
                "required": ["x", "y", "z", "note"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "release",
            "Open the gripper where it is, dropping whatever is held.",
            {
                "type": "object",
                "properties": {"note": NOTE_SCHEMA},
                "required": ["note"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "observe",
            "Take a fresh look without moving. Use it after something has changed.",
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
                "properties": {"summary": {"type": "string"}, "hindsight": HINDSIGHT_SCHEMA},
                "required": ["summary"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            "give_up",
            "The task cannot be done with this rig. Say why. The trial ends.",
            {
                "type": "object",
                "properties": {"reason": {"type": "string"}, "hindsight": HINDSIGHT_SCHEMA},
                "required": ["reason"],
                "additionalProperties": False,
            },
        ),
    ]


HINDSIGHT_SCHEMA = {
    "type": "string",
    "description": (
        "Optional. One or two sentences for whoever runs this task next: what worked, "
        "what got in the way, what you would do differently."
    ),
}


@dataclass
class L2Config:
    effort: str | None = None
    max_tokens: int = 2048
    send_marked_image: bool = True
    send_wrist: bool = True
    image_scale: float = 1.0
    history_turns: int = 0
    """Previous turns to resend. Zero rebuilds the prompt each turn from the catalog and
    memory, which is cheaper and keeps state owned by the harness."""
    horizon_s: float | None = None
    """The episode's simulated horizon in seconds, if known. Shown to the model with the
    time used, since a single primitive can consume a large share of it."""


class L2Policy:
    """A model driving the primitives."""

    interface = "L2"

    def __init__(
        self,
        backend: ModelBackend,
        executor: Executor,
        config: L2Config | None = None,
    ):
        self.backend = backend
        self.ex = executor
        self.cfg = config or L2Config()
        self.instruction = ""
        self.caps: EmbodimentCapabilities | None = None
        self.last_result: PrimitiveResult | None = None
        self.last_error: str | None = None
        self.turn = 0
        self.prompts: list[str] = []
        self.pending_usage = ModelUsage()
        self.t0: float | None = None
        self.last_snapshot: dict | None = None

    @property
    def name(self) -> str:
        return f"l2/{self.backend.model}"

    def attach_trace(self, trace, episode_id: str) -> None:
        """The loop calls this so the executor's own motions land in the same trace."""
        self.ex.trace = trace
        self.ex.episode = episode_id

    def reset(self, instruction: str, capabilities: EmbodimentCapabilities) -> None:
        self.instruction = instruction
        self.caps = capabilities
        self.ex.instruction = instruction
        self.ex.memory.reset()
        self.ex.unverified_hold = False
        self.ex.hold_threshold = 0.95
        self.ex.jaw_informative = True
        self.ex.catalog = None
        self.last_result = None
        self.last_error = None
        self.turn = 0
        self.prompts = []
        self.pending_usage = ModelUsage()
        self.ex.pending_usage = ModelUsage()
        self.t0 = None

    def act(self, observation: Observation, feedback: Feedback) -> Decision:
        self.turn += 1
        if self.t0 is None:
            self.t0 = float(observation.t)
        cached = self.ex.catalog
        if cached is not None and abs(float(cached.t) - float(observation.t)) < 1e-9:
            # No simulated time has passed since the last look (a primitive ends with one),
            # so the scene is unchanged. Reusing the catalog avoids seconds of perception
            # latency, during which a real-time robot stands still.
            catalog = cached
            self.ex.log(
                "perception",
                reused=True,
                t_catalog=float(cached.t),
                note="no simulated time has passed since the last look; reused it",
            )
        else:
            catalog = self.ex.refresh()
        self.last_snapshot = _catalog_snapshot(catalog)
        request = self.build_request(observation, catalog)
        self.prompts.append(_render_prompt(request))
        self.ex.log(
            "model_request",
            model=self.backend.model,
            turn=self.turn,
            prompt=self.prompts[-1],
            tools=[t.name for t in request.tools],
            images=[
                b.label for m in request.messages for b in m.content if isinstance(b, ImageBlock)
            ],
        )
        try:
            response = self.backend.complete(request)
        except Exception as exc:
            self.ex.log(
                "model_response",
                model=self.backend.model,
                turn=self.turn,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self.pending_usage.add(response.usage)
        import dataclasses as _dc

        self.ex.log(
            "model_response",
            model=self.backend.model,
            turn=self.turn,
            text=response.text or None,
            thoughts=response.thoughts or None,
            tool_calls=[{"name": c.name, "arguments": c.arguments} for c in response.tool_calls],
            stop_reason=response.stop_reason,
            refusal=response.refusal,
            usage=_dc.asdict(response.usage),
        )

        if response.refusal:
            return Decision(
                "refusal",
                f"The model declined: {response.refusal}",
                verdict=Verdict.GIVE_UP,
                reason=response.refusal,
                payload={"usage": response.usage},
            )

        call = response.call
        if call is None:
            self.last_error = "no tool call in the response. Reply with exactly one tool call."
            raise ValueError(self.last_error)

        try:
            return self.dispatch(call, response)
        except ValueError as exc:
            # The loop retries the turn. The prompt is built from policy state, so the error
            # must be stored here for the model to see it.
            self.last_error = str(exc)
            raise

    def build_request(self, observation: Observation, catalog: Catalog) -> ModelRequest:
        blocks: list[Any] = [TextBlock(f"Task: {self.instruction}")]

        blocks.append(TextBlock("Camera view:"))
        blocks.append(_image_block(observation.scene.rgb, "scene", self.cfg.image_scale))
        if self.cfg.send_marked_image and catalog.objects:
            marked = mark_objects(observation, catalog)
            blocks.append(TextBlock("The same view with each object marked by its id:"))
            blocks.append(_image_block(marked, "scene_marked", self.cfg.image_scale))
        if self.cfg.send_wrist and observation.wrist is not None:
            blocks.append(TextBlock("Wrist view:"))
            blocks.append(_image_block(observation.wrist.rgb, "wrist", self.cfg.image_scale))

        jaw = self.caps.gripper.max_opening_m if self.caps else 0.085
        blocks.append(TextBlock(f"Objects the harness can see (jaw opens to {jaw * 100:.1f} cm):"))
        blocks.append(TextBlock(catalog.render(jaw)))
        if catalog.failures:
            blocks.append(TextBlock("Perception could not use: " + "; ".join(catalog.failures)))
        blocks.append(TextBlock(self.ex.memory.render()))
        used = float(observation.t) - (self.t0 if self.t0 is not None else float(observation.t))
        if self.cfg.horizon_s:
            blocks.append(
                TextBlock(
                    f"Robot time used: {used:.0f} s of {self.cfg.horizon_s:.0f} s. Each primitive "
                    "takes several seconds of robot time; the trial ends when the budget does."
                )
            )

        if self.last_error:
            blocks.append(TextBlock(f"Your last reply could not be used: {self.last_error}"))
            self.last_error = None
        elif self.last_result is not None:
            blocks.append(TextBlock("Result of your last call: " + self.last_result.render()))
            for key in ("rendered", "grasps", "description"):
                extra = self.last_result.detail.get(key)
                if extra:
                    blocks.append(TextBlock(str(extra)))

        blocks.append(TextBlock("Choose one tool."))
        return ModelRequest(
            system=SYSTEM,
            messages=[Message.user(*blocks)],
            tools=tool_specs(has_wrist=observation.wrist is not None),
            effort=self.cfg.effort,
            max_tokens=self.cfg.max_tokens,
        )

    def dispatch(self, call: ToolCall, response: ModelResponse) -> Decision:
        args = call.arguments
        note = str(args.get("note") or args.get("summary") or args.get("reason") or "")
        payload: dict[str, Any] = {"usage": response.usage, "tool": call.name, "arguments": args}
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

        known = {tool.name for tool in tool_specs(True)}
        if call.name not in known:
            raise ValueError(f"unknown tool {call.name!r}")
        if call.name == "move_to":
            _pose_from(args)  # Reject malformed geometry before any execution.
        _require_arguments(call.name, args)
        return Decision(kind=call.name, note=note, payload=payload)

    def drain_usage(self) -> ModelUsage:
        usage = self.pending_usage
        usage.add(self.ex.pending_usage)
        self.pending_usage = ModelUsage()
        self.ex.pending_usage = ModelUsage()
        return usage

    def execute_decision(self, decision: Decision, execute_target) -> None:
        """Called by the episode loop, after the decision and usage have been recorded."""
        if decision.verdict is not Verdict.CONTINUE:
            return
        call = ToolCall(decision.payload["tool"], decision.payload["arguments"])
        self.ex.motion_handler = execute_target
        try:
            try:
                result = self._run(call)
            except MotionIncomplete as exc:
                result = PrimitiveResult(
                    call.name,
                    Outcome.BLOCKED,
                    reason=str(exc),
                    detail={"execution": exc.outcome.value},
                )
            except (KeyError, ValueError, TypeError) as exc:
                # Malformed call, or a primitive that could not interpret the scene. Reported
                # as a rejection so the model can choose differently next turn.
                result = PrimitiveResult(
                    call.name,
                    Outcome.REJECTED,
                    reason=f"that call could not be carried out: {exc}",
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
            result = self.ex.emit_primitive(result)
            self.last_result = result
            decision.payload.update(result=result.render(), outcome=result.outcome.value)
            decision.reason = result.reason
        finally:
            self.ex.motion_handler = None

    def _run(self, call: ToolCall) -> PrimitiveResult:
        args = call.arguments
        name = call.name
        if name == "inspect":
            return self.ex.inspect(str(args["object_id"]))
        if name == "propose_grasps":
            return self.ex.propose_grasps(str(args["object_id"]))
        if name == "grasp":
            return self.ex.grasp(str(args["object_id"]), args.get("candidate_id"))
        if name == "place":
            return self.ex.place(str(args["target_id"]), str(args.get("relation", "in")))
        if name == "release":
            return self.ex.release()
        if name == "observe":
            self.ex.refresh()
            return PrimitiveResult("observe", Outcome.OBSERVED, None, reason="took a fresh look")
        if name == "move_to":
            pose = _pose_from(args)
            return self.ex.move_to(pose)
        known = ", ".join(t.name for t in tool_specs(True))
        raise ValueError(f"unknown tool {name!r}; available: {known}")


REQUIRED_ARGUMENTS = {
    "inspect": ("object_id",),
    "propose_grasps": ("object_id",),
    "grasp": ("object_id",),
    "place": ("target_id",),
    "move_to": ("x", "y", "z"),
}


def _require_arguments(name: str, args: dict[str, Any]) -> None:
    """Check required tool arguments while the turn can still be retried.

    Runs inside ``act``, so a missing argument is fed back to the model for another turn.
    Raising during execution would instead end the episode.
    """
    missing = [key for key in REQUIRED_ARGUMENTS.get(name, ()) if args.get(key) is None]
    if missing:
        raise ValueError(
            f"{name} needs {', '.join(missing)}; you sent "
            f"{sorted(k for k in args if k != 'note') or 'nothing'}"
        )


def _pose_from(args: dict[str, Any]) -> Pose:
    position = np.array([args["x"], args["y"], args["z"]], dtype=float)
    if not np.all(np.isfinite(position)):
        raise ValueError("TCP position must be finite")
    if "quat_wxyz" in args:
        if "yaw_deg" in args:
            raise ValueError("choose quat_wxyz or yaw_deg, not both")
        q = np.asarray(args["quat_wxyz"], dtype=float)
        if q.shape != (4,) or not np.all(np.isfinite(q)) or not np.isclose(np.linalg.norm(q), 1):
            raise ValueError("quat_wxyz must be a finite unit quaternion")
        return Pose(position, q)
    from rung.grasp.analytic import top_down_pose

    yaw = float(np.radians(float(args.get("yaw_deg", 0.0))))
    if not np.isfinite(yaw):
        raise ValueError("yaw_deg must be finite")
    return top_down_pose(position, yaw)


def _catalog_snapshot(catalog: Catalog) -> dict:
    """The catalog as the model saw it, plus the detection boxes it was measured from."""
    objects = []
    for o in catalog.objects:
        det = o.detection
        objects.append(
            {
                "object_id": o.object_id,
                "label": o.label,
                "box_px": list(det.box) if det is not None else None,
                "centroid": o.centroid,
                "grasp_centre": o.grasp_centre,
                "top_z": o.top_z,
                "bottom_z": o.bottom_z,
                "length": o.length,
                "width": o.width,
                "height": o.height,
                "grasp_width": o.grasp_width,
                "grasp_length": o.grasp_length,
                "n_points": o.n_points,
                "visible_fraction": o.visible_fraction,
                "uncertain": o.uncertain,
                "uncertainty_reason": o.uncertainty_reason,
            }
        )
    return {
        "t": catalog.t,
        "source": catalog.source,
        "version": catalog.version,
        "failures": list(catalog.failures),
        "objects": objects,
    }


def mark_objects(observation: Observation, catalog: Catalog) -> np.ndarray:
    """Draw each object's id at its projected position.

    The unmarked image is always sent as well: a label can sit on top of the thing it
    names, and a small object can disappear underneath its own mark.
    """
    from PIL import Image, ImageDraw

    cam = observation.scene
    img = Image.fromarray(np.asarray(cam.rgb, dtype=np.uint8)).convert("RGB")
    if cam.K is None or cam.pose is None:
        return np.asarray(img)
    draw = ImageDraw.Draw(img, "RGBA")
    for obj in catalog.objects:
        uv, z = project(np.asarray(obj.grasp_centre)[None], cam.K, cam.pose)
        if z[0] <= 0:
            continue
        u, v = float(uv[0, 0]), float(uv[0, 1])
        if not (0 <= u < cam.width and 0 <= v < cam.height):
            continue
        draw.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(255, 90, 30, 255), width=2)
        label = obj.object_id
        tw = 6 * len(label) + 6
        tx, ty = min(max(0, u + 7), cam.width - tw), max(0, v - 14)
        draw.rectangle([tx, ty, tx + tw, ty + 13], fill=(20, 20, 20, 190))
        draw.text((tx + 3, ty + 2), label, fill=(255, 255, 255, 255))
    return np.asarray(img)


def _image_block(rgb: np.ndarray, label: str, scale: float = 1.0) -> ImageBlock:
    """Encode an image for the model as JPEG, quality 92.

    JPEG encodes in tens of milliseconds where PNG takes hundreds per 720p frame, time a
    real-time robot spends standing still. Resolution, and so image token cost, is
    unchanged. The recorder keeps lossless copies.
    """
    from PIL import Image

    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return ImageBlock(buf.getvalue(), "image/jpeg", label)


def _render_prompt(request: ModelRequest) -> str:
    """The exact text the model saw, for the recorder. Images appear as placeholders."""
    parts = [f"[system]\n{request.system}"]
    for message in request.messages:
        parts.append(f"[{message.role}]")
        for block in message.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            else:
                parts.append(f"<image {block.label} {len(block.data)} bytes>")
    parts.append("[tools] " + ", ".join(t.name for t in request.tools))
    if request.effort:
        parts.append(f"[effort] {request.effort}")
    return "\n".join(parts)


def schema_json() -> str:
    """The tool surface as JSON, for docs and for provenance hashing."""
    return json.dumps(
        [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in tool_specs(True)
        ],
        indent=2,
    )
