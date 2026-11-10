"""direct-v0: one native actor, root-frame XYZ/axis-angle and independent gripper.

Development profile. No privileged object poses, automatic grasp planner, or success
heuristic is exposed. Native benchmark grading remains outside this policy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace

import numpy as np

from rung.backends.base import TextBlock, ToolSpec
from rung.backends.native_session import NativeSession
from rung.contracts.action import ExecutionOutcome, GripperAction, GripperCommand, TCPTarget
from rung.contracts.frames import Pose, quat_angle_between, quat_from_axis_angle, quat_mul
from rung.interfaces.l2 import _image_block
from rung.policy import Decision, ModelUsage, Verdict

SYSTEM = """Control one robot arm with a parallel-jaw gripper using the actual images
and measured TCP state. Positions and translation deltas are in robot-root metres;
orientation is quaternion wxyz, child-to-root. move_relative rotation_vector_rad is
axis times angle in radians, about root axes; it left-multiplies the current rotation.
Root +x points away from the robot base across the table, +y to the robot's left,
and +z upward. Image left/right are camera dependent, not root x/y.
Positive rotations follow the right-hand rule.
Translation/rotation bounds apply to vector magnitude, not each axis separately.
TCP is the fingertip midpoint, local +y closes the jaws and local +z is approach.
Finger pads extend back towards the wrist along local -z; TCP is not the pad centre.
The table height and workspace are supplied by the embodiment; do not assume z=0.
Choose exactly one tool per turn. Motion holds the gripper; use set_gripper separately.
Use execution receipts to distinguish planned from achieved motion. Gripper travel
alone does not prove grasping or absence of an object. Report a brief observable note.
done is your belief; a separate native grader decides benchmark success."""


SYSTEM_BIMANUAL = """Control a rig of several robot arms, each with a parallel-jaw gripper,
using the actual images and the measured state of every arm. Every arm is addressed by
name with the arm argument; an arm you do not command holds its pose and gripper. All
positions, for every arm, are in ONE shared rig-root frame, metres; root_frame in the
state says which way its axes point. Orientation is quaternion wxyz, child-to-root.
move_relative rotation_vector_rad is axis times angle in radians about root axes; it
left-multiplies that arm's current rotation. Image left/right are camera dependent; each
wrist camera rides the arm it is named after.
Positive rotations follow the right-hand rule.
Translation/rotation bounds apply to vector magnitude, not each axis separately.
Each TCP is that gripper's fingertip midpoint, local +y closes the jaws and local +z is
approach. Each arm has its own reachable box; the arms can collide with each other and
nothing checks that for you, so keep them apart unless the task needs them together.
The table height is supplied by the embodiment; do not assume z=0.
Choose exactly one tool per turn. Motion holds the gripper; use set_gripper separately.
move_both moves several arms together in one motion, for holding one object with two
hands or handing over; otherwise move one arm at a time.
Use execution receipts to distinguish planned from achieved motion. Gripper travel
alone does not prove grasping or absence of an object. Report a brief observable note.
done is your belief; a separate native grader decides benchmark success."""


SEQUENCE_TOOLS = ("move_relative", "move_to", "move_relative_tcp", "set_gripper", "wait")
"""Tools a guarded sequence may contain. Geometry queries, observe, done and give_up stay
single-turn: a sequence is motion the model already decided on, nothing else."""

STEP_FIELDS = {
    "move_relative": ("delta_xyz_m", "rotation_vector_rad"),
    "move_relative_tcp": ("delta_xyz_m", "rotation_vector_rad"),
    "move_to": ("xyz_m", "quat_wxyz"),
    "set_gripper": ("action",),
    "wait": ("seconds",),
}


TABLE_EDGE_WARN_M = 0.06


def table_edge_report(tcp_pos, workspace):
    """Signed margin from the TCP to the nearest declared table edge, with a warning."""
    margin = workspace.table_margin(tcp_pos)
    report = {"tcp_margin_m": None if margin is None else round(margin, 3)}
    if margin is not None and margin < 0:
        report["warning"] = (
            f"TCP is {-margin * 100:.0f} cm beyond the table edge; an object "
            "released or carried here falls to the floor"
        )
    elif margin is not None and margin < TABLE_EDGE_WARN_M:
        report["warning"] = (
            f"TCP is {margin * 100:.0f} cm from the table edge; a long object "
            "or a release here can leave the table"
        )
    return report


def tools(
    motion_tools="relative",
    table_projection=False,
    projection_cameras=None,
    stereo_triangulation=False,
    tcp_frame_motions=False,
    depth_probe=False,
    depth_cameras=None,
    action_sequence=False,
    max_sequence_steps=6,
    terminal_tools=True,
    done_awaits_confirmation=False,
    look_tool=False,
    look_cameras=None,
    look_max_steps=12,
    arms=None,
):
    """Tool schemas. ``arms`` (arm names, primary first) switches on per-arm addressing:
    an ``arm`` argument on motion and gripper tools and sequence steps, ``move_both``, and
    a two-camera ``triangulate_point``. None leaves the single-arm schemas untouched."""
    vector = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    arm_names = list(arms or ())
    arm_tools = ("move_relative", "move_to", "move_relative_tcp", "set_gripper")

    def spec(name, description, properties):
        if arm_names and name in arm_tools:
            properties = {"arm": {"type": "string", "enum": arm_names}, **properties}
        return ToolSpec(
            name,
            description,
            {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        )

    note = {"type": "string"}
    movements = []
    if action_sequence:
        step_tools = [t for t in SEQUENCE_TOOLS if t != "move_relative_tcp" or tcp_frame_motions]
        if motion_tools == "relative":
            step_tools.remove("move_to")
        elif motion_tools == "absolute":
            step_tools.remove("move_relative")
        step = {
            "type": "object",
            "additionalProperties": False,
            "required": ["tool"],
            "properties": {
                "tool": {"type": "string", "enum": step_tools},
                "delta_xyz_m": vector,
                "rotation_vector_rad": vector,
                "xyz_m": vector,
                "quat_wxyz": {**vector, "minItems": 4, "maxItems": 4},
                "action": {"type": "string", "enum": ["open", "close"]},
                "seconds": {"type": "number", "minimum": 0.05, "maximum": 3},
            },
        }
        if arm_names:
            step["properties"]["arm"] = {"type": "string", "enum": arm_names}
        movements.append(
            spec(
                "execute_sequence",
                f"Execute up to {max_sequence_steps} motion/gripper/wait steps in order in ONE "
                "turn (a waypoint chunk), each with the same bounds as the single tools, each "
                "receiving its own execution receipt. The harness STOPS the sequence early and "
                "returns to you when a step does not reach its target (blocked or expired) or "
                "when a close "
                "step ends closed on nothing; later steps are then not executed. Relative "
                "steps are relative to the measured pose after the previous step. Use it for "
                "motion you have already decided on (approach, descend, close, lift); use "
                "single tools when the next step depends on what you will see.",
                {
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": max_sequence_steps,
                        "items": step,
                    },
                    "note": note,
                },
            )
        )
    if arm_names and len(arm_names) > 1:
        relative = motion_tools in ("relative", "both")
        goal = (
            {"delta_xyz_m": vector, "rotation_vector_rad": vector}
            if relative
            else {"xyz_m": vector, "quat_wxyz": {**vector, "minItems": 4, "maxItems": 4}}
        )
        movements.append(
            spec(
                "move_both",
                "Move every arm together in ONE motion: all start and stop together, each "
                "with the same bounds as the single-arm tool"
                + (
                    " (relative to its own measured TCP, root axes)."
                    if relative
                    else " (absolute TCP pose in the root frame)."
                )
                + " Grippers hold. Use for two-"
                "handed holds and handovers; one arm's failure is reported per arm.",
                {
                    **{
                        a: {
                            "type": "object",
                            "properties": goal,
                            "required": list(goal),
                            "additionalProperties": False,
                        }
                        for a in arm_names
                    },
                    "note": note,
                },
            )
        )
    if motion_tools in ("relative", "both"):
        movements.append(
            spec(
                "move_relative",
                "Bounded relative TCP motion; gripper unchanged.",
                {"delta_xyz_m": vector, "rotation_vector_rad": vector, "note": note},
            )
        )
    if motion_tools in ("absolute", "both"):
        movements.append(
            spec(
                "move_to",
                "Absolute TCP position and unit quaternion wxyz in root "
                "frame. The same displacement/rotation bounds apply; hold gripper.",
                {
                    "xyz_m": vector,
                    "quat_wxyz": {**vector, "minItems": 4, "maxItems": 4},
                    "note": note,
                },
            )
        )
    if tcp_frame_motions:
        movements.append(
            spec(
                "move_relative_tcp",
                "Relative motion in the CURRENT measured "
                "TCP axes. Translate and rotate about the fingertip midpoint; "
                "rotation right-multiplies the current orientation. Holds gripper.",
                {"delta_xyz_m": vector, "rotation_vector_rad": vector, "note": note},
            )
        )
    projection_cameras = ["scene", "wrist"] if projection_cameras is None else projection_cameras
    if table_projection and projection_cameras:
        movements.append(
            spec(
                "project_table",
                "Select an image pixel and map its ray to the "
                "declared table plane. Geometry only, no motion; not object height.",
                {
                    "camera": {"type": "string", "enum": projection_cameras},
                    "u": {"type": "number", "minimum": 0, "maximum": 1},
                    "v": {"type": "number", "minimum": 0, "maximum": 1},
                    "capture_time_s": {"type": "number"},
                    "note": note,
                },
            )
        )
    depth_cameras = [] if depth_cameras is None else depth_cameras
    if depth_probe and depth_cameras:
        movements.append(
            spec(
                "probe_point",
                "Select an image pixel on a visible surface and "
                "read the rig's calibrated depth there: returns the 3D surface "
                "point in root metres, its height above the declared table, "
                "and validity flags (no depth, depth edge). Geometry only, no "
                "motion; not an object centre or a grasp pose.",
                {
                    "camera": {"type": "string", "enum": depth_cameras},
                    "u": {"type": "number", "minimum": 0, "maximum": 1},
                    "v": {"type": "number", "minimum": 0, "maximum": 1},
                    "capture_time_s": {"type": "number"},
                    "note": note,
                },
            )
        )
    if stereo_triangulation and arm_names and len(projection_cameras) >= 2:
        uv = {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1},
            "minItems": 2,
            "maxItems": 2,
        }
        camera = {"type": "string", "enum": projection_cameras}
        movements.append(
            spec(
                "triangulate_point",
                "Select the same visible physical point "
                "in two different calibrated RGB views. Calibrated rays yield "
                "a 3D point and ray mismatch; no depth lookup or object "
                "detection, no motion.",
                {
                    "first_camera": camera,
                    "first_uv": uv,
                    "second_camera": camera,
                    "second_uv": uv,
                    "capture_time_s": {"type": "number"},
                    "note": note,
                },
            )
        )
    elif stereo_triangulation and set(projection_cameras) == {"scene", "wrist"}:
        uv = {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1},
            "minItems": 2,
            "maxItems": 2,
        }
        movements.append(
            spec(
                "triangulate_point",
                "Select the same visible physical point "
                "in scene and wrist RGB. Calibrated rays yield a 3D point and "
                "ray mismatch; no depth lookup or object detection, no motion.",
                {
                    "scene_uv": uv,
                    "wrist_uv": uv,
                    "capture_time_s": {"type": "number"},
                    "note": note,
                },
            )
        )
    if look_tool:
        movements.append(look_spec(look_cameras or [], look_max_steps))
    return [
        *movements,
        spec(
            "set_gripper",
            "Open or close at the current TCP pose.",
            {"action": {"type": "string", "enum": ["open", "close"]}, "note": note},
        ),
        spec("observe", "Request fresh camera views without motion.", {"note": note}),
        spec(
            "wait",
            "Hold measured TCP and current gripper command while physics advances; "
            "use to inspect settling/delayed motion. Counts against simulated horizon.",
            {"seconds": {"type": "number", "minimum": 0.05, "maximum": 3}, "note": note},
        ),
        *(
            [
                spec(
                    "done",
                    (
                        "Declare the task complete and hold still. The trial ends only if "
                        "the independent native grader confirms success; otherwise you "
                        "receive the next observation and must continue."
                        if done_awaits_confirmation
                        else "End trial; independent native grader determines success."
                    ),
                    {"note": note},
                ),
                spec("give_up", "End trial because you cannot continue.", {"note": note}),
            ]
            if terminal_tools
            else []
        ),
    ]


def look_spec(pixel_cameras, max_steps):
    """The opt-in ``look`` tool. ``target_xyz_m`` or ``target_pixel`` (exactly one) names the
    point; the pixel form is offered only when a calibrated camera can resolve it."""
    from rung.interfaces import look as geometry

    vector = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    properties = {"target_xyz_m": {**vector, "description": "Point to look at, root metres."}}
    if pixel_cameras:
        properties["target_pixel"] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["camera", "u", "v", "capture_time_s"],
            "description": "Alternative to target_xyz_m: a pixel in a current calibrated "
            "image, resolved to a root point through public calibration.",
            "properties": {
                "camera": {"type": "string", "enum": list(pixel_cameras)},
                "u": {"type": "number", "minimum": 0, "maximum": 1},
                "v": {"type": "number", "minimum": 0, "maximum": 1},
                "capture_time_s": {"type": "number"},
            },
        }
    lo_d, hi_d = geometry.DISTANCE_RANGE_M
    lo_e, hi_e = geometry.ELEVATION_RANGE_DEG
    d, e, a = (
        f"{v:g}"
        for v in (
            geometry.DEFAULT_DISTANCE_M,
            geometry.DEFAULT_ELEVATION_DEG,
            geometry.DEFAULT_AZIMUTH_DEG,
        )
    )
    side = f"{geometry.SIDE_VIEW_BELOW_DEG:g}"
    properties.update(
        {
            "distance_m": {
                "type": "number",
                "minimum": lo_d,
                "maximum": hi_d,
                "description": f"Camera-to-target distance. Default {d}.",
            },
            "elevation_deg": {
                "type": "number",
                "minimum": lo_e,
                "maximum": hi_e,
                "description": f"90 straight down. Default {e}. Below {side} is a "
                "side view that may not be reachable near the "
                "robot base.",
            },
            "azimuth_deg": {
                "type": "number",
                "minimum": -180,
                "maximum": 180,
                "description": f"180 from the robot's side. Default {a}.",
            },
            "return_to_previous_pose": {
                "type": "boolean",
                "description": "Retrace back after the capture. Default false.",
            },
            "note": {"type": "string"},
        }
    )
    required = ["note"]
    return ToolSpec(
        "look",
        "Re-localise: move the wrist camera to a viewpoint looking straight at a point and "
        "capture fresh views, without planning the motion yourself. The camera goes to "
        "target + distance_m * [cos(el)cos(az), cos(el)sin(az), sin(el)] in root axes: "
        "elevation_deg 90 looks straight down, azimuth_deg 180 views from the robot's side, "
        f"0 from the far side, 90 from the robot's left. If unsure, use distance_m {d}, "
        f"elevation_deg {e}, azimuth_deg {a}; these are the defaults for omitted fields. "
        f"Elevations below {side} are side views that may not be reachable near the robot "
        "base. The harness rises, rotates in place at a safe height, translates at that "
        "height, then descends, in bounded steps with the usual speed limits; it stops at "
        "the first step that is blocked or expires and captures there. It starts inside the "
        "workspace, or within one bounded step of it. "
        "With return_to_previous_pose true (default false) the arm then retraces the path "
        "back and the look-pose wrist view is shown as LOOK VIEW in the next observation; "
        f"otherwise the arm stays at the viewpoint. Gripper unchanged. At most {max_steps} "
        "steps; costs simulated time like any motion.",
        {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )


@dataclass
class DirectConfig:
    max_translation_m: float = 0.10
    max_rotation_rad: float = 0.50
    max_speed_mps: float = 0.10
    max_ang_speed_rps: float = 0.50
    timeout_s: float = 10
    image_scale: float = 1
    motion_tools: str = "relative"
    public_overlay: bool = False
    table_projection: bool = False
    stereo_triangulation: bool = False
    tcp_marker: bool = False
    spatial_frame_cues: bool = False
    tcp_frame_motions: bool = False
    geometry_feedback: bool = False
    depth_probe: bool = False
    """Offer ``probe_point``: the rig's depth at a model-selected pixel. Declared RGB-D
    input; label runs that use it. Off for the RGB-only reference."""
    action_sequence: bool = False
    """Offer ``execute_sequence``: a guarded list of motion steps executed in one turn,
    stopped at the first step that does not reach or that closes on nothing."""
    max_sequence_steps: int = 6
    sequence_grasp_guard: bool = False
    """Also stop a sequence when a carried object is lost: after a close that ended on
    something (jaw travel between 0.2 and 0.95), or when the sequence starts holding
    something, a later step whose end travel rises by more than 0.10 means the jaw closed
    further on nothing. The 2026-10-23 clips show Gemini carrying on for about 7 s after a
    drop; the jaw travel is public, so this is a bridge rule, not model input."""
    table_extent: bool = False
    """Tell the model the declared table-top footprint (``table_extent_m`` in the state)
    and add a ``table_edge`` receipt with the TCP's signed margin to the nearest edge,
    warning within 6 cm of or beyond it. Seven of 32 counted Gemini failures in the
    2026-10-23 subset were targets released or knocked off the table edge."""
    done_awaits_confirmation: bool = False
    """``done`` no longer ends the episode on the policy's belief. The policy holds still
    for ``done_confirmation_wait_s`` of simulated time; if the native grader confirms
    success meanwhile the episode ends as before, otherwise the next observation carries a
    ``done_not_confirmed`` receipt and the policy continues. ``give_up`` stays terminal.
    Screen 1 (2026-10-24): both counted ``PickUpBluePitcherTask`` failures under the speed
    arm were a ``done`` declared with the pitcher not in hand. The information given back
    (the episode did not end) is what the no-terminal-tools arm already has."""
    done_confirmation_wait_s: float = 3.0
    terminal_tools: bool = True
    """Offer ``done`` and ``give_up``. When False the policy cannot end the episode: it
    ends when the native evaluator confirms success or the native horizon runs out, as in
    the Astranomy protocol. The 2026-10-22 subset had 16 failures ending on a ``done``
    at a median 46 percent of the horizon."""
    camera_geometry_prompt: bool = False
    """Write camera poses, intrinsics and the wrist mount into the prompt text every
    call. Astranomy's largest single effect for a direct policy."""
    recent_frames: bool = False
    """Also show the frames the embodiment buffered shortly before the current one
    (``observation.extra["recent_frames"]``), so the model sees the last motion."""
    extra_views: bool = False
    """Also show additional calibrated views the embodiment supplies
    (``observation.extra["views"]``), such as a second shoulder camera."""
    look_tool: bool = False
    """Offer ``look`` (rung.interfaces.look): move the wrist camera to a model-chosen
    viewpoint on a point and capture fresh views, as one guarded multi-step motion,
    optionally returning to the previous pose. For the target-lost failure class (6 of
    13 counted Gemini Screen 2b failures). Off: tools, prompt and receipts are unchanged."""
    look_max_steps: int = 12
    bimanual: bool = False
    """Address every arm the embodiment declares (``capabilities.arms``): per-arm state,
    an ``arm`` argument on motion and gripper tools, ``move_both`` and every wrist view.
    Off keeps the single-arm profile byte for byte, also on a multi-arm rig, where it
    then drives the primary arm only."""

    def __post_init__(self):
        if self.public_overlay and self.tcp_marker:
            raise ValueError("choose one rig image presentation profile")
        if self.look_tool and self.bimanual:
            raise ValueError(
                "look_tool is single-arm only; it has not been built for per-arm addressing"
            )
        if self.motion_tools not in ("relative", "absolute", "both"):
            raise ValueError("invalid motion tool profile")
        if self.bimanual and (self.sequence_grasp_guard or self.tcp_marker or self.public_overlay):
            raise ValueError("bimanual has no per-arm grasp guard, TCP marker or overlay yet")
        steps = self.max_sequence_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 16:
            raise ValueError("max_sequence_steps must be an integer in 1..16")
        steps = self.look_max_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 32:
            raise ValueError("look_max_steps must be an integer in 1..32")
        for name in (
            "max_translation_m",
            "max_rotation_rad",
            "max_speed_mps",
            "max_ang_speed_rps",
            "timeout_s",
            "image_scale",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class DirectPolicy:
    interface = "direct-v0"

    def __init__(
        self,
        session: NativeSession,
        config: DirectConfig | None = None,
        budgets: dict | None = None,
    ):
        self.session, self.cfg = session, config or DirectConfig()
        self.budgets = budgets
        self.pending_usage = ModelUsage()
        self._accepted_call_id = None
        self._geometry_receipt = None
        self._sim_start = None
        self._sequence_targets = None
        self._sequence_result = None
        self._clear_look()
        self._look_snapshot = None

    @property
    def name(self):
        return f"direct-v0/{self.session.model}"

    @property
    def model_calls_attempted(self):
        return self.session.turn

    def reset(self, instruction, capabilities):
        if self.cfg.bimanual and len(capabilities.arms) < 2:
            raise ValueError("bimanual profile needs an embodiment that declares two arms")
        self.instruction, self.caps = instruction, capabilities
        self.session.reset()
        self.pending_usage = ModelUsage()
        self._accepted_call_id = None
        self._geometry_receipt = None
        self._sim_start = None
        self._sequence_targets = None
        self._sequence_result = None
        self._done_pending = None
        self._clear_look()
        self._look_snapshot = None

    def drain_usage(self):
        usage, self.pending_usage = self.pending_usage, ModelUsage()
        return usage

    def act(self, observation, feedback):
        if self._sim_start is None:
            self._sim_start = observation.t
        if self.cfg.table_extent and self.caps.workspace.table_extent is not None:
            feedback.structured["table_edge"] = table_edge_report(
                observation.tcp.pos, self.caps.workspace
            )
        if getattr(self, "_done_pending", None) is not None:
            waited = observation.t - self._done_pending
            feedback.structured["done_not_confirmed"] = (
                f"you declared done {waited:.1f} s of simulated time ago; the independent "
                "grader has not confirmed success, so the task is not complete. Re-check "
                "every object the instruction names in the current views and continue."
            )
            self._done_pending = None
        if self.session.pending:
            self.session.resolve(
                {
                    c.id: {
                        "execution_feedback": feedback.structured,
                        "geometry_result": self._geometry_receipt,
                        "sequence_result": self._sequence_result,
                        **({"look_result": self._look_result} if self.cfg.look_tool else {}),
                        "measured_tcp_xyz_m": observation.tcp.pos.tolist(),
                        "measured_tcp_quat_wxyz": observation.tcp.quat.tolist(),
                        "gripper_travel": observation.gripper.travel,
                        **(
                            {"measured_arms": self._arm_state(observation)}
                            if self.cfg.bimanual
                            else {}
                        ),
                    }
                    for c in self.session.pending
                }
            )
        self._accepted_call_id = None
        self._geometry_receipt = None
        self._sequence_targets = None
        self._sequence_result = None
        self._clear_look()
        state = {
            "task": self.instruction,
            "sim_time_s": observation.t,
            "tcp_xyz_m": observation.tcp.pos.tolist(),
            "tcp_quat_wxyz": observation.tcp.quat.tolist(),
            "tcp_axes_in_root": {
                name: observation.tcp.rot[:, index].tolist()
                for index, name in enumerate(("x", "y", "z"))
            },
            "gripper_travel": observation.gripper.travel,
            "gripper_opening_m": observation.gripper.opening_m,
            "gripper_geometry_m": {
                "max_opening": self.caps.gripper.max_opening_m,
                "finger_length": self.caps.gripper.finger_length_m,
                "pad_width": self.caps.gripper.pad_width_m,
            },
            "workspace_min_m": np.asarray(self.caps.workspace.min_xyz).tolist(),
            "workspace_max_m": np.asarray(self.caps.workspace.max_xyz).tolist(),
            "table_z_m": self.caps.workspace.table_z,
            **(
                {
                    "table_extent_m": {
                        "x": list(self.caps.workspace.table_extent[:2]),
                        "y": list(self.caps.workspace.table_extent[2:]),
                    }
                }
                if self.cfg.table_extent and self.caps.workspace.table_extent
                else {}
            ),
            "motion_limits": {
                k: getattr(self.cfg, k)
                for k in (
                    "max_translation_m",
                    "max_rotation_rad",
                    "max_speed_mps",
                    "max_ang_speed_rps",
                    "timeout_s",
                    "image_scale",
                )
            },
        }
        if self.cfg.bimanual:
            state = self._bimanual_state(observation, state)
        blocks = [TextBlock(json.dumps(state, allow_nan=False))]
        if self.budgets is not None:
            elapsed = observation.t - self._sim_start
            state["budgets"] = {
                "simulated_horizon_s": self.budgets["horizon_s"],
                "simulated_remaining_s": max(0, self.budgets["horizon_s"] - elapsed),
                "model_calls_remaining_including_this_call": max(
                    0, self.budgets["max_model_calls"] - self.session.turn
                ),
                "inference_time_mode": self.budgets["time_mode"],
                "note": "Geometry and observation tools also consume one model call; "
                "physical motion/wait time counts against the simulated horizon.",
            }
            blocks[0] = TextBlock(json.dumps(state, allow_nan=False))
        frames = self._frames(observation)
        projection_cameras = [
            name
            for name, frame in frames.items()
            if frame is not None
            and frame.K is not None
            and frame.pose is not None
            and frame.convention == "opencv"
        ]
        if self.cfg.table_projection:
            blocks.append(
                TextBlock(
                    "project_table uses the current named camera: u=0 left, "
                    "u=1 right, v=0 top, v=1 bottom. Supply its exact capture "
                    "time. Returns calibrated table intersection only, not "
                    "object height, grasp pose or object identity. "
                    f"Calibrated cameras available: {projection_cameras}."
                )
            )
        depth_cameras = [name for name in projection_cameras if frames[name].depth is not None]
        if self.cfg.depth_probe and depth_cameras:
            blocks.append(
                TextBlock(
                    "probe_point reads the rig's depth image at your selected "
                    "pixel (u=0 left, u=1 right, v=0 top, v=1 bottom) and "
                    "returns the visible surface point in root metres with its "
                    "height above the table. It is a surface, not an object "
                    "centre; probe the top of an object for its height and a "
                    "clear table pixel for the table. Supply the exact capture "
                    f"time. Depth cameras available: {depth_cameras}."
                )
            )
        if self.cfg.action_sequence:
            blocks.append(
                TextBlock(
                    "execute_sequence runs several already-decided steps in "
                    "one turn with per-step receipts. It stops early when a "
                    "step is blocked or expires, or when a close ends on "
                    "nothing; the receipt names the stopped step. Each step "
                    "keeps the single-tool bounds."
                )
            )
        if self.cfg.stereo_triangulation:
            blocks.append(
                TextBlock(
                    (
                        "triangulate_point requires selecting the same physical "
                        "feature in two different current RGB images, not two "
                        "different bounding-box centres. Coordinates are "
                        "normalized [u,v]: 0 left/top, 1 right/bottom. Supply the "
                        "current capture time. Ray mismatch is a geometry check, "
                        "not object truth."
                    )
                    if self.cfg.bimanual
                    else "triangulate_point requires selecting the same physical "
                    "feature in the two current RGB images, not two different "
                    "bounding-box centres. Coordinates are normalized [u,v]: "
                    "0 left/top, 1 right/bottom. Supply the current capture "
                    "time. Ray mismatch is a geometry check, not object truth."
                )
            )
        extra_views = (
            list((observation.extra or {}).get("views") or []) if (self.cfg.extra_views) else []
        )
        if self.cfg.camera_geometry_prompt:
            from rung.interfaces.camera_prompt import camera_geometry_text

            blocks.append(
                TextBlock(
                    camera_geometry_text(
                        self._shown(observation, extra_views), observation.tcp, self.cfg.image_scale
                    )
                )
            )
        if self.cfg.recent_frames:
            recent = list((observation.extra or {}).get("recent_frames") or [])
            if recent:
                blocks.append(
                    TextBlock(
                        "RECENT FRAMES: earlier captures from the last motion, oldest first, "
                        "then the current frames. Use them to judge what moved; act on the "
                        "current frames and the measured state."
                    )
                )
            for age_s, frame in recent:
                blocks += [
                    TextBlock(
                        f"Camera {frame.name}, {age_s:.2f}s before the current "
                        f"frame (captured at {frame.t}s)"
                    ),
                    _image_block(frame.rgb, f"{frame.name}-past", self.cfg.image_scale),
                ]
        snapshot, self._look_snapshot = self._look_snapshot, None
        if snapshot is not None:
            frame, tcp = snapshot
            blocks += [
                TextBlock(
                    f"LOOK VIEW: camera {frame.name} at the look viewpoint, "
                    f"captured at {frame.t:.3f}s "
                    f"with the TCP at {np.round(tcp.pos, 3).tolist()} m. The arm has since "
                    "returned (see look_result); the current frames follow."
                ),
                _image_block(frame.rgb, f"{frame.name}-look", self.cfg.image_scale),
            ]
        for frame in self._shown(observation, extra_views):
            if frame is not None:
                rgb = frame.rgb
                if self.cfg.public_overlay and frame is observation.scene:
                    from rung.interfaces.public_overlay import public_geometry_overlay

                    rgb = public_geometry_overlay(frame, observation.tcp, self.caps.workspace)
                    blocks.append(
                        TextBlock(
                            "Rig-only overlay: cyan grid labels are root (x,y) metres on the "
                            "declared table plane; magenta marks measured TCP. Grid lines may "
                            "overlay nearer objects; use RGB to assess support and object height."
                        )
                    )
                if self.cfg.tcp_marker:
                    from rung.interfaces.public_overlay import public_tcp_marker

                    rgb = public_tcp_marker(frame, observation.tcp)
                    blocks.append(
                        TextBlock(
                            "Magenta TCP cross marks the projected measured fingertip midpoint, "
                            "not the finger-pad centre. It uses public rig calibration only; "
                            "it does not identify objects, contact, depth or a grasp location."
                        )
                    )
                blocks += [
                    TextBlock(f"Camera {frame.name}, captured at {frame.t}s"),
                    _image_block(rgb, frame.name, self.cfg.image_scale),
                ]
        before, started = self.session.turn, time.monotonic()
        tool_specs = tools(
            self.cfg.motion_tools,
            self.cfg.table_projection,
            projection_cameras,
            self.cfg.stereo_triangulation,
            self.cfg.tcp_frame_motions,
            self.cfg.depth_probe,
            depth_cameras,
            self.cfg.action_sequence,
            self.cfg.max_sequence_steps,
            self.cfg.terminal_tools,
            self.cfg.done_awaits_confirmation,
            self.cfg.look_tool and observation.wrist is not None,
            projection_cameras,
            self.cfg.look_max_steps,
            **({"arms": [a.name for a in self.caps.arms]} if self.cfg.bimanual else {}),
        )
        try:
            system = SYSTEM_BIMANUAL if self.cfg.bimanual else SYSTEM
            if self.cfg.tcp_frame_motions:
                system += (
                    "\nmove_relative_tcp uses the current measured TCP axes, not camera "
                    "axes: local +y is jaw closing, +z is approach. Translation is "
                    "transformed by the pre-action TCP orientation; rotation pivots "
                    "at the fingertip midpoint. Use move_relative for root axes."
                )
            if self.cfg.table_extent and self.caps.workspace.table_extent is not None:
                system += (
                    "\ntable_extent_m in the state is the table top's footprint in root "
                    "metres. Anything released or carried past an edge falls to the "
                    "floor and the task can no longer succeed; the table_edge receipt "
                    "gives the TCP's margin to the nearest edge after each action."
                )
            if self.cfg.terminal_tools and self.cfg.done_awaits_confirmation:
                system += (
                    "\ndone does not end the trial by itself: you hold still for "
                    f"{self.cfg.done_confirmation_wait_s:g} s while the independent "
                    "grader checks; if it confirms, the trial ends, otherwise the next "
                    "observation carries a done_not_confirmed receipt and you continue. "
                    "Calling done when the task is not complete only costs time."
                )
            if not self.cfg.terminal_tools:
                system += (
                    "\nThere is no done or give_up tool. The episode ends by itself "
                    "when the evaluator confirms the task is complete, or when "
                    "simulated time runs out; time spent waiting on you is free. "
                    "If you believe the task is complete, re-check every object the "
                    "instruction names in all current views; if any is not yet where "
                    "it must be, keep working; otherwise hold with observe or a short "
                    "wait. Never stop trying while time remains."
                )
            if self.cfg.look_tool and observation.wrist is not None:
                system += (
                    "\nlook moves the wrist camera to view a point from a chosen "
                    "distance, elevation and azimuth and captures there, as one guarded "
                    "motion. Use it when the target is out of view, occluded or "
                    "uncertain: aim it at the last known or likely location (an "
                    "earlier receipt, a scene-camera pixel) before searching blindly. "
                    "The look_result receipt gives the reached viewpoint and, with a "
                    "calibrated wrist camera, where the point should appear in the view."
                )
            if self.cfg.spatial_frame_cues:
                system += (
                    "\nFor directional task instructions, use the robot-root frame. "
                    "An object to the robot's left has greater root y than the reference; "
                    "to its right has smaller root y. Keep root x approximately aligned "
                    "for a lateral placement. Screen-left depends on the camera and "
                    "need not be robot-left. Estimate reference locations from RGB; "
                    "no object state or native score is available."
                )
            response = self.session.complete(system, blocks, tool_specs)
        except Exception:
            if self.session.turn > before:
                self.pending_usage.add(
                    ModelUsage(calls=1, unknown_cost_calls=1, latency_s=time.monotonic() - started)
                )
            raise
        self.pending_usage.add(response.usage)
        try:
            if len(response.tool_calls) != 1:
                raise ValueError("exactly one tool call required; no calls executed")
            call = response.tool_calls[0]
            specs = {t.name: t for t in tool_specs}
            if call.name not in specs:
                raise ValueError("unknown tool")
            args = call.arguments
            if isinstance(args, dict) and "malformed_arguments_json" in args:
                # A CLI transport forwards a reply whose arguments were not valid JSON
                # under this key; the model gets the parse error as an ordinary receipt.
                raise ValueError(
                    "arguments were not valid JSON: "
                    f"{args['malformed_arguments_json']}; "
                    "send one JSON object with exactly the tool's fields"
                )
            if isinstance(args, dict) and "images_not_viewed" in args:
                # A CLI transport forwards a proposal made without opening every current
                # image under this key; nothing is dispatched and the model is told.
                raise ValueError(f"proposal rejected: {args['images_not_viewed']}")
            if call.name == "look" and isinstance(args, dict):
                # Optional target fields: strict transports send unused ones as null.
                args = {k: v for k, v in args.items() if v is not None}
                schema = specs[call.name].parameters
                if not (set(schema["required"]) <= set(args) <= set(schema["properties"])):
                    raise ValueError("invalid arguments or missing note")
            if (
                not isinstance(args, dict)
                or (
                    call.name != "look"
                    and set(args) != set(specs[call.name].parameters["required"])
                )
                or not isinstance(args.get("note"), str)
            ):
                raise ValueError("invalid arguments or missing note")
            payload = {
                "tool_call_id": call.id,
                "arguments": args,
                "requested_model": self.session.model,
                "returned_model": (response.raw or {}).get("model"),
                "native_session_id": self.session.session_id,
                "native_turn": self.session.turn,
            }
            if call.name == "project_table":
                from rung.interfaces.table_projection import project_table_pixel

                if not isinstance(args["camera"], str):
                    raise ValueError("camera must be a name")
                frame = frames.get(args["camera"])
                if frame is None:
                    raise ValueError("requested camera unavailable")
                self._geometry_receipt = project_table_pixel(
                    frame, args["u"], args["v"], args["capture_time_s"], self.caps.workspace.table_z
                )
                if self.cfg.geometry_feedback:
                    from rung.interfaces.table_projection import reproject_public_point

                    self._geometry_receipt["reprojection"] = reproject_public_point(
                        self._geometry_receipt["table_intersection_xyz_m"], list(frames.values())
                    )
                payload["geometry_result"] = self._geometry_receipt
                self._accepted_call_id = call.id
                return Decision(call.name, args["note"], payload=payload)
            if call.name == "probe_point":
                from rung.interfaces.depth_probe import probe_depth_pixel

                if not isinstance(args["camera"], str):
                    raise ValueError("camera must be a name")
                frame = frames.get(args["camera"])
                if frame is None:
                    raise ValueError("requested camera unavailable")
                self._geometry_receipt = probe_depth_pixel(
                    frame,
                    args["u"],
                    args["v"],
                    args["capture_time_s"],
                    table_z=self.caps.workspace.table_z,
                )
                if self.cfg.geometry_feedback:
                    from rung.interfaces.table_projection import reproject_public_point

                    self._geometry_receipt["reprojection"] = reproject_public_point(
                        self._geometry_receipt["surface_point_xyz_m"], list(frames.values())
                    )
                payload["geometry_result"] = self._geometry_receipt
                self._accepted_call_id = call.id
                return Decision(call.name, args["note"], payload=payload)
            if call.name == "execute_sequence":
                steps = args["steps"]
                if (
                    not isinstance(steps, list)
                    or not 1 <= len(steps) <= self.cfg.max_sequence_steps
                ):
                    raise ValueError("steps must be a non-empty list within the step limit")
                targets, pose, planned = [], observation.tcp, []
                poses = {a: s.tcp for a, s in (observation.arms or {}).items()}
                for index, step in enumerate(steps):
                    if not isinstance(step, dict) or not isinstance(step.get("tool"), str):
                        raise ValueError(f"step {index} needs a tool name")
                    # Strict-mode transports send every step field, null when unused.
                    step = {k: v for k, v in step.items() if v is not None}
                    name = step["tool"]
                    allowed = {t.name for t in tool_specs} & set(SEQUENCE_TOOLS)
                    if name not in allowed:
                        raise ValueError(f"step {index}: {name} is not a sequence step tool")
                    fields = STEP_FIELDS[name]
                    if self.cfg.bimanual and name != "wait":
                        fields = ("arm", *fields)
                    if set(step) != {"tool", *fields}:
                        raise ValueError(f"step {index}: {name} needs exactly {list(fields)}")
                    if name == "wait":
                        seconds = self._wait_seconds(step)
                        targets.append(
                            TCPTarget(
                                pose,
                                timeout_s=self.cfg.timeout_s,
                                min_dwell_s=seconds,
                                max_lin_speed_mps=min(
                                    self.cfg.max_speed_mps, self.caps.max_lin_speed_mps
                                ),
                                max_ang_speed_rps=min(
                                    self.cfg.max_ang_speed_rps, self.caps.max_ang_speed_rps
                                ),
                                meta={
                                    "profile": self.interface,
                                    "tool_call_id": call.id,
                                    "sequence_step": index,
                                    "step_tool": name,
                                    "step_args": step,
                                },
                            )
                        )
                    else:
                        # Relative steps chain from the predicted pose; the receipt tells the
                        # model what was actually reached.
                        arm = step.get("arm") if self.cfg.bimanual else None
                        if arm is not None:
                            pose, gripper = self._arm_motion(name, step, poses)
                        else:
                            pose, gripper = self._motion(name, step, pose)
                        targets.append(
                            TCPTarget(
                                pose,
                                gripper,
                                min(self.cfg.max_speed_mps, self.caps.max_lin_speed_mps),
                                min(self.cfg.max_ang_speed_rps, self.caps.max_ang_speed_rps),
                                self.cfg.timeout_s,
                                meta={
                                    "profile": self.interface,
                                    "tool_call_id": call.id,
                                    "sequence_step": index,
                                    "step_tool": name,
                                    "step_args": step,
                                    "stop_if_closed_on_nothing": name == "set_gripper"
                                    and step["action"] == "close",
                                },
                                arm=arm,
                            )
                        )
                    planned.append(
                        {
                            "step": index,
                            "tool": name,
                            "target_xyz_m": pose.pos.tolist(),
                            "target_quat_wxyz": pose.quat.tolist(),
                            **({"arm": arm} if self.cfg.bimanual and name != "wait" else {}),
                        }
                    )
                payload["sequence"] = {"planned_steps": planned}
                self._sequence_targets = targets
                if self.cfg.bimanual:
                    self._sequence_start_poses = {a: s.tcp for a, s in observation.arms.items()}
                travel = observation.gripper.travel
                self._sequence_hold_ref = (
                    travel if self.cfg.sequence_grasp_guard and 0.2 < travel < 0.95 else None
                )
                self._sequence_result = {
                    "planned": len(targets),
                    "executed": 0,
                    "stopped_after_step": None,
                    "stop_reason": None,
                    "interrupted": False,
                    "steps": [],
                }
                self._accepted_call_id = call.id
                return Decision(call.name, args["note"], payload=payload)
            if call.name == "look":
                return self._plan_look(call, args, observation, payload)
            if call.name == "triangulate_point":
                from rung.interfaces.table_projection import triangulate_selected_pixels

                if self.cfg.bimanual:
                    first, second = (
                        frames.get(args["first_camera"]),
                        frames.get(args["second_camera"]),
                    )
                    if first is None or second is None or first is second:
                        raise ValueError("choose two different available cameras")
                    self._geometry_receipt = triangulate_selected_pixels(
                        first, second, args["first_uv"], args["second_uv"], args["capture_time_s"]
                    )
                else:
                    self._geometry_receipt = triangulate_selected_pixels(
                        observation.scene,
                        observation.wrist,
                        args["scene_uv"],
                        args["wrist_uv"],
                        args["capture_time_s"],
                    )
                if self.cfg.geometry_feedback:
                    from rung.interfaces.table_projection import reproject_public_point

                    self._geometry_receipt["reprojection"] = reproject_public_point(
                        self._geometry_receipt["selected_point_xyz_m"], list(frames.values())
                    )
                payload["geometry_result"] = self._geometry_receipt
                self._accepted_call_id = call.id
                return Decision(call.name, args["note"], payload=payload)
            if call.name in ("done", "give_up", "observe"):
                verdict = {
                    "done": Verdict.DONE,
                    "give_up": Verdict.GIVE_UP,
                    "observe": Verdict.CONTINUE,
                }[call.name]
                self._accepted_call_id = call.id
                hold = call.name == "observe"
                dwell = self.caps.control_period_s
                if call.name == "done" and self.cfg.done_awaits_confirmation:
                    # Hold still while the native grader checks; the loop continues unless
                    # the native side ends the episode on confirmed success.
                    verdict, hold = Verdict.CONTINUE, True
                    dwell = max(dwell, self.cfg.done_confirmation_wait_s)
                    self._done_pending = observation.t
                    payload["done_pending_confirmation"] = {
                        "hold_s": dwell,
                        "declared_at_sim_time_s": observation.t,
                    }
                targets = (
                    [
                        TCPTarget(
                            observation.tcp,
                            timeout_s=self.cfg.timeout_s,
                            max_lin_speed_mps=min(
                                self.cfg.max_speed_mps, self.caps.max_lin_speed_mps
                            ),
                            max_ang_speed_rps=min(
                                self.cfg.max_ang_speed_rps, self.caps.max_ang_speed_rps
                            ),
                            min_dwell_s=dwell,
                        )
                    ]
                    if hold
                    else []
                )
                return Decision(
                    call.name, args["note"], targets=targets, verdict=verdict, payload=payload
                )
            if call.name == "wait":
                seconds = self._wait_seconds(args)
                self._accepted_call_id = call.id
                return Decision(
                    call.name,
                    args["note"],
                    targets=[
                        TCPTarget(
                            observation.tcp,
                            timeout_s=self.cfg.timeout_s,
                            min_dwell_s=seconds,
                            max_lin_speed_mps=min(
                                self.cfg.max_speed_mps, self.caps.max_lin_speed_mps
                            ),
                            max_ang_speed_rps=min(
                                self.cfg.max_ang_speed_rps, self.caps.max_ang_speed_rps
                            ),
                        )
                    ],
                    payload=payload,
                )
            if self.cfg.bimanual:
                return self._bimanual_motion(call, args, observation, payload)
            pose, gripper = self._motion(call.name, args, observation.tcp)
            target = TCPTarget(
                pose,
                gripper,
                min(self.cfg.max_speed_mps, self.caps.max_lin_speed_mps),
                min(self.cfg.max_ang_speed_rps, self.caps.max_ang_speed_rps),
                self.cfg.timeout_s,
                meta={"profile": self.interface, "tool_call_id": call.id},
            )
            self._accepted_call_id = call.id
            return Decision(call.name, args["note"], targets=[target], payload=payload)
        except ValueError as exc:
            self.session.resolve(
                {c.id: {"error": str(exc), "executed": False} for c in self.session.pending}
            )
            raise

    @staticmethod
    def _wait_seconds(args):
        seconds = args["seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not np.isfinite(seconds)
            or not 0.05 <= seconds <= 3
        ):
            raise ValueError("wait seconds must be finite in [.05,3]")
        return seconds

    def _motion(self, name, args, start, workspace=None):
        """Target pose and gripper command for one motion or gripper tool from ``start``.

        The same bounds apply whether the tool is called alone or as a sequence step;
        Sequence preflight uses planned poses; execution revalidates against achieved poses.
        """
        pose, gripper = start, GripperCommand()
        if name == "set_gripper":
            if args["action"] not in ("open", "close"):
                raise ValueError("invalid gripper action")
            return pose, GripperCommand(GripperAction(args["action"]))

        def vector_arg(key, size):
            v = args[key]
            if (
                not isinstance(v, list)
                or len(v) != size
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v)
            ):
                raise ValueError(f"{key} must contain {size} numbers")
            result = np.asarray(v, dtype=float)
            if not np.all(np.isfinite(result)):
                raise ValueError("nonfinite motion vector")
            return result

        if name == "move_to":
            xyz, quat = vector_arg("xyz_m", 3), vector_arg("quat_wxyz", 4)
            if abs(float(np.linalg.norm(quat)) - 1) > 0.05:
                raise ValueError("absolute quaternion must be approximately unit length")
            pose = Pose(xyz, quat)
            delta = pose.pos - start.pos
            angle = quat_angle_between(start.quat, pose.quat)
        elif name in ("move_relative", "move_relative_tcp"):
            delta, rot = vector_arg("delta_xyz_m", 3), vector_arg("rotation_vector_rad", 3)
            angle = np.linalg.norm(rot)
            quat = start.quat
            tcp_local = name == "move_relative_tcp"
            if tcp_local:
                delta = start.rot @ delta
            if angle > 1e-12:
                increment = quat_from_axis_angle(rot, angle)
                quat = quat_mul(quat, increment) if tcp_local else quat_mul(increment, quat)
            pose = Pose(start.pos + delta, quat)
        else:
            raise ValueError("unknown motion tool")
        if np.linalg.norm(delta) > self.cfg.max_translation_m or angle > self.cfg.max_rotation_rad:
            raise ValueError("nonfinite or out-of-bounds motion; no motion executed")
        if not (workspace or self.caps.workspace).contains(pose.pos):
            raise ValueError("target outside reachable workspace")
        return pose, gripper

    def _clear_look(self):
        self._look_targets = None
        self._look_result = None
        self._look_plan = None
        self._look_start = None
        self._look_return = False

    def _look_target(self, args, observation):
        """The root point a look aims at, and where it came from."""
        has_xyz, has_pixel = "target_xyz_m" in args, "target_pixel" in args
        if has_xyz == has_pixel:
            raise ValueError("look needs exactly one of target_xyz_m or target_pixel")
        if has_xyz:
            xyz = args["target_xyz_m"]
            if (
                not isinstance(xyz, list)
                or len(xyz) != 3
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in xyz)
            ):
                raise ValueError("target_xyz_m must contain 3 numbers")
            return np.asarray(xyz, dtype=float), {"source": "target_xyz_m"}
        pixel = args["target_pixel"]
        if not isinstance(pixel, dict) or set(pixel) != {"camera", "u", "v", "capture_time_s"}:
            raise ValueError("target_pixel needs exactly camera, u, v and capture_time_s")
        frame = (
            {"scene": observation.scene, "wrist": observation.wrist}.get(pixel["camera"])
            if isinstance(pixel["camera"], str)
            else None
        )
        if frame is None:
            raise ValueError("requested camera unavailable")
        if self.cfg.depth_probe and frame.depth is not None:
            # Depth only where the run already declares RGB-D input (depth_probe).
            from rung.interfaces.depth_probe import probe_depth_pixel

            probe = probe_depth_pixel(
                frame,
                pixel["u"],
                pixel["v"],
                pixel["capture_time_s"],
                table_z=self.caps.workspace.table_z,
            )
            return np.asarray(probe["surface_point_xyz_m"], float), {
                "source": f"depth surface under {frame.name} pixel",
                "normalized_uv": [pixel["u"], pixel["v"]],
            }
        from rung.interfaces.table_projection import project_table_pixel

        hit = project_table_pixel(
            frame, pixel["u"], pixel["v"], pixel["capture_time_s"], self.caps.workspace.table_z
        )
        return np.asarray(hit["table_intersection_xyz_m"], float), {
            "source": f"table plane under {frame.name} pixel (raised objects cause parallax)",
            "normalized_uv": [pixel["u"], pixel["v"]],
        }

    def _plan_look(self, call, args, observation, payload):
        from rung.interfaces import look as geometry

        if observation.wrist is None:
            raise ValueError("look needs a wrist camera")
        defaults = {
            "distance_m": geometry.DEFAULT_DISTANCE_M,
            "elevation_deg": geometry.DEFAULT_ELEVATION_DEG,
            "azimuth_deg": geometry.DEFAULT_AZIMUTH_DEG,
            "return_to_previous_pose": False,
        }
        filled = sorted(k for k in defaults if k not in args)
        view = {k: args.get(k, defaults[k]) for k in defaults}
        back = view.pop("return_to_previous_pose")
        if not isinstance(back, bool):
            raise ValueError("return_to_previous_pose must be true or false")
        target, origin = self._look_target(args, observation)
        plan = geometry.plan_look(
            observation.tcp,
            observation.wrist,
            target,
            view["distance_m"],
            view["elevation_deg"],
            view["azimuth_deg"],
            self.caps.workspace,
            self.cfg.max_translation_m,
            self.cfg.max_rotation_rad,
            self.cfg.look_max_steps,
        )
        lin = min(self.cfg.max_speed_mps, self.caps.max_lin_speed_mps)
        ang = min(self.cfg.max_ang_speed_rps, self.caps.max_ang_speed_rps)
        targets, previous = [], observation.tcp
        for index, pose in enumerate(plan.waypoints):
            step = {"tool": "move_to", "xyz_m": pose.pos.tolist(), "quat_wxyz": pose.quat.tolist()}
            self._motion("move_to", step, previous)  # same bounds as a single move_to
            last = index == len(plan.waypoints) - 1
            targets.append(
                TCPTarget(
                    pose,
                    GripperCommand(),
                    lin,
                    ang,
                    self.cfg.timeout_s,
                    meta={
                        "profile": self.interface,
                        "tool_call_id": call.id,
                        "look_step": index,
                        "step_tool": "move_to",
                        "step_args": step,
                        "phase": "look",
                    },
                    min_dwell_s=geometry.SETTLE_S if last else 0.0,
                )
            )
            previous = pose
        wrist = observation.wrist
        planned = {
            "target_xyz_m": np.round(target, 4).tolist(),
            "target": origin,
            "view": view,
            "view_model": plan.view_model,
            "planned_look_tcp_xyz_m": np.round(plan.look_tcp.pos, 4).tolist(),
            "planned_look_tcp_quat_wxyz": np.round(plan.look_tcp.quat, 4).tolist(),
            "planned_camera_xyz_m": np.round(plan.camera_pose.pos, 4).tolist(),
            "transit_z_m": round(plan.transit_z, 4),
            "planned_steps": len(targets),
            "return_to_previous_pose": back,
        }
        if filled:
            planned["defaults_used"] = filled
        if plan.view_model == geometry.CALIBRATED and wrist.K is not None:
            planned["view_span_at_target_m"] = geometry.view_span(
                wrist.K, wrist.width, wrist.height, view["distance_m"]
            )
        if plan.notes:
            planned["notes"] = plan.notes
        payload["look"] = planned
        self._look_targets, self._look_plan = targets, plan
        self._look_start, self._look_return = observation.tcp, back
        self._look_result = {
            **planned,
            "executed_steps": 0,
            "reached_look_pose": False,
            "stop_reason": None,
            "interrupted": False,
        }
        self._accepted_call_id = call.id
        return Decision(call.name, args["note"], payload=payload)

    def _execute_look(self, execute_target, observe):
        """Forward path to the viewpoint, capture, optional retrace. Guarded like a sequence:
        the first step that does not reach stops the path, and the capture happens there."""
        from rung.interfaces import look as geometry

        targets, result, plan = self._look_targets, self._look_result, self._look_plan
        self._look_targets = None
        lin = min(self.cfg.max_speed_mps, self.caps.max_lin_speed_mps)
        ang = min(self.cfg.max_ang_speed_rps, self.caps.max_ang_speed_rps)

        def run(path, label):
            achieved, attempted = None, 0
            for index, target in enumerate(path):
                if achieved is not None:
                    try:
                        self._motion("move_to", target.meta["step_args"], achieved)
                    except ValueError as exc:
                        return attempted, f"{label} step {index} rejected from measured pose: {exc}"
                report = execute_target(target)
                achieved, attempted = report.final_tcp, index + 1
                if report.outcome is not ExecutionOutcome.REACHED:
                    reason = f"{label} step {index} {report.outcome.value}"
                    return attempted, reason + (f": {report.reason}" if report.reason else "")
            return attempted, None

        try:
            attempted, stop = run(targets, "look")
            result.update(
                executed_steps=attempted,
                stop_reason=stop,
                reached_look_pose=stop is None and attempted == len(targets),
            )
            if observe is None:
                result["frames"] = (
                    "no capture callback: the arm stays where the look "
                    "stopped and the current observation is the look view"
                )
                return
            seen = observe()
            result["achieved_tcp_xyz_m"] = np.round(seen.tcp.pos, 4).tolist()
            result["capture_time_s"] = round(seen.wrist.t if seen.wrist is not None else seen.t, 4)
            expected = geometry.expected_pixel(seen.wrist, plan.target_xyz)
            if expected is not None:
                result["target_in_look_view"] = expected
            if not self._look_return:
                result["frames"] = "the current observation is the look view"
                return
            if seen.wrist is not None:
                self._look_snapshot = (seen.wrist, seen.tcp)
            back = [*plan.waypoints[: max(attempted - 1, 0)][::-1], self._look_start]
            path = [
                TCPTarget(
                    pose,
                    GripperCommand(),
                    lin,
                    ang,
                    self.cfg.timeout_s,
                    meta={
                        "profile": self.interface,
                        "look_return_step": index,
                        "step_tool": "move_to",
                        "phase": "look_return",
                        "step_args": {
                            "tool": "move_to",
                            "xyz_m": pose.pos.tolist(),
                            "quat_wxyz": pose.quat.tolist(),
                        },
                    },
                )
                for index, pose in enumerate(back)
            ]
            done, stop = run(path, "return")
            result.update(
                frames="the look view is shown as LOOK VIEW before the current frames",
                return_steps=len(path),
                return_executed_steps=done,
                returned=stop is None,
                return_stop_reason=stop,
            )
        except BaseException:
            result["interrupted"] = True
            raise

    def _frames(self, observation):
        """Cameras by role name: scene and wrist, or scene and every arm's wrist."""
        if not self.cfg.bimanual:
            return {"scene": observation.scene, "wrist": observation.wrist}
        return {
            "scene": observation.scene,
            **{s.wrist.name: s.wrist for s in observation.arms.values() if s.wrist is not None},
        }

    def _shown(self, observation, extra_views):
        if not self.cfg.bimanual:
            return [observation.scene, *extra_views, observation.wrist]
        return [
            observation.scene,
            *extra_views,
            *(s.wrist for s in observation.arms.values() if s.wrist is not None),
        ]

    def _arm_state(self, observation):
        return {
            a: {
                "tcp_xyz_m": s.tcp.pos.tolist(),
                "tcp_quat_wxyz": s.tcp.quat.tolist(),
                "gripper_travel": s.gripper.travel,
            }
            for a, s in observation.arms.items()
        }

    def _bimanual_state(self, observation, state):
        """The single-arm state with the primary-arm fields replaced by one block per arm."""
        for key in (
            "tcp_xyz_m",
            "tcp_quat_wxyz",
            "tcp_axes_in_root",
            "gripper_travel",
            "gripper_opening_m",
            "gripper_geometry_m",
            "workspace_min_m",
            "workspace_max_m",
        ):
            state.pop(key, None)
        arms = {}
        for spec in self.caps.arms:
            s = observation.arms[spec.name]
            arms[spec.name] = {
                "tcp_xyz_m": s.tcp.pos.tolist(),
                "tcp_quat_wxyz": s.tcp.quat.tolist(),
                "tcp_axes_in_root": {
                    n: s.tcp.rot[:, i].tolist() for i, n in enumerate(("x", "y", "z"))
                },
                "gripper_travel": s.gripper.travel,
                "gripper_opening_m": s.gripper.opening_m,
                "gripper_geometry_m": {
                    "max_opening": spec.gripper.max_opening_m,
                    "finger_length": spec.gripper.finger_length_m,
                    "pad_width": spec.gripper.pad_width_m,
                },
                "workspace_min_m": np.asarray(spec.workspace.min_xyz).tolist(),
                "workspace_max_m": np.asarray(spec.workspace.max_xyz).tolist(),
                "base_xyz_m": spec.base.pos.tolist(),
                "wrist_camera": spec.wrist_camera,
            }
        return {
            "task": state.pop("task"),
            "sim_time_s": state.pop("sim_time_s"),
            "root_frame": self.caps.notes.get("root_frame", "robot-root frame"),
            "arms": arms,
            **state,
        }

    def _arm_motion(self, name, args, poses):
        """``_motion`` for one named arm, from that arm's pose in ``poses`` (updated)."""
        arm = args.get("arm")
        if arm not in poses:
            raise ValueError(f"unknown arm {arm!r}; arms are {sorted(poses)}")
        step = {k: v for k, v in args.items() if k != "arm"}
        pose, gripper = self._motion(name, step, poses[arm], self.caps.arm(arm).workspace)
        poses[arm] = pose
        return pose, gripper

    def _bimanual_motion(self, call, args, observation, payload):
        speeds = (
            min(self.cfg.max_speed_mps, self.caps.max_lin_speed_mps),
            min(self.cfg.max_ang_speed_rps, self.caps.max_ang_speed_rps),
        )
        meta = {"profile": self.interface, "tool_call_id": call.id}
        poses = {a: s.tcp for a, s in observation.arms.items()}
        if call.name == "move_both":
            relative = self.cfg.motion_tools in ("relative", "both")
            goals = []
            for spec in self.caps.arms:
                goal = args[spec.name]
                if not isinstance(goal, dict):
                    raise ValueError(f"{spec.name} needs an object of motion fields")
                pose, gripper = self._arm_motion(
                    "move_relative" if relative else "move_to", {**goal, "arm": spec.name}, poses
                )
                goals.append(
                    TCPTarget(pose, gripper, *speeds, self.cfg.timeout_s, meta=meta, arm=spec.name)
                )
            target = replace(goals[0], paired=tuple(goals[1:]))
        else:
            pose, gripper = self._arm_motion(call.name, args, poses)
            target = TCPTarget(
                pose, gripper, *speeds, self.cfg.timeout_s, meta=meta, arm=args["arm"]
            )
        self._accepted_call_id = call.id
        return Decision(call.name, args["note"], targets=[target], payload=payload)

    def execute_decision(self, decision, execute_target, observe=None):
        """Run a guarded sequence through the loop's motion callback.

        Single-tool decisions carry their target in ``decision.targets`` and are executed
        by the loop itself. A sequence stops at the first step that does not reach, or at a
        close that ends on nothing, so the model gets the next observation at the point
        where its plan diverged from the world. A loop limit raised mid-sequence propagates
        after the partial result is recorded. ``observe`` (the embodiment's) lets a look
        capture the view at its viewpoint before returning.
        """
        if getattr(self, "_look_targets", None):
            return self._execute_look(execute_target, observe)
        targets = self._sequence_targets
        if not targets:
            return
        result = self._sequence_result
        self._sequence_targets = None
        held = getattr(self, "_sequence_hold_ref", None)
        measured = dict(getattr(self, "_sequence_start_poses", None) or {})
        try:
            achieved = None
            for index, target in enumerate(targets):
                if achieved is not None:
                    step = target.meta["step_args"]
                    try:
                        if target.arm is not None:
                            pose, gripper = self._arm_motion(step["tool"], step, measured)
                        else:
                            pose, gripper = (
                                (achieved, target.gripper)
                                if step["tool"] == "wait"
                                else self._motion(step["tool"], step, achieved)
                            )
                    except ValueError as exc:
                        result.update(
                            stopped_after_step=index - 1,
                            stop_reason=f"step {index} rejected from measured pose: {exc}",
                        )
                        return
                    target = replace(target, pose=pose, gripper=gripper)
                report = execute_target(target)
                achieved = report.final_tcp
                for arm, receipt in (report.arms or {}).items():
                    measured[arm] = Pose(receipt["final_tcp_xyz_m"], receipt["final_tcp_quat_wxyz"])
                result["steps"].append(
                    {
                        "step": index,
                        "outcome": report.outcome.value,
                        "target_xyz_m": target.pose.pos.tolist(),
                        "final_xyz_m": achieved.pos.tolist(),
                        "final_quat_wxyz": achieved.quat.tolist(),
                        "gripper_travel_end": report.gripper_travel_end,
                    }
                )
                result["executed"] = index + 1
                if report.outcome is not ExecutionOutcome.REACHED:
                    result.update(
                        stopped_after_step=index, stop_reason=f"step {index} {report.outcome.value}"
                    )
                    return
                if (target.meta or {}).get(
                    "stop_if_closed_on_nothing"
                ) and report.gripper_travel_end > 0.95:
                    result.update(
                        stopped_after_step=index, stop_reason=f"step {index} closed on nothing"
                    )
                    return
                if self.cfg.sequence_grasp_guard:
                    action = getattr(target.gripper, "action", None)
                    end = report.gripper_travel_end
                    if action == "close":
                        held = end if end is not None and 0.2 < end < 0.95 else None
                    elif action == "open":
                        held = None
                    elif held is not None and end is not None and end > held + 0.10:
                        result.update(
                            stopped_after_step=index,
                            stop_reason=f"step {index} grasp lost: jaw travel rose "
                            f"from {held:.2f} to {end:.2f}, the object "
                            f"is no longer between the fingers",
                        )
                        return
        except BaseException:
            result["interrupted"] = True
            raise

    def finalize(self, observation, feedback, end_reason):
        """Archive the final transaction even when there is no subsequent API turn.

        Last-known state is labeled honestly; an interrupted action has no fabricated
        completion report. Native success is not available to this callback.
        """
        if self.session.pending:
            self.session.resolve(
                {
                    c.id: {
                        "episode_end_reason": str(end_reason),
                        "geometry_result": self._geometry_receipt,
                        "sequence_result": self._sequence_result,
                        **({"look_result": self._look_result} if self.cfg.look_tool else {}),
                        "accepted": c.id == self._accepted_call_id,
                        "last_observation_time_s": observation.t,
                        "last_measured_tcp_xyz_m": observation.tcp.pos.tolist(),
                        "last_measured_tcp_quat_wxyz": observation.tcp.quat.tolist(),
                        "last_execution_feedback": feedback.structured
                        if c.id == self._accepted_call_id
                        else None,
                        "terminal_tool_acknowledged": c.name in ("done", "give_up")
                        and c.id == self._accepted_call_id,
                    }
                    for c in self.session.pending
                }
            )
