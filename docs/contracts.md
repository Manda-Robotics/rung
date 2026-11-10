# Contracts

Reference for `src/rung/contracts/`: field names, units, conventions and validation rules,
plus the `Embodiment` protocol that adapters implement. Where this document and the code
disagree, the code is authoritative.

| Module | Contents |
|---|---|
| `frames.py` | Conventions, `Pose`, rotations, camera geometry |
| `observation.py` | `CameraFrame`, `GripperState`, `JointState`, `Observation` |
| `capabilities.py` | `GripperGeometry`, `Workspace`, `CameraSpec`, `EmbodimentCapabilities` |
| `action.py` | `GripperCommand`, `TCPTarget`, `ExecutionOutcome`, `ExecutionReport`, `residual_of` |
| `trace.py` | Event kinds, `TraceEvent`, `TraceWriter`, `RunProvenance` |
| `validate.py` | `ContractError`, `validate_camera`, `validate_observation`, `validate_target` |

## Frames and conventions (`frames.py`)

Conventions are stated once in the `frames.py` module docstring and imported elsewhere.

- Positions in metres, angles in radians.
- Quaternions are `(w, x, y, z)`, unit length, and rotate child-frame vectors into the
  parent frame.
- A `Pose` of frame B in frame A holds `pos` (origin of B in A) and `quat` (B-coordinates
  to A-coordinates). Construction coerces `pos` to a length-3 and `quat` to a length-4
  `float64` array, normalises `quat`, and raises `ValueError` if it is non-finite or has
  norm below `1e-9`. Composition: `pose_A_C = pose_A_B * pose_B_C` (`Pose.compose`,
  `Pose.__mul__`).
- The robot root frame is the fixed base frame of the arm. Table height is declared by
  `EmbodimentCapabilities.workspace.table_z`, never assumed.
- Cameras use OpenCV axes inside the harness: `+x` right, `+y` down, `+z` forward.
  `CAMERA_CONVENTIONS = ("opencv", "opengl", "ros")`; `ros` names the same axes as
  `opencv`. `convert_camera_pose(pose, source_convention)` returns OpenCV axes: identity for
  `opencv` and `ros`, and a `diag(1, -1, -1)` axis flip for `opengl` (Isaac / USD:
  `+y` up, `-z` forward). The camera origin does not move.
- Depth is optical-axis depth (camera-frame `z`), not ray distance: metres, `float32`,
  `NaN` where invalid.
- Intrinsics `K` are valid for the delivered resolution. `rescale_intrinsics` scales
  `fx, fy, cx, cy` on resize; `crop_intrinsics` shifts `cx, cy` on crop.

Geometry helpers: `quat_to_rot`, `rot_to_quat` (trace-based branch selection), `quat_mul`,
`quat_from_axis_angle`, `quat_angle_between`, `yaw_of`, `deproject` (pixels and depth to
camera or root-frame points), `project` (points to pixels; callers mask `z <= 0`), and
`depth_image_to_points` (valid depth pixels to points, optional stride).

## Observation (`observation.py`)

Per environment and flat. Every field names its frame and unit; adapters convert, the
core never guesses.

### `CameraFrame`

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Role name: `"scene"` or `"wrist"` |
| `rgb` | `uint8[H, W, 3]` | Required |
| `t` | `float` | Capture time, simulated seconds |
| `depth` | `float32[H, W] \| None` | Optical-axis depth, metres, `NaN` where invalid |
| `K` | `float64[3, 3] \| None` | Intrinsics for the delivered resolution of `rgb` and `depth` |
| `pose` | `Pose \| None` | Camera in the root frame, OpenCV axes |
| `convention` | `str` | Default `"opencv"` |
| `state_at_capture` | `Pose \| None` | TCP pose at capture time |

Properties: `height`, `width`, and `has_geometry` (`depth`, `K` and `pose` all present).

### `GripperState`

| Field | Type | Meaning |
|---|---|---|
| `travel` | `float` | `0.0` fully open, `1.0` fingers touching |
| `opening_m` | `float \| None` | Jaw gap in metres, reported or derived by the adapter from `travel` |
| `velocity` | `float \| None` | `d(travel)/dt` |
| `effort` | `float \| None` | Normalised `0..1` |

### `JointState`

`positions` (required), `velocities` (optional), `names` (optional tuple).

### `Observation`

| Field | Type | Meaning |
|---|---|---|
| `t` | `float` | Observation time, simulated seconds |
| `scene` | `CameraFrame` | Required |
| `tcp` | `Pose` | Tool centre point: closing axis local `+y`, approach direction local `+z`. The adapter applies `EmbodimentCapabilities.tcp_from_ee` |
| `gripper` | `GripperState` | Required |
| `wrist` | `CameraFrame \| None` | May be RGB only |
| `joints` | `JointState \| None` | When the embodiment exposes joint state |
| `extra` | `dict[str, Any]` | Free-form, default `{}` |

`Observation.cameras` returns `{name: CameraFrame}` for `scene` and, when present, `wrist`.

## Capabilities (`capabilities.py`)

Declared once by the adapter. These are facts about the rig, not tunables.

### `GripperGeometry`

`max_opening_m`, `finger_length_m`, `pad_width_m` (required, metres) and `close_time_s`
(default `1.0`). `opening_for_travel(travel)` returns
`clamp(1 - travel, 0, 1) * max_opening_m`.

### `Workspace`

`min_xyz`, `max_xyz` (axis-aligned reachable box, root frame) and `table_z`.
`contains(p, margin=0.0)` tests `min_xyz + margin <= p <= max_xyz - margin`; a positive
margin shrinks the box, a negative margin grows it.

### `CameraSpec`

`name`, `width`, `height`, `has_depth`, `mounted_on` (`"world"` or `"tcp"`, default
`"world"`).

### `EmbodimentCapabilities`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | | Embodiment identifier |
| `cameras` | `tuple[CameraSpec, ...]` | | Declared cameras |
| `gripper` | `GripperGeometry` | | |
| `workspace` | `Workspace` | | |
| `control_period_s` | `float` | | Control step, seconds |
| `tcp_from_ee` | `Pose` | | TCP in the reported end-effector frame; applied by the adapter, recorded for audit |
| `command_kinds` | `tuple[str, ...]` | `("tcp_target",)` | `COMMAND_KINDS = ("tcp_target",)` is the only kind defined |
| `has_joint_state` | `bool` | `False` | |
| `max_lin_speed_mps` | `float` | `0.25` | Hard cap, enforced by `validate_target` |
| `max_ang_speed_rps` | `float` | `1.0` | Hard cap, enforced by `validate_target` |
| `steps_per_efold` | `float` | `3.0` | Measured control steps to reduce the remaining distance by a factor of e |
| `notes` | `dict[str, str]` | `{}` | Free-form |

`camera(name)` returns the `CameraSpec` or raises `KeyError`.
`time_to_converge(distance_m, tolerance_m=0.002, safety=2.0)` returns a motion time budget:
`safety * steps_per_efold * ln(distance / tolerance) * control_period_s`, or
`max(4 * control_period_s, 0.5)` when the distance is within tolerance. The executor
clamps it to 6 to 45 s per motion.

## Action (`action.py`)

The core emits TCP targets only. An adapter plays a target out for at most `timeout_s`
or until a newer target supersedes it, and reports what happened.

### `GripperCommand`

`action`: `GripperAction.OPEN`, `CLOSE` or `HOLD` (default `HOLD`). `opening_m`: target
jaw gap for `OPEN`; `None` means fully open.

### `TCPTarget`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `pose` | `Pose` | | Absolute TCP pose in the root frame, reached along a straight line |
| `gripper` | `GripperCommand` | hold | |
| `max_lin_speed_mps` | `float` | `0.10` | |
| `max_ang_speed_rps` | `float` | `0.50` | |
| `timeout_s` | `float` | `5.0` | Includes gripper actuation as well as motion |
| `id` | `str` | `uuid4().hex[:12]` | |
| `meta` | `dict` | `{}` | Provenance (interface, primitive, phase); never read by adapters |

There is no cancel field. A newer target preempts the previous one, which is reported as
`SUPERSEDED`. `REACHED` describes command execution; it does not establish attachment.

### `ExecutionOutcome`

String enum: `reached`, `expired`, `superseded`, `clamped`, `rejected`, `blocked`.

### `ExecutionReport`

Attempted and executed motion are logged separately so controller interventions stay
visible.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `target_id` | `str` | | `TCPTarget.id` |
| `outcome` | `ExecutionOutcome` | | |
| `t_start`, `t_end` | `float` | | Simulated seconds |
| `final_tcp` | `Pose` | | TCP pose at the end |
| `residual_m` | `float` | | Largest remaining per-axis position offset |
| `residual_axis` | `str` | | `"x"`, `"y"`, `"z"` or `"rot"` |
| `residual_rot_rad` | `float` | `0.0` | |
| `executed_lin_m` | `float` | `0.0` | Distance travelled |
| `executed_rot_rad` | `float` | `0.0` | Rotation travelled |
| `clamped_to` | `Pose \| None` | `None` | Set when clamped |
| `reason` | `str \| None` | `None` | Human-readable explanation when not reached |
| `gripper_travel_end` | `float \| None` | `None` | |
| `waited_s` | `float` | `0.0` | Real seconds spent in the harness before this motion; held out in simulation in real-time mode |

`duration_s` returns `t_end - t_start`. `residual_of(target, actual)` returns
`(residual_m, residual_axis, residual_rot_rad)`: the largest absolute per-axis position
difference (`argmax`, so the axis is `x`, `y` or `z`) and `quat_angle_between` of the two
orientations.

## Trace (`trace.py`)

One JSONL row per event, numpy-safe, append-only.

### Event kinds

| Kind | Content |
|---|---|
| `observation` | Policy input at reset: instruction, task, policy, interface, limits |
| `model_call` | Request summary, response, usage, latency, dollars (defined; current policies log `model_request` and `model_response` instead) |
| `decision` | Parsed decision, verdict, reason and payload (usage as plain numbers), or the error |
| `attempt` | Target sent to the adapter, with requested and bounded timeout |
| `intervention` | Clamp or blocked motion |
| `execution` | The adapter's `ExecutionReport` summary, including `waited_s` |
| `primitive` | L2 primitive result and reason |
| `policy_end` | Episode ended by the policy loop, not yet graded |
| `execution_error` | Adapter exception on a target |
| `episode_end` | End reason, grader verdict, totals |
| `setup` | Run configuration at episode start (model, interface, table estimate, TCP offset, camera convention, time mode, tracking gain) |
| `model_request`, `model_response` | Transcript of each L2 model exchange: prompt, tools, images, response, thoughts, usage |
| `grounding`, `perception`, `plan`, `check` | Harness work between model calls: detections, catalog builds and reuse, grasp plans, verification checks |

**Servo flag.** `attempt` and `execution` rows carry `servo` in their payload so motions
never inflate decision counts. `TraceWriter.count(kind, episode=None, servo=None)` filters
on it.

### `TraceEvent`

`kind` (validated against `EVENT_KINDS`, else `ValueError`), `t_sim`, `payload`, `episode`,
`env_id` (default `0`), `t_wall` (default `time.time()`), `seq` (assigned by
`TraceWriter.write`).

### `TraceWriter`

Thread-safe append-only writer. A `None` path keeps events in memory (`events`) only.
`write(event)` assigns `seq`, appends in memory and, with a path, to the file.
`emit(kind, t_sim, episode, env_id=0, **payload)` wraps `write`.

The encoder serialises numpy arrays and scalars, enums, dataclasses, `Path`, and
torch-like tensors (`detach().cpu().tolist()`) without importing torch. `to_jsonable(x)`
round-trips a value through it; `read_trace(path)` reads JSONL rows, skipping blank lines.

### `RunProvenance`

Written once per run next to the traces.

| Field | Type | Default |
|---|---|---|
| `run_id` | `str` | |
| `started_wall` | `float` | |
| `harness_revision` | `str` | |
| `embodiment` | `str` | |
| `embodiment_revision` | `str \| None` | |
| `interface` | `str` | |
| `model` | `str \| None` | |
| `effort` | `str \| None` | |
| `prompt_hashes` | `dict[str, str]` | |
| `perception_version` | `str \| None` | |
| `grasp_version` | `str \| None` | |
| `task_role` | `str \| None` | |
| `limits` | `dict[str, Any]` | |
| `initial_state_audit` | `dict[str, Any]` | `{}` |
| `prices_version` | `str \| None` | `None` |
| `notes` | `dict[str, Any]` | `{}` |

`write(path)` serialises to indented JSON. There is no dedicated cost field; run totals go
in `notes`.

## Validation (`validate.py`)

All failures raise `ContractError` (a `ValueError`) with a message naming the field and
rule. The mock embodiment validates every observation and target; the RoboLab bridge
validates every target; `rung.probes` and `deploy/robolab/check.py` validate observations.

### `validate_camera(frame, caps=None)`

- `rgb` is `H x W x 3` `uint8`.
- `convention` is in `CAMERA_CONVENTIONS` and is `"opencv"` or `"ros"`; `"opengl"` raises
  with a pointer to `frames.convert_camera_pose`.
- If `depth` is present: shape equals `rgb.shape[:2]`, dtype `float32`, no negative finite
  values, and both `K` and `pose` present.
- If `K` is present: `3x3`, finite, positive `fx` and `fy`, principal point strictly
  inside the image (`0 < cx < width`, `0 < cy < height`).
- If `pose` is present: `pose.pos` is finite.
- With `caps`: the frame size matches the declared `CameraSpec` (unknown names raise
  `KeyError`), and a camera declared with depth delivers depth.

### `validate_observation(obs, caps=None)`

- `obs.t` finite; `scene` and `wrist` (if present) pass `validate_camera`.
- `tcp.pos` finite; `gripper.travel` in `[0, 1]`; `gripper.opening_m`, if present,
  non-negative.
- With `caps`: `workspace.contains(tcp.pos, margin=-0.05)`, allowing the TCP up to 5 cm
  outside the declared box to absorb small TCP-transform or root-frame errors; and
  `scene.t <= obs.t + 1e-6`.

### `validate_target(target, caps)`

- `pose.pos`, `timeout_s` and both speeds are finite; `timeout_s > 0`; both speeds `> 0`.
- `max_lin_speed_mps <= caps.max_lin_speed_mps` and
  `max_ang_speed_rps <= caps.max_ang_speed_rps` (tolerance `1e-9`).
- `"tcp_target"` is in `caps.command_kinds`.
- `gripper.opening_m`, if present, is in `[0, caps.gripper.max_opening_m]`.

## Embodiment protocol (`rung/embodiment.py`)

| Member | Contract |
|---|---|
| `capabilities` | `EmbodimentCapabilities` |
| `reset(episode_seed=None)` | Start an episode; return the first observation |
| `observe()` | Current observation without advancing the world |
| `execute(target)` | Play one target to completion, expiry or supersession; advance simulated time; return an `ExecutionReport`. Normal failures (unreachable, blocked, expired) are outcomes; exceptions are for contract violations and broken rigs. `EpisodeStopped` signals that the external evaluator ended the episode |

`Gradeable` (`grade() -> (success, detail)`, `instruction`) is a separate protocol.
`run_episode` grades only after the policy loop ends; the verdict never reaches a primitive
or a prompt.

## Adapter checklist

- **Convert camera convention.** Pass source poses through `frames.convert_camera_pose`
  and tag frames `"opencv"`. Measure the source convention per simulator tree (for
  RoboLab, `translate.detect_camera_convention`); do not assume it.
- **Apply `tcp_from_ee`.** Compose the reported end-effector pose with it to produce
  `Observation.tcp`.
- **Keep `K` valid.** Use `rescale_intrinsics` / `crop_intrinsics` on resize or crop.
- **Stamp capture times.** Set `CameraFrame.t` and `state_at_capture` at capture, not at
  assembly.
- **Translate `TCPTarget`.** Convert to the embodiment's command space; the core emits
  nothing else.
- **Report honestly.** Return one of the six `ExecutionOutcome` values, residuals computed
  as in `residual_of`, and a `reason` that distinguishes an arm still converging from one
  that is not moving.
- **Declare capabilities from measurement.** Speed caps, `steps_per_efold`, gripper
  geometry and `tcp_from_ee` feed validation, time budgets and grasp planning.
- **Pass `rung.probes`** before any policy runs.

### RoboLab adapter (`rung.adapters.robolab`)

| Aspect | Implementation |
|---|---|
| Threading | `RoboLabEmbodiment` (harness thread) and `RungInferenceClient` (simulator thread) meet at two queues; `run_in_thread` starts the harness loop |
| Observation | `translate.to_observation`: `image_obs` and `proprio_obs` to one `Observation`; camera keys renamed to `scene` and `wrist`; missing geometry raises `KeyError` |
| TCP | `base_link` pose (`ee_pos`, `ee_quat`) composed with `droid_tcp_transform`: offset along `base_link +x` (`FLANGE_TO_FINGERTIP_M = 0.14883` fallback; deployment reads a calibration file), tool frame rotated +90 degrees about `base_link y` |
| Action | `translate.action_chunk`: `(N, 7)` rows of `dx dy dz droll dpitch dyaw grip` on root axes, per-step ceilings 2 cm and 0.10 rad at 15 Hz, divided by `REL_IK_SCALE * tracking_gain` |
| Tracking gain | `tracking_gain` in `(0, 1]`; `RoboLabEmbodiment` defaults to 1.0 (uncompensated), `deploy/robolab/run.py` to the measured DROID value `TRACKING_GAIN = 0.283` (ADR-014) |
| Arrival | Within 6 mm and 0.05 rad, with the gripper settled (full close time or jaw stopped) |
| Stall | No progress of 2 mm or 0.02 rad for 1.5 s reports `blocked` |
| Time | `time_mode="realtime"` (default) holds the robot for the harness's wall time before each target; `"paused"` is a diagnostic (ADR-013) |
| Deployment | `client.RungPolicyClient`: `begin_episode`, `infer_batch`, `reset`, `record_grades` for RoboLab-Verified (ADR-011) |

## Real-robot transfer

Moving to a real embodiment changes camera calibration, TCP calibration, timing, gripper
behaviour, controller dynamics and collision geometry together. Each maps to a declared
field and a check:

| What changes | Field | Check |
|---|---|---|
| Camera calibration | `CameraSpec` per camera; `CameraFrame.K` and `pose` at runtime | `probe_camera_geometry`, `probe_depth_reconstructs_the_table` |
| TCP calibration | `tcp_from_ee` | Independent TCP calibration; `validate_observation` workspace check |
| Timing | `control_period_s`; `CameraFrame.t`, `Observation.t` | `probe_clock`, `probe_timeout_is_honoured` |
| Gripper behaviour | `GripperGeometry`; `GripperState` at runtime | `probe_gripper_states` |
| Controller dynamics | `max_lin_speed_mps`, `max_ang_speed_rps`, `steps_per_efold`; adapter tracking gain | `validate_target`; `probe_commanded_equals_travelled`, `probe_rotation_direction` |
| Collision geometry | `workspace` box and gripper pad and finger dimensions only | No field for arm or gripper-body collision geometry |

The collision-geometry row is a gap in the current contract: an embodiment with different
arm-body geometry has no field in `EmbodimentCapabilities` to declare it.
