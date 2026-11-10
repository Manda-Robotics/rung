# Interfaces

An interface level fixes which decisions the model makes and which the harness makes: what
to act on, which primitive and when, where in 3D and in what pose, and how to move there.
Changing level changes information, grounding, action vocabulary, control horizon and
recovery together, so a score difference between levels is an **interface-package
effect**, not an isolated model effect (ADR-002).

| Level | Status | What to act on | Which primitive, when | Where in 3D, what pose | How to move there |
|---|---|---|---|---|---|
| Direct, bounded tool calls | Implemented (`rung.interfaces.direct`) | Model, from RGB views | Model, by tool or guarded sequence | Model, using camera geometry and optional geometry tools | Harness bounds targets and reports measured execution |
| L2, semantic primitives | Implemented (`rung.interfaces.l2`) | Model, by object id | Model, by tool | Harness proposes grasp candidates; model may choose one or give a pose | Harness: plan, check, execute, verify |
| L1, point and phase | Planned | Model, by pixel | Model, by phase word | Harness: deproject, extents | Harness: servo, watchdogs |
| L0, pose targets | Implemented (`rung.interfaces.l0`) | Model | Model | Model: metres and yaw | Harness: interpolate, clamp |

Both implemented levels run through the same episode loop (`rung.loop.run_episode`), the
same `Embodiment` protocol and the same backends. Every tool call carries a mandatory
`note` (or `summary` / `reason` for the terminal tools), and the episode ends only on an
explicit `done`, `give_up`, or a limit (ADR-008).

## Direct: bounded tool calls

`rung.interfaces.direct.DirectPolicy` uses a native model session. Each turn supplies
camera images, measured TCP pose, gripper state, workspace limits and receipts from
earlier tools. The session preserves tool-call IDs and resolves every outstanding call
before requesting another decision. Camera geometry, recent frames and extra views
are configurable through `DirectConfig` and a JSON profile.

| Tool | Behavior |
|---|---|
| `move_relative` | Translate in root-frame metres and rotate by a rotation vector in radians |
| `move_to` | Move to an absolute root-frame position and unit quaternion in `w, x, y, z` order |
| `set_gripper` | Open or close at the current measured TCP |
| `observe` | Request another observation without motion |
| `wait` | Hold pose while physics advances, within the simulated horizon |
| `done`, `give_up` | End the policy or declare completion; the external grader determines success |

The profile selects relative motions, absolute motions or both. Optional tools include
`project_table` (pixel ray to the table plane), `triangulate_point` (matched pixels in
calibrated RGB views), `probe_point` (calibrated depth), `look` (bounded viewpoint
motion), `move_relative_tcp` (TCP-frame motion) and `execute_sequence` (bounded motion,
gripper and wait steps). The authoritative schemas are `rung.interfaces.direct.tools`.

Translations, rotations, speed and timeout are bounded before execution. Each attempted
motion receives a measured report. A guarded sequence stops early when a motion does
not reach its target or a close ends with an empty gripper. The next model turn sees
receipts for the steps actually executed.

The public Gemini profile is [`configs/gemini-4-argon-direct.json`](../configs/gemini-4-argon-direct.json).
Geometry tools expose their declared sensor inputs; using depth queries changes the
information available to a policy. State these settings in evaluation reports.

## L2: semantic primitives

### Model input per turn

`L2Policy.build_request` assembles one user message:

| Block | Source | Condition |
|---|---|---|
| Task instruction | Embodiment | Always |
| Scene image | `Observation.scene.rgb`, JPEG quality 92 | Always |
| Scene image with each catalogued object marked by id | `mark_objects` | `send_marked_image` and a non-empty catalog |
| Wrist image | `Observation.wrist.rgb` | `send_wrist` and a wrist camera present |
| Object catalog, one line per object: id, label, position, size, top height, width at grasp height, uncertainty | `Catalog.render` | Always |
| Perception failures | `Catalog.failures` | When present |
| Memory: probably-held object, objects not visible, last six attempts | `ObjectMemory.render` | Always |
| Robot time used of the horizon | `L2Config.horizon_s` | When set |
| Result of the last call, or why the last reply could not be used | `PrimitiveResult.render` | After the first turn |

Both the raw and the marked scene image are sent because a mark can occlude the object it
labels. The prompt is rebuilt from catalog and memory each turn (`history_turns=0`).

### Tool surface

The authoritative schemas are `rung.interfaces.l2.tool_specs()`; `schema_json()` exports
them. All tools except `done` and `give_up` require `note`.

| Tool | Arguments | Behaviour | Moves the arm |
|---|---|---|---|
| `inspect` | `object_id` | Object description, visible fraction, uncertainty and rendered grasp plan | No |
| `propose_grasps` | `object_id` | Feasible candidate ids with jaw opening and clearance, plus rejection reasons | No |
| `grasp` | `object_id`, optional `candidate_id` | Approach, descend, close, lift, verify; best candidate if none given | Yes |
| `place` | `target_id`, `relation` in `in`, `on`, `left_of`, `right_of`, `in_front_of`, `behind` (default `in`) | Rise, carry, release above the target (`in`, `on`) or at a free table spot on that side | Yes |
| `move_to` | `x`, `y`, `z`; optional `quat_wxyz` or `yaw_deg` | Absolute TCP target in root-frame metres | Yes |
| `release` | none | Open the gripper in place and clear the held object | Yes |
| `observe` | optional `camera` | Rebuild the catalog from the current observation | No |
| `done` | `summary`, optional `hindsight` | End the policy; the grader decides success | No |
| `give_up` | `reason`, optional `hindsight` | End the policy with a reason | No |

`quat_wxyz` is a finite unit quaternion mapping TCP coordinates into root coordinates and
cannot be combined with `yaw_deg`. Without it, `move_to` uses a top-down orientation at
`yaw_deg` (default 0). Relations are in the robot's frame: `left_of` is `+y`, `in_front_of`
is toward the base. `move_to` does not imply full-arm collision planning.

`observe` reads the embodiment's current observation without advancing simulated time; it
does not force a new sensor capture, and the `camera` argument does not switch the
grounding camera.

### Execution

1. `act()` builds the catalog, calls the backend, and parses one tool call into a
   `Decision` without moving the arm. If no simulated time has passed since the catalog
   was built (a primitive ends with a look), that catalog is reused.
2. Malformed calls (unknown tool, missing required argument, invalid pose) raise inside
   `act()`. The loop feeds the error back and retries the turn, up to
   `Limits.max_consecutive_errors`.
3. The loop traces the decision and collects usage, then calls
   `execute_decision(decision, execute_target)`.
4. Every motion inside a primitive goes through the loop's `execute_target`, which checks
   limits, bounds the target timeout by the remaining horizon, and traces `attempt` and
   `execution` events.
5. A motion that expires while still making progress is re-issued (up to
   `MOTION_RETRIES`); one that makes no progress raises `MotionIncomplete` and the
   primitive returns `blocked`. `KeyError`, `ValueError` and `TypeError` inside a primitive
   return `rejected` with the error as reason. Adapter exceptions end the episode as
   `embodiment_error`.
6. The result is emitted as a `primitive` trace event and rendered into the next prompt.

Grounding calls made during primitives are billed to the episode. After a verified grasp
the previous detection boxes are re-measured in fresh depth instead of calling the
grounding model again.

### Grasp verification

`grasp` reports `grasped` only on observed evidence (ADR-012):

| Step | Check | Result if it fails |
|---|---|---|
| Close | Jaw travel stops short of the closed-on-nothing threshold (skipped for holds under 1.5 cm) | `missed` |
| Lift 5 cm | Hand rises at least 70 percent of the requested lift | `closure_obstructed` (hand releases and retreats) |
| Former location | Surface at the old grasp point is at least 2 cm lower; step aside and re-measure if inconclusive | `closure_obstructed` if the object stayed |
| Jaw after moving | Jaw still stopped, not run shut | `missed` (object knocked aside) |

Object-surface matches below the fingertips at the lifted and displaced poses grade
confidence (`confirmed`, `corroborated once`, `inferred`) without vetoing. If the former
location cannot be measured, a jaw that stayed partly closed through the lift gives
`grasped` with confidence `jaw only`; otherwise the result is `unknown`, and a release is
required before another grasp. This is geometric evidence from observations, not
ground-truth attachment. `place` re-reads the jaw during transport and returns `dropped`
when the object has fallen out. The external grader is authoritative for task success.

### Primitive results

Primitive results (`rung.executor.primitives.Outcome`) are distinct from the adapter-level
`ExecutionOutcome`; one primitive spans several `TCPTarget` executions and checks.

| Outcome | Meaning |
|---|---|
| `grasped` | Verified hold (see above) |
| `unknown` | Holding something whose identity could not be confirmed; release required |
| `closure_obstructed` | Fingers stopped on something that did not come with the hand |
| `missed` | Grasp executed without capturing the object |
| `rejected` | Refused before execution (no feasible candidate, uncertain hold, malformed call) |
| `blocked` | A motion stopped short without progress |
| `placed` | Object released at the target relation |
| `dropped` | Object fell out of the hand during transport |
| `not_holding` | `place` called with nothing held |
| `moved` | `move_to` or `release` completed |
| `observed` | `inspect`, `propose_grasps` or `observe` completed |
| `clearance_violated` | Reserved for a broken clearance constraint; not currently emitted |

Results and reasons populate the `primitive` trace event (`docs/contracts.md`) and the
failure taxonomy (`docs/protocol.md`).

### Object memory

`rung.perception.memory.ObjectMemory` is part of the interface and is rendered into every
prompt.

| Mechanism | Rule |
|---|---|
| Stable ids | Greedy nearest-centroid match within a label; match distance is the larger of 6 cm and half the object's size |
| Label changes | A detection within 3 cm of a tracked object keeps its id under a different label |
| Hand occlusion | Match tolerance widens for objects within 15 cm of the hand |
| Held object | Matches up to 15 cm, positioned from the gripper |
| Placed object | Relocated in memory before the next look, so it keeps its id when out of view |
| Attempt history | Every primitive call with object, result and reason |

### What L2 measures

Object identification and selection against the catalog, primitive sequencing, candidate
selection when `propose_grasps` is used, and recovery after `missed`, `rejected`, `blocked`
or `closure_obstructed`. Component interventions (shared versus self perception,
model-chosen versus ranked candidate, recovery on or off) run at this level on development
tasks.

## L0: pose targets

`rung.interfaces.l0`. The model does all metric reasoning from images; the harness only
interpolates and clamps. The interface is aligned in shape with RoboCurve's open agent
policy (ADR-008). Scores are not comparable with their real-arm numbers because simulator,
rig, cameras and grader differ.

### Model input per turn

- Scene image, and wrist image if present and `send_wrist` is set.
- Fingertip position (metres, root frame), jaw yaw in degrees, and gripper state
  (`describe_gripper`: open, closed on nothing, or probably holding).
- Reachable box from `EmbodimentCapabilities.workspace` and maximum jaw opening.
- Feedback on the previous motion from `rung.loop.build_feedback`, or the reason the last
  reply could not be used.

### Tool surface

| Tool | Arguments | Behaviour |
|---|---|---|
| `move_to` | `x`, `y`, `z`; optional `yaw_deg`, `gripper` in `open`, `close`, `hold` (default `hold`) | One top-down `TCPTarget`; the gripper command applies on arrival |
| `observe` | optional `camera` | No motion; next turn sees the current observation |
| `done` | `summary` | End the policy |
| `give_up` | `reason` | End the policy with a reason |

Targets use `L0Config.max_speed_mps` (default 0.10 m/s, capped by the embodiment),
angular speed up to 1.0 rad/s (capped), and `L0Config.timeout_s` (default 10 s).

### Motion feedback

After each motion the next prompt carries one line per executed target:

```
The move {outcome}. largest remaining offset from the requested target is {cm:.1f} cm on {axis}. rotation is {rad:.2f} rad short. {reason}.
Gripper: {state}.
```

The offset clause appears only when the outcome is not `reached`; the rotation clause only
when the rotation residual exceeds 0.02 rad. `axis` is whichever of `x`, `y`, `z` has the
largest absolute error (`ExecutionReport.residual_axis`), not a combined distance.

### What L0 measures

Metric spatial reasoning from images, unaided: whether the model can propose reachable
targets and close a pick-and-place loop from proprioceptive feedback alone.

## L1: point and phase (planned)

Not implemented. L1 is intended as a frozen reference ported from the predecessor
controller (ADR-001): the model returns a pixel `(u, v)` and a phase word (`approach`,
`descend`, `close`, `lift`, `retreat`); the harness deprojects the pixel with the scene
camera's depth, `K` and pose (`frames.deproject`), estimates extents, and servos with its
own watchdogs. Two legacy behaviours are confined to L1: a 2 cm interior shift along the
deprojection ray, and a single travel threshold for "holding". Neither is used by L0 or L2.

## Deferred: code as policy over L2

Not implemented. A code-as-policy layer would call the L2 tools (`inspect`,
`propose_grasps`, `grasp`, `place`, `move_to`) from generated code instead of one tool call
per model turn.
