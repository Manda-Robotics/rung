# Decisions

Architecture decision records for rung. Entries are dated and numbered. A later decision
that changes an earlier one gets its own number and names the entry it supersedes; the
superseded entry carries a pointer forward.

Several early entries (for example ADR-007 and ADR-013) were written while Rung ran on a
patched RoboLab fork. The public [RoboLab guide](../deploy/robolab/README.md) targets
upstream RoboLab with its unmodified grader; see the [evaluation protocol](protocol.md).

| ADR | Date | Title | Status |
|---|---|---|---|
| [001](#adr-001-new-repository) | 2026-09-09 | New repository | Accepted |
| [002](#adr-002-two-experiment-tracks) | 2026-09-09 | Two experiment tracks | Accepted |
| [003](#adr-003-l2-first-on-gemini) | 2026-09-09 | L2 first, on Gemini | Accepted; tool surface amended by 010 |
| [004](#adr-004-tcp-target-action-contract) | 2026-09-09 | TCP target action contract | Accepted |
| [005](#adr-005-opencv-camera-convention-in-the-core) | 2026-09-09 | OpenCV camera convention in the core | Accepted |
| [006](#adr-006-task-roles-fixed-by-template) | 2026-09-09 | Task roles fixed by template | Accepted |
| [007](#adr-007-one-simulator-configuration-for-the-interface-table) | 2026-09-09 | One simulator configuration for the interface table | Accepted |
| [008](#adr-008-no-dependency-on-inspect-robots) | 2026-09-09 | No dependency on inspect-robots | Accepted |
| [009](#adr-009-heavy-models-as-optional-services) | 2026-09-09 | Heavy models as optional services | Accepted |
| [010](#adr-010-loop-owned-l2-execution) | 2026-09-09 | Loop-owned L2 execution | Accepted; grasp evidence superseded by 012 |
| [011](#adr-011-robolab-verified-deployment-client) | 2026-09-10 | RoboLab-Verified deployment client | Accepted |
| [012](#adr-012-grasp-verification-from-the-vacated-spot-and-the-jaw) | 2026-09-11 | Grasp verification from the vacated spot and the jaw | Accepted |
| [013](#adr-013-real-time-execution-by-default) | 2026-09-14 | Real-time execution by default | Accepted |
| [014](#adr-014-tracking-gain-feed-forward) | 2026-09-15 | Tracking-gain feed-forward | Accepted |
| [015](#adr-015-raced-gemini-requests) | 2026-09-15 | Raced Gemini requests | Accepted |

## ADR-001: New repository

Date: 2026-09-09

**Context.** rung supersedes vlm-pinpoint 0.1.0, a single-interface point-and-phase
controller whose camera convention and relative-IK scaling were embedded in controller
logic.

**Decision.** Start a new repository. Code that still applies is ported (flood-fill object
extents, principal-axis alignment, the container rim model, the gripper state machine);
git history is not carried over.

**Consequences.** Provenance of ported code is documented here, not in `git log`. Claims
made by the predecessor do not carry over with its code.

## ADR-002: Two experiment tracks

Date: 2026-09-09

**Context.** Moving between interfaces changes information, grounding, action vocabulary,
control horizon and recovery at once. A single score difference cannot be attributed to
the model.

**Decision.** Run two tracks.

| Track | Question | Held fixed | Free |
|---|---|---|---|
| Capability | Strongest system per model within a stated development budget | Tasks, sensors, development time and dollars, evaluation protocol | Prompting, effort, tool-use style, self-grounding; chosen on development tasks and documented |
| Controlled | Effect of one interface component | Everything else: shared perception, executor, prompts, settings | The single component under test (interface level, perception source, candidate selection, recovery) |

Differences between L0, L1 and L2 are reported as interface-package effects. Attribution
comes from targeted comparisons on development tasks: shared versus self perception on the
same L2 executor; model-chosen versus geometrically ranked grasp candidates; recovery
enabled versus disabled.

**Consequences.** Every reported comparison names its track. Capability-track and
controlled-track numbers are reported in separate sections and are not interchangeable.

## ADR-003: L2 first, on Gemini

Date: 2026-09-09

**Context.** A vertical slice through one interface and one vendor exposes contract and
executor problems before paid cross-vendor comparisons.

**Decision.** Build L2 (semantic primitives) first, on Gemini 3.7 Flash. L0 and the other
vendors follow. L1 is to be ported onto the current contracts and frozen as a reference.

**Consequences.** Grasp-stack reliability is established with the scripted reference
policy before any model comparison, so model scores do not measure grasp-controller
failures. The L2 tool surface was later extended to full-pose `move_to` (ADR-010).

## ADR-004: TCP target action contract

Date: 2026-09-09

**Context.** Embodiment-specific kinematics inside controller logic makes the core
depend on one rig.

**Decision.** The core emits one action kind, `TCPTarget`: an absolute TCP pose in the root
frame (metres, unit quaternion `(w, x, y, z)`), reached along a straight line at bounded
speed, with a gripper command, per-target speed caps and a timeout. Adapters translate it
into relative IK steps, Cartesian impedance commands or joint trajectories.

**Consequences.** The core contains no embodiment kinematics. Each adapter owns its
translation and is checked with `rung.probes` (commanded versus travelled distance,
rotation direction, camera geometry, gripper states, clock, timeout) before policy runs.

## ADR-005: OpenCV camera convention in the core

Date: 2026-09-09

**Context.** Isaac and USD cameras use OpenGL axes; deprojection written against one
convention is silently wrong for others.

**Decision.** The core uses OpenCV axes (`+x` right, `+y` down, `+z` forward), stated once
in `frames.py`. Adapters convert with `frames.convert_camera_pose` before an observation
reaches the core. `validate_camera` rejects any frame not tagged `"opencv"` or `"ros"`
(identical axes).

**Consequences.** No core code branches on camera convention. A missing conversion fails
at validation instead of producing wrong 3D points.

## ADR-006: Task roles fixed by template

Date: 2026-09-09

**Context.** An object-name or relation swap within one task template is not independent
evidence of generalisation.

**Decision.** Tasks are grouped by template, and templates are assigned to development,
validation or test in `tasks/roles.yaml` before tuning. A test task whose failure
influences a build decision moves to validation, and the move is logged.

**Consequences.** Controller, prompt and perception code never reference validation or
test tasks by name. The test set can only shrink; a task never moves from validation to
test. `tests/test_task_roles.py` enforces the structural invariants.

## ADR-007: One simulator configuration for the interface table

Date: 2026-09-09

**Context.** Numbers from different simulator, asset or grader configurations are not
comparable.

**Decision.** Every cell of the interface table (all models, interfaces and tracks) runs on
one configuration: RoboLab-Verified, patched. Runs on a separate Isaac Sim 6 fork are
labelled comparability runs.

**Consequences.** Table cells are mutually comparable. Fork runs carry a visible label and
are never merged into table rows.

## ADR-008: No dependency on inspect-robots

Date: 2026-09-09

**Context.** inspect-robots (RoboCurve, MIT) exposes SAM 3, Contact-GraspNet and Pyroki IK
to generated policy code. It does not provide a controlled comparison across interfaces on
one task set with shared perception.

**Decision.** rung does not depend on inspect-robots. Four ideas from its agent policy are
adopted and cited: absolute targets, a largest-remaining-offset feedback line after every
motion, a mandatory note on every model call, and a per-trial budget with explicit `done`
and `give_up`.

**Consequences.** No exposure to its release schedule. Citations appear where the ideas are
used: `rung.interfaces.l0`, the note field in L0 and L2 tool schemas, and `rung.loop`.

## ADR-009: Heavy models as optional services

Date: 2026-09-09

**Context.** GPU-bound perception and grasp models make the core test suite slow and
hardware-dependent.

**Decision.** SAM 3, Contact-GraspNet and similar models run as optional, versioned
services behind the same client interfaces as the defaults (model box grounding with
depth clustering for perception, `AnalyticGraspPlanner` for grasps). The core test suite
needs no GPU, simulator or API key.

**Consequences.** `make check` runs on a laptop. A run that uses an optional service
records its version in `RunProvenance.perception_version` or `grasp_version`.

## ADR-010: Loop-owned L2 execution

Date: 2026-09-09

**Context.** Executing primitives inside `act()` hid motions, deadlines and usage from the
episode loop.

**Decision.** `L2Policy.act()` selects a tool without moving the arm. The loop traces the
decision and collects usage, then calls `execute_decision` with its own motion callback,
which bounds each target's timeout by the remaining horizon, checks limits, traces attempt
and execution, and counts motions. Adapter exceptions end the episode as
`embodiment_error`. Usage from malformed responses and perception refreshes is billed. The
acting model receives full-quaternion `move_to` targets (`quat_wxyz`), replacing the
yaw-only form of ADR-003; `yaw_deg` remains as a top-down convenience. Grasp success
requires object-surface matches at two moved poses (superseded by ADR-012).

**Consequences.** Every motion appears in the trace under one accounting path. Full-arm
collision planning, identity-proof attachment detection and cancellation of in-flight
model requests are not provided.

## ADR-011: RoboLab-Verified deployment client

Date: 2026-09-10

**Context.** RoboLab-Verified owns the episode loop, stepping and grading; the harness
owns its own blocking loop.

**Decision.** `rung.adapters.robolab.client.RungPolicyClient` implements Verified's
`begin_episode`, `infer_batch` and `reset` contract with one physics action per call and
fresh policy, memory, bridge and trace state per episode. The harness runs in a thread
beside the simulator and meets it at a pair of queues (`bridge.py`). The relative-IK
action body is `base_link` (`ee_pos`/`ee_quat`); `eef_quat` carries a separate rotation
and is never substituted. The TCP offset comes from a calibration file, with
`FLANGE_TO_FINGERTIP_M = 0.14883` as the fallback. Box grounding uses normalised 0 to 1000
`[y0, x0, y1, x1]` coordinates regardless of image size.

**Consequences.** Policy completion leaves a hold command until Verified ends the trial.
Results are ungraded (`success: null`) until `record_grades` joins the evaluator's verdict
after all policy threads have stopped. L0 gripper commands apply on pose arrival, and their
settling time shares the target deadline.

## ADR-012: Grasp verification from the vacated spot and the jaw

Date: 2026-09-11

Supersedes the grasp-evidence rule of ADR-010.

**Context.** Requiring object-surface matches at two moved poses fails when the grasp
works: from the scene camera, a held object is behind the hand holding it.

**Decision.** `grasp` reports `grasped` when the object's former location is measured at
least 2 cm lower after the lift (stepping aside if the reading from above is inconclusive)
and the jaw is still stopped short of its closed-on-nothing threshold. Surface matches
grade confidence (`confirmed`, `corroborated once`, `inferred`) rather than veto. Holds
thinner than 1.5 cm do not use jaw travel as evidence. An unmeasurable former location
with a jaw that stayed partly closed through the lift, and no contradicting surface match,
reports `grasped` with confidence `jaw only`; otherwise it reports `unknown`, which requires
a release before another grasp.

**Consequences.** Verification is observation-based and never reads the grader. It is
geometric evidence, not ground-truth attachment. `place` re-reads the jaw during transport
and reports `dropped` when the object has fallen out.

## ADR-013: Real-time execution by default

Date: 2026-09-14

**Context.** The simulator blocks while the harness perceives, calls the model and plans.
If that time is free, results overstate what a real robot would achieve: a paused diagnostic
run scored 4 of 6 on a task set where real time scored 2 of 6.

**Decision.** The RoboLab adapter defaults to `time_mode="realtime"`. Wall time between
motions, excluding recording writes (`exclude_wall`), is played out as robot hold steps at
the control period before the next target starts, and is reported per target as
`ExecutionReport.waited_s`. `time_mode="paused"` is a labelled diagnostic and is not
reported as a result.

**Consequences.** Harness and model latency cost simulated horizon. Latency reductions
(catalog reuse at an unchanged instant, box re-measurement after a verified grasp, JPEG
image encoding, raced model requests) directly affect task outcomes. `EndReason.WALL_CLOCK`
names a real-time runaway guard, not a model inference budget.

## ADR-014: Tracking-gain feed-forward

Date: 2026-09-15

**Context.** The DROID relative-IK controller (damped least squares) realises a fixed
fraction of each commanded Cartesian step, 0.283 measured identically on Isaac Sim 5.1
and 6.0, so uncompensated motion converges geometrically at about 30 control steps per
e-fold.

**Decision.** `action_chunk` divides commanded deltas by `REL_IK_SCALE * gain`. The gain is
set per run with `--tracking-gain` (`RoboLabEmbodiment.tracking_gain`, in `(0, 1]`). The
RoboLab runner defaults to the measured DROID value `translate.TRACKING_GAIN = 0.283`; the
embodiment itself defaults to 1.0 (uncompensated) for other controllers.
`deploy/robolab/calibrate_speed.py` measures it. Motion timeouts inside primitives scale
with distance through `EmbodimentCapabilities.time_to_converge`.

**Consequences.** The gain is recorded in the trace `setup` event and belongs in run
provenance. A wrong gain overshoots or undershoots; the movement probes detect either.

## ADR-015: Raced Gemini requests

Date: 2026-09-15

**Context.** Gemini request latency has a heavy tail (median 6.0 s, one in six above 12 s,
worst 61 s in 24 paired grounding requests), and paired requests stall nearly
independently. Under ADR-013 a stalled call is time the robot stands still.

**Decision.** `GeminiBackend` races identical requests: two from the start
(`hedge_after_s=0`) and a third after 8 s (`rehedge_after_s=8`), each on its own client.
The first successful response wins; losers finish in daemon workers and are discarded.
`hedge_after_s=None` sends a single request.

**Consequences.** Every request sent is counted in `ModelUsage.calls` and billed at the
winning response's token usage. Races cannot be cancelled once sent, so the dollar cap can
be exceeded by in-flight requests.


