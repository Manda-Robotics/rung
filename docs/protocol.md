# Evaluation protocol

Task roles, matched starts, time, budgets, controls, statistics, reporting, the failure
taxonomy, and the boundary between simulator ground truth and policy inputs. The protocol
applies to both experiment tracks (ADR-002) unless a section states otherwise.

## Task roles

Templates, not individual tasks, are assigned to roles, because an object-name or relation
swap within a template is not independent evidence (ADR-006). The assignment is committed
in `tasks/roles.yaml`; `tests/test_task_roles.py` enforces its structure.

| Role | Tasks | Use |
|---|---|---|
| Development | 14: tasks the predecessor controller was tuned on or failed, plus geometry and failure cases (tall bottle, wide object with a narrow part, cluttered container, relational placement) | Building controllers, prompts and perception; component interventions. Never reported as results. |
| Validation | 15, template-disjoint | Choosing effort, prompt variants, perception backend and grasp configuration per model. Reported and labelled as validation. |
| Test | 30, template-disjoint, untouched until configuration freeze | Paired comparison across models and interfaces. |
| Full suite | All 120 | Aggregate including development and validation tasks, labelled as such. Never described as held out. |

Two large templates (`put_single_in_container`, `put_all_of_kind_in_bin`) are split across
roles by object identity, as declared in `tasks/roles.yaml`.

**Contamination rule.** If a test-task failure influences a build decision (prompt,
perception, grasp candidates), the task moves to validation and the move is logged with
date and reason. A task never moves from validation to test. Effort and prompt sweeps run
on validation tasks only, one model at a time.

## Matched starts and simulator configuration

Matching by task, environment count and run index is a candidate for comparability, not
a guarantee. The protocol requires each run to save an initial-state manifest and audit it
against a reference with a SHA-256 digest over all non-camera state
(`RunProvenance.initial_state_audit`). A mismatch is labelled, not dropped, so systematic
mismatches stay visible. The RoboLab runner does not yet perform this audit and records
initial-state matching as unaudited; such runs make no matched-state claim.

Controlled interface comparisons use one declared simulator, asset and grader
configuration. The original comparison protocol used RoboLab-Verified (ADR-007).
The public [deployment guide](../deploy/robolab/README.md) instead targets unmodified
upstream RoboLab. Record its revision and result class, and keep runs on different
simulator trees separate.

## Time

Controlled interface comparisons use real-time execution (ADR-013). With `time_mode="realtime"`, wall
time the harness spends between motions (perception, model calls, planning) is played out
in simulation as robot hold steps before the next target, and each `ExecutionReport`
records it as `waited_s`. Recording writes are excluded. `time_mode="paused"` lets the
world wait while the model thinks. It is used for diagnostics and for upstream RoboLab
submissions that follow the benchmark's synchronous timing. Label those runs with
`result_class="leaderboard_submission"`, record inference latency separately, and
compare them only under the same timing rules.

The tracking gain (`--tracking-gain`, ADR-014) and camera convention are recorded in each
episode's `setup` trace event and are part of a run's configuration.

## Budgets

One L2 decision can execute a full grasp; one L0 decision moves the arm once. Decision
counts and cost are therefore not comparable raw across interfaces. Every run states its
limits (`rung.loop.Limits`), and every episode records which one ended it
(`EpisodeResult.end_reason`).

| Limit | Field | End reason | Enforcement |
|---|---|---|---|
| Simulated horizon | `horizon_s` | `simulated_horizon` | Checked between operations; target timeouts are bounded by the remaining horizon |
| Decisions | `max_decisions` | `decision_limit` | Checked before each model turn |
| Model calls | `max_model_calls` | `model_call_limit` | Optional native-policy cap, including rejected replies and failed attempts |
| Dollars | `max_dollars` | `dollar_cap` | Checked between decisions and motions; an in-flight request can cross the cap and is still billed |
| Wall clock | `max_wall_s` | `wall_clock_limit` | Runaway guard over model calls and simulation together; cannot interrupt a blocking request |
| Consecutive policy errors | `max_consecutive_errors` | `policy_error` | Malformed responses retried until the limit |

Other end reasons: `policy_done`, `policy_gave_up`, `external_episode_end` (the evaluator
ended the episode during a command), and `embodiment_error`. `EpisodeResult.servo_steps`
counts completed target executions, not physics ticks. Re-issues of one motion inside a
primitive are bounded by `MOTION_RETRIES`. A limit on repeated calls of the same primitive
and a spending cap across episodes are protocol requirements, not enforced by the loop.

Success is reported against cost and elapsed simulated time as the primary view, with an
equal-decision-count cut as a secondary view.

## Controls

| Control | Definition | Compared against |
|---|---|---|
| Policy-image ablation | No image; object catalog still provided | Scripted and randomised references |
| Scene-information ablation | No image and no catalog | Scripted and randomised references |
| Scripted reference | Hand-written pick and place with reference object choices (`rung.reference.ScriptedReference`, mock embodiment) | Reported as what the primitives can do, not an upper bound |
| Randomised reference | Random actions through the same executor (`rung.reference.RandomReference`: random reachable top-down pose and gripper action, mock embodiment) | Prices the executor's contribution |
| Equivariance | Image-coordinate equivariance on the perception service; physically randomised layouts across matched episodes | |

Both references read simulator truth and carry the `scripted` or `random` interface
label; they are diagnostics, not policies under test. Mirrored images are not used as a
control: a mirrored image is not a valid robot-frame transform. Ablations are compared
with the executable references, not with a notion of chance.

## Statistics and reporting

- Ten episodes per task.
- Paired differences across matched starts (same initial-state manifest).
- Task-macro averages: each task contributes equally.
- Template-grouped uncertainty: tasks sharing a template are not independent draws.
- Denominator policy: timeouts, give-ups, malformed responses and infrastructure failures
  stay in the denominator. `done` ends the policy's turn; the benchmark grader decides
  success.

### Reporting row

| Field | Meaning |
|---|---|
| model | Model id, pinned version |
| effort | Effort or reasoning level |
| interface | Direct, L0 or L2; L1 is planned |
| track | Capability or controlled |
| perception source | Shared perception service, or the model's own grounding (capability track only) |
| declared inputs | Images, catalog, memory, ablations applied |
| time mode | `realtime` for controlled comparisons; `paused` for labelled diagnostics or upstream benchmark submissions |
| decisions | Model decisions, excluding motions (see the servo flag in `docs/contracts.md`) |
| model calls | `ModelUsage.calls`, including raced requests and grounding calls |
| tokens | Input, output and thinking tokens, separately |
| dollars | From the versioned price table (`rung.backends.prices.VERSION`, recorded as `RunProvenance.prices_version`) |
| sim seconds, wall seconds | `EpisodeResult.sim_seconds`, `wall_seconds` |
| ending limit | Distribution of `end_reason` over the row's episodes |
| failure category | From the taxonomy below, for unsuccessful episodes |

## Failure taxonomy

Applied to every unsuccessful episode.

| Category | Definition |
|---|---|
| grounding | Wrong object identified or localised, or a present object not identified |
| infeasible grasp | No viable grasp for the object geometry, before collision checks |
| collision rejection | A motion refused because it would collide (table, neighbour, wrist, arm) |
| failed attachment | Gripper closed as planned without capturing the object (`missed`, not `closure_obstructed`) |
| dropped object | Verified hold lost before placement (`dropped`) |
| placement | Object released away from the intended target or relation |
| sequencing | Primitives succeeded but their order or combination did not accomplish the task |
| timeout | A budget limit ended the episode before completion |
| infrastructure | Failure in harness, adapter, backend or service, not in policy or primitives |
| unknown | No category fits, or the trace lacks the information to decide |

## Diagnostic boundary

Simulator ground truth (exact object poses, contact state, anything a real embodiment
would not report) is reachable only through a separate diagnostic boundary, and only for
post-hoc failure localisation after an episode ends. It never feeds a primitive result or
a prompt, and evaluator predicates never reach the policy. The `Gradeable` protocol is
separate from `Embodiment`, and `run_episode` calls `grade()` only after the policy loop
has finished. Primitive results are computed from what the executor observed (jaw travel,
depth, motion reports), so a real embodiment adapter can produce the same results.
