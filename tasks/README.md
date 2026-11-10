# tasks/roles.yaml

Assigns each RoboLab-120 task to exactly one evaluation role.
`robolab120_tasks.txt` is the full manifest for `deploy/robolab/suite.py`.

## Roles

- **development**: tasks used to build and debug the harness, chosen to
  cover known geometry and failure modes (tall bottle, wide-footprint
  object with a narrow part, cluttered container, relational placement,
  removal direction, OR-language). 14 tasks.
- **validation**: used to choose effort, prompts, perception and grasp
  configuration. 15 tasks.
- **test**: untouched until configuration freeze, then run once. Spans
  the same difficulty mix as validation, plus a few clearly tagged
  out-of-scope probes (stacking, counting). 30 tasks.
- **full_only**: everything else, only run in the full 120-task
  aggregate. Each carries an `in_scope` flag and `scope_note` explaining
  why it is out of scope (counting, multi-destination sorting, stacking,
  reorientation) or in scope but simply unassigned. 61 tasks.

## The template rule

Tasks are grouped by template first (verb, object count, target type,
relation), because swapping an object name or a spatial relation inside
the same template is not independent evidence of generalization. A
template's tasks mostly sit in one role. The two large
single-object-into-container templates are the deliberate exception:
split across development, validation and test by object identity, so
every role sees different objects on the same skill.

## Changing an assignment

Edit `tasks/roles.yaml` directly. Moving a task from test into validation
(or development) must be logged in `docs/decisions.md` with the date and
reason, since test is supposed to stay untouched until configuration
freeze. Moving a task the other way, into test or full_only, needs no log.

## astranomy30_tasks.txt

The 30-task ablation suite from NVIDIA's Astranomy report (Table 1,
https://research.nvidia.com/labs/srl/projects/astranomy/), mapped one to one onto
RoboLab-120 task classes on 2026-10-22. Their direct-policy ablations ran 5 episodes per
task on this subset; Rung's Day 2 subset run uses it at seeds 0 to 4.
