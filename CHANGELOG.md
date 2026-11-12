# Changelog

## 0.1.1 (unreleased)

### Added

- **Two arms.** RoboDojo's dual ARX X5 rig (`rung.adapters.robodojo`) on Isaac Sim 5.1,
  through RoboDojo's stock XPolicyLab policy server, graded by RoboDojo's own `run_reward`.
  The shared two-arm bridge (`rung.adapters.bimanual`) turns per-arm TCP targets into
  per-step setpoints and reports `reached`, `blocked` or `expired` for each arm.
- The bimanual Direct profile (`DirectConfig(bimanual=True)`) is now usable end to end:
  `configs/gemini-4-argon-robodojo.json`, per-arm tools, `move_both`, per-arm receipts.
  Single-arm requests are unchanged byte for byte.
- `deploy/robodojo/`: pinned install script, one-episode runner and the policy-server shim,
  with a [deploy guide](deploy/robodojo/README.md).
- `rung.adapters.robodojo.fake.FakeRoboDojo`: a CPU stand-in with RoboDojo's real
  observation and action dicts, and `examples/robodojo_mock.py`.
- The fake model has a two-arm script for bimanual profiles.

## 0.1.0 (2026-11-10)

First public release: the Direct, L0 and L2 interfaces, the CPU tabletop mock, RoboLab
DROID on Isaac Sim 6.0.1, the Gemini, Claude and OpenAI backends, and the remote API client.
