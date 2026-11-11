# Rung documentation

Start with [installation](../README.md#installation), then run an offline mock episode.
The guides below render directly on GitHub and are included in source distributions.

| Guide | What you will find |
|---|---|
| [Quickstart](quickstart.md) | Offline runs, paid model setup, budgets and output files |
| [Examples](../examples/README.md) | Offline examples, a remote API client, and recording inspection/export |
| [Interfaces](interfaces.md) | Direct control, L0 pose targets and L2 semantic primitives |
| [Bring your own model](own-model.md) | The model seam and a backend for OpenAI-compatible servers (vLLM, SGLang, Ollama) |
| [Contracts and adapters](contracts.md) | Frames, units, observations, actions, traces and adapter checklist |
| [Evaluation protocol](protocol.md) | Task roles, timing, controls, budgets and reporting |
| [RoboLab deployment](../deploy/robolab/README.md) | Single-GPU setup, calibration, probes and task suites |
| [Architecture decisions](decisions.md) | Design rationale and historical changes |
| [Contributing](../CONTRIBUTING.md) | Development setup, tests, formatting and coverage |

## How an episode works

An embodiment exposes calibrated observations and capabilities. A policy reads the
observation and previous execution feedback, then returns a decision. The episode loop
checks budgets, sends bounded targets to the embodiment and records the reports. Once
the policy ends, the embodiment's grader determines success.

The built-in mock checks this flow on a CPU. RoboLab supplies contact physics, rendered
cameras and its own task scorer. A new hardware adapter implements the same
[`Embodiment` protocol](contracts.md#embodiment-protocol-rungembodimentpy).

The public package includes the remote API client and wire codec under `rung.api`.
Direct, L0 and L2 policies can run locally using provider API keys.
