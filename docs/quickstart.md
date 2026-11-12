# Quickstart

Complete [installation](../README.md#installation) and activate the environment first.
The mock examples work on a CPU without Isaac Sim or model credentials.

## Run an offline episode

```sh
rung --scene banana_in_bowl --policy scripted --out runs/first
rung-probes --embodiment mock
```

The scripted reference reads mock object poses and executes a pick and place. It should
report `1/1 episodes solved`. The probes check camera geometry, frame conventions,
gripper state, motion and timing. The mock has metric depth and occlusion but no contact
physics; it validates control logic rather than physical grasp stability.

To exercise the model-tool loop with no API request:

```sh
python examples/direct_mock.py
```

This fake model follows a fixed script and ends with `policy_done`, without solving the
task. It prints synthetic metered usage to exercise cost accounting; it incurs no charge.
The equivalent `rung --policy direct --model fake` command exits 1 because CLI exit codes
reflect task success, even when the loop completed correctly.

## Run two arms on a CPU

RoboDojo's dual ARX X5 rig has a CPU fake with the simulator's real observation and
action dicts. This drives it through RoboDojo's own call sequence with the bimanual
Direct profile and the fake model, so per-arm tools, `move_both` and per-arm receipts
run end to end with no simulator or key:

```sh
python examples/robodojo_mock.py
python examples/robodojo_mock.py --policy scripted
```

`scripted` is the two-arm motion check that the [RoboDojo guide](../deploy/robodojo/README.md)
runs first on a GPU host: each arm alone, both together, then each gripper closes.

## Use a model

The direct Gemini route uses the standard library's HTTP client, so no SDK extra is
needed. Set your own key and choose explicit per-episode limits:

```sh
export GEMINI_API_KEY="your-api-key"
rung --scene banana_in_bowl --policy direct --model gemini-4-argon \
    --profile configs/gemini-4-argon-direct.json \
    --max-decisions 12 --max-dollars 0.50 --horizon 120 --out runs/direct
```

The model ID must be available to your account. The profile pins camera and tool
settings; inspect it before a run. To change models, choose an ID recorded in
[`src/rung/backends/prices.py`](../src/rung/backends/prices.py) and pass `--model`.
Unknown prices are rejected instead of being treated as zero cost.

For L0 or L2, install the provider SDK extra (`gemini`, `anthropic` or `openai`). For example:

```sh
uv pip install -e ".[gemini]"
rung --scene banana_in_bowl --policy l2 --model gemini-4-argon \
    --max-decisions 12 --max-dollars 0.50 --out runs/l2
```

Provider credentials are read from `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY`. Calls to real models incur provider charges. Dollar limits are checked
between operations and cannot cancel an in-flight request, which can cross the cap.
The mock's featureless boxes make it unsuitable for evaluating visual grounding quality.

## Read a run

| Path under `--out` | Contents |
|---|---|
| `trace.jsonl` | Decisions, execution reports, primitive results and episode end events |
| `episodes/<id>/` | Camera images, depth, state, prompts when available and `result.json` |
| `provenance.json` | Revision, model, interface, budgets and price-table version |
| `results.json` | Human-readable episode summaries |

Some native sessions also archive full provider requests and responses when a trace
directory is configured. Keep recordings outside version control.

```sh
rung-inspect runs/first --out runs/first/inspection.json
```

`rung-inspect` reads existing traces and indexes decisions, errors and related frames.
It does not run a policy or grade a task.

## Next steps

- Try the [remote API example](../examples/README.md#try-a-provisioned-rung-api) with a
  provisioned endpoint, or inspect and export the offline recording first.
- Read the [interface guide](interfaces.md) to choose Direct, L0 or L2, and for the
  two-arm profile.
- Use the [RoboLab guide](../deploy/robolab/README.md) for a GPU benchmark.
- Use the [RoboDojo guide](../deploy/robodojo/README.md) for two arms in Isaac Sim 5.1.
- Follow the [adapter checklist](contracts.md#adapter-checklist) for a new embodiment.
- See [contributing](../CONTRIBUTING.md) for tests, formatting and coverage.
