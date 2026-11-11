# Examples

Run these from the repository root after [installation](../README.md#installation).
The first two examples use only the core dependencies, run on a CPU, and make no API requests.

| Example | Run | What it shows |
|---|---|---|
| [Scripted episode](mock_episode.py) | `python examples/mock_episode.py` | Create an embodiment, apply limits, run a pick-and-place policy and read the result |
| [Direct control with a fake model](direct_mock.py) | `python examples/direct_mock.py` | Model session, tool calls, measured receipts and usage accounting |
| [Your own model](own_model.py) | `python examples/own_model.py --selftest` | A complete backend for any OpenAI-compatible endpoint, run on the mock; `--selftest` needs no model |
| [Remote API episode](remote_episode.py) | `python examples/remote_episode.py --out runs/remote` | Connect a local environment to a provisioned API and save observations/actions locally |
| [Inspect and export](inspect_recording.py) | `python examples/inspect_recording.py runs/example --export runs/episode.zip` | Read a recording and export original files with SHA-256 checksums; standard library only |
| [One RoboLab episode](robolab_episode.py) | `python examples/robolab_episode.py --out /out/first` | Inside the GPU container: calibrate, probe, run one Gemini task, and keep native recordings |
| [Read native controls](read_robolab.py) | `python examples/read_robolab.py PATH/TO/run_0.hdf5` | Inspect dense actions and robot state after a RoboLab episode; requires h5py, no GPU |

To save the scripted episode's images, state and trace:

```sh
python examples/mock_episode.py --out runs/example
rung-inspect runs/example --out runs/example/inspection.json
```

Use a fresh output directory for each recorded example. Generated files are ignored by
Git. The direct example prints a task failure and exits zero when its expected five-call
script completes: it tests plumbing and does not attempt a successful pick and place.

For a paid model run, see the [quickstart](../docs/quickstart.md#use-a-model).
For a GPU benchmark, follow the [RoboLab guide](../deploy/robolab/README.md).

## Inspect a real recorded episode

Download and unzip the [worked RoboLab recording](https://github.com/Manda-Robotics/rung/releases/download/v0.1.0/example-robolab-episode.zip).
It contains one native success, 75 aligned control steps, videos, model requests and
responses, observations and execution receipts. Its README records timing, usage,
contact events and a rejected motion request.

```sh
uv pip install h5py
python examples/read_robolab.py robolab-example/episode/BananaInBowlTask/run_0.hdf5
```

This inspection needs no GPU or API key. The HDF5 contains controls and state; RGB videos
and model-decision images are separate and require alignment for vision-policy training.

## Try a provisioned Rung API

You need an endpoint and a Rung API key from the service operator. This repository
ships the client; it does not deploy a hosted service. A Rung key is separate from
a model provider's key. Install the client dependencies and configure your access:

```sh
uv pip install -e ".[api]"
export RUNG_API_URL="wss://YOUR-PROVISIONED-ENDPOINT"
export RUNG_API_KEY="YOUR-RUNG-KEY"
python examples/remote_episode.py --out runs/remote --max-usd 0.50 --max-model-calls 12
python examples/inspect_recording.py runs/remote --export runs/remote-episode.zip
```

The mock runs on your CPU, while the service chooses motions. Calls can incur service
charges. The server may enforce lower limits, and an in-flight model request may cross
a dollar cap. The example exits 0 on caller-graded success, 1 on task failure and 2 on
a connection or recording error. A failed task is still a completed, inspectable episode.
No hosted endpoint is required for the recorded scripted example above.

The example keeps camera images, calibration, robot state, requested TCP targets,
execution receipts and the caller's result on your machine. It also retains completed
local records if the connection fails. The final observation is included. Your images
and state are still transmitted to the service to obtain actions; saving locally does
not make inference offline. API keys and authorization headers are not recorded.

These are **target-boundary recordings**, not a dense action dataset: one observation
before the first target and after each executed target. The adapter's internal control
steps and the service's model prompts/responses are not included. For imitation learning
at a fixed control rate, enable your simulator's native observation/action recorder;
RoboLab writes HDF5 trajectories in the [GPU guide](../deploy/robolab/README.md#output).
Its standard HDF5 contains dense controls and robot/simulator state; RGB videos and
the model's decision images are separate files. These outputs require alignment and
format conversion before they form a vision-policy training dataset.
Successful task completion alone does not establish that a trajectory is useful training data.

To use your own simulator, replace `SCENES["banana_in_bowl"]()` with your
[`Embodiment`](../docs/contracts.md#embodiment-protocol-rungembodimentpy) adapter and
call `run_remote_episode(..., record_dir="runs/my-task")`. Keep grading in the environment.
The CPU mock checks integration, not physical manipulation performance.

## Recording and export layout

```text
runs/remote/
  trace.jsonl
  episodes/<local-id>/
    session.json              instruction, requested budgets and recording scope
    server.json               reported model/backend and effective server limits
    capabilities.json         robot and camera specification
    obs_0000.json             state and camera calibration
    obs_0000_scene.png         RGB (additional cameras have their own files)
    obs_0000_scene_depth.npy   depth when the adapter supplies it
    ...                       one observation after each target
    result.json               present only after a normal session end
    interruption.json         present when the recorded run was interrupted
```

The inspector also works with `mock_episode.py --out`. Export puts all source files
under `recording/` inside a new ZIP and adds `manifest.json` with a SHA-256 and byte count
for each file. It refuses to overwrite an archive, follow symlinks, or export into the
recording itself. Review the full recording before sharing: it includes your task text
and scene images. Source files remain untouched.

## Read a native RoboLab trajectory

After a GPU episode, copy its output directory locally and inspect a recording:

```sh
uv pip install h5py
python examples/read_robolab.py out/first/episode/BananaInBowlTask/run_0.hdf5
```

The reader reports action, TCP and joint-state array shapes and checks that their
control-step counts agree and their values are finite. Read `config.json` and the
task's `provenance.json` for the action space, timing and coordinate conventions.
The reader does not infer task success from HDF5 attributes; use
`episode_results.jsonl` for the native grade. It does not decode or align RGB videos.
