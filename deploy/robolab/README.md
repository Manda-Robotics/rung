# Running Rung on RoboLab with one GPU

This guide runs Rung as the policy inside NVIDIA's [RoboLab](https://github.com/NVlabs/RoboLab)
benchmark on a single NVIDIA GPU. RoboLab steps the simulator, records every episode and
grades it with its own scorer; Rung only supplies the actions. Nothing in RoboLab is
patched.

| Piece | Version |
|---|---|
| RoboLab | upstream `main` at [`9cce00c`](https://github.com/NVlabs/RoboLab/commit/9cce00cadd5dc575fd2b8ea3836f5bdcde574229) (v0.3.2 `b6cfcf6` plus the merged Isaac Sim 6 port, PR #47) |
| Simulator | Isaac Sim 6.0.1, Isaac Lab 3.0, Python 3.12, installed by RoboLab's own `uv.lock` |
| GPU | One NVIDIA RTX-class or data-center GPU with RT cores (for example L40S, RTX 4090), 24 GB or more |
| Driver | NVIDIA 595 or newer, the version this stack was run on |
| Model | Gemini 4 Argon through the paid Gemini API (`GEMINI_API_KEY`) |

The first-run command below was validated on an L40S with driver 595.91.07: TCP
calibration and all 20 adapter findings passed or were explicitly unmeasurable, then one
BananaInBowlTask episode received native success. The [worked recording](../../examples/README.md#inspect-a-real-recorded-episode)
includes its 75-step trajectory and full model trace. This single run is a setup check,
not a benchmark score.

The scripts in this folder:

| Script | Purpose |
|---|---|
| `calibrate_tcp.py` | Measures the DROID flange to fingertip offset. Run once per stack. |
| `check.py` | Adapter probe: camera convention, depth units, frames, motion scale, gripper. `run.py` refuses to start without a passing probe. |
| `calibrate_speed.py` | Optional. Re-measures the relative-IK tracking gain (0.283 on this stack). |
| `run.py` | Runs episodes of one task and writes RoboLab's results plus Rung's per-decision trace. |
| `suite.py` | Runs a task manifest, one Isaac process per task, so a simulator crash costs one task. |
| `connector.py`, `registration.py` | Glue for RoboLab's policy and task registries. |
| `Dockerfile` | RoboLab at `9cce00c` on Isaac Sim 6.0.1 with Rung on the path. |

## 1. Build the image

From the repository root:

```sh
docker build -f deploy/robolab/Dockerfile -t rung-robolab .
docker run --rm --gpus all rung-robolab
```

The second command prints the RoboLab revision and the Rung version. Building installs
Isaac Sim from NVIDIA's pip index; doing so means you accept the
[NVIDIA Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html).
Building downloads the simulator, robot assets and dependency wheels. The validated
image occupied 32.4 GiB. Allow substantial free disk space for extraction and build
layers as well as the final image. The recipe
uses Linux hardlinks to avoid duplicating the package cache during installation.
Rung does not publish a prebuilt image.

Without Docker, follow RoboLab's own install guide for Isaac Sim 6
(`uv sync --python 3.12 --extra isaac60` in a checkout of `9cce00c`), then put Rung on the
path instead of installing it, so pip cannot replace the numpy Isaac pins:

```sh
export PYTHONPATH=<RoboLab>:<rung>/src:<rung>/deploy/robolab
export OMNI_KIT_ACCEPT_EULA=Y
cd <RoboLab> && PY=.venv/bin/python
```

The remaining steps use the container. Inside it, `python` is RoboLab's interpreter and
the working directory is the RoboLab tree.

For a first episode, the example performs calibration, adapter checks and one Gemini
task with a per-episode model budget:

```sh
mkdir -p out
docker run --rm --gpus all -e GEMINI_API_KEY -v "$PWD/out:/out" rung-robolab \
    python /opt/rung/examples/robolab_episode.py --out /out/first --max-dollars 1.00
```

Use a fresh `--out` directory. Add `--qualify-only` to run calibration and checks
without a model call, or `--dry-run` to print the commands without starting the
simulator. The first episode uses `BananaInBowlTask`; pass `--task` for another
registered task. Calibration and the camera probe always use the same DROID
rig in `BananaInBowlTask`. The example exits 0 on native task success, 1 on a
completed task failure, and 2 if setup, qualification or execution is incomplete.
An in-flight model request can exceed the model-dollar cap. GPU rental is billed
separately by your provider.

The files remain in your mounted `out/first` directory: qualification JSON under
`calibration/` and `probe.json`, plus native results, HDF5 trajectories, videos and
model traces under `episode/`. The detailed commands below expose every setting
and let you reuse qualification files for later episodes.

```sh
mkdir -p out
docker run --rm -it --gpus all -e GEMINI_API_KEY -v "$PWD/out:/out" rung-robolab bash
```

## 2. Calibrate and probe (once per stack)

```sh
python /opt/rung/deploy/robolab/calibrate_tcp.py --out /out/calibration --headless
python /opt/rung/deploy/robolab/check.py --task BananaInBowlTask \
    --wrist-calibration --wrist-visual-fiducial --fiducial-position 0.45 0.1 0.15 \
    --tcp-offset-m "$(python -c 'import json; print(json.load(open("/out/calibration/calibration.json"))["tcp_offset_m"])')" \
    --out /out/probe.json --headless
```

Every script writes a JSON verdict with `"passed"`; an Isaac exit code of zero is not a
pass. Expect a TCP offset near 0.1488 m. `run.py` refuses an offset outside 0.06 to 0.20 m,
and refuses to start without both files passing.

## 3. Run one task

The flags below configure the direct Gemini policy through the paid API. The same
camera and tool settings are in [`configs/gemini-4-argon-direct.json`](../../configs/gemini-4-argon-direct.json).

```sh
python /opt/rung/deploy/robolab/run.py --task BananaInBowlTask --num-runs 1 --num-envs 1 \
    --interface direct --campaign-profile frontier-capability-dev \
    --model gemini-4-argon --effort medium \
    --model-max-tokens 8192 --model-timeout-s 900 \
    --direct-prices /opt/rung/configs/direct-gemini-4-argon-prices.json \
    --motion-tools both --max-translation-m 0.35 --max-rotation-rad 1.57 \
    --max-speed-mps 0.3 --max-ang-speed-rps 1.5 \
    --wrist-calibration --table-projection --stereo-triangulation \
    --camera-geometry-prompt --recent-frames --history-steps 14,7 --right-shoulder \
    --image-scale 0.3 --action-sequence --max-sequence-steps 15 \
    --done-awaits-confirmation \
    --time-mode paused --native-horizon --result-class leaderboard_submission \
    --max-decisions 180 --max-dollars 1.00 \
    --probe /out/probe.json --calibration /out/calibration/calibration.json \
    --out /out/banana --headless --kit_args=--/UJITSO/enabled=false
```

| Flag | Meaning |
|---|---|
| `--interface direct` | The model moves the gripper through tools (relative and absolute end-effector motions, gripper, guarded multi-step sequences). Also `l0` (pose targets) and `l2` (semantic primitives). |
| `--time-mode paused` | The simulation waits while the model thinks, as RoboLab's official synchronous runner does. Real inference latency is recorded separately. `realtime` instead replays model latency as robot time. |
| `--native-horizon` | The episode ends at the task's own step limit, with no extra harness cap. |
| `--result-class leaderboard_submission` | Records that the run used the upstream tree in paused time, the conditions of the official leaderboard. |
| `--max-decisions`, `--max-dollars` | Per-episode caps on model calls and spend. An episode that hits a cap ends and is graded as it stands. |
| `--model` without a key | Omit `--model` (and use `--interface l2`) for a scripted motion smoke test with no API calls. |

Only RoboLab's grader decides success. A policy that calls `done` holds still; with
`--done-awaits-confirmation` the episode continues unless the grader confirms success.

## 4. Run the benchmark

```sh
python /opt/rung/deploy/robolab/suite.py --python python \
    --tasks-file /opt/rung/tasks/robolab120_tasks.txt --episodes 10 \
    --max-dollars 1.00 --max-total-dollars 1200 --task-timeout-s 21600 \
    --out /out/robolab120 \
    <the run.py flags from step 3, without --task, --num-runs, --num-envs, --max-dollars and --out>
```

Unknown flags are forwarded to `run.py`. Each task runs in its own process; crashes are
recorded in `suite_status.json` and `status/<task>.json`, and `run_complete.json` appears
only when the whole manifest finished. Before each task the suite reserves
`--max-dollars` for every remaining episode and skips the task if that would pass
`--max-total-dollars`. `--resume` extends completed tasks. The 120 tasks
are identical to RoboLab's task metadata at `b6cfcf6` and `9cce00c`
(`configs/leaderboard-tasks-b6cfcf6.json`).

Cost: Gemini 4 Argon is priced at $2 per million input tokens and $10 per million output
tokens (`configs/direct-gemini-4-argon-prices.json`, dated). Rung meters every call and
stops an episode at `--max-dollars`. Measure a few tasks before launching all 1,200
episodes.

## Output

```
/out/banana/
  episode_results.jsonl          RoboLab grades, one row per episode
  config.json                    run configuration, source hashes, timing statement
  status/<task>.json             running | complete | failed
  <TaskName>/
    provenance.json              probe, calibration, effective horizon
    *.hdf5, *.mp4, log_*.json    RoboLab recordings and event logs
    rung/episode-<n>-env-<k>/
      trace.jsonl                every harness event with simulated and wall time
      native/                    the exact model requests and responses
```

Copy `/out` off the machine after every task if the GPU host is ephemeral.

To inspect native control/state arrays on a CPU, install `h5py` and run
`python examples/read_robolab.py PATH/TO/run_0.hdf5` from the Rung checkout.
The standard HDF5 records actions and robot/simulator state; RGB videos and
model-decision images are separate. The files require alignment and conversion
for a vision-policy training pipeline.

## Troubleshooting

- `vulkaninfo --summary` must list the GPU inside the container. If it does not, check
  the driver version and that the NVIDIA Container Toolkit is installed.
- A probe failure means the adapter does not match this stack; do not run episodes
  until it passes.
- `run.py` refuses a reused output directory unless `--resume` is given and the
  earlier task completed.
