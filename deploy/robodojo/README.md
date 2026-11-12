# Running Rung on RoboDojo with two arms

This guide runs Rung as the policy for [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)'s
dual ARX X5 rig on one NVIDIA GPU. RoboDojo steps the simulator, ends each episode on success
or on its action limit, and grades with its own `run_reward`; Rung only supplies the actions,
through RoboDojo's stock XPolicyLab policy server. Nothing in RoboDojo is patched; the setup
adds one env_cfg that switches on depth and camera calibration.

| Piece | Version |
|---|---|
| RoboDojo | upstream `main` at [`f57e163`](https://github.com/RoboDojo-Benchmark/RoboDojo/commit/f57e1633ffed59761bc8e568cfe7530c4b33381c), XPolicyLab submodule `5d0be14` |
| Simulator | Isaac Sim 5.1, Isaac Lab 2.3, cuRobo, Python 3.11 in RoboDojo's conda env, installed by RoboDojo's own `scripts/install.sh` |
| Robot | `dual_x5` with env_cfg `arx_x5`: two ARX X5 arms, head camera plus one wrist camera per arm, simulator-side cuRobo IK |
| GPU | One NVIDIA RTX-class or data-center GPU with RT cores and 24 GB or more (validated on an RTX 4090; an L40 also ran it) |
| Driver | **Below 595** (570 works). Isaac Sim 5.1 segfaults at renderer start (`librtx.scenedb`) on driver 595.x, including on L40S |
| Disk | About 300 GB: the RoboDojo assets are about 66 GB, plus Isaac Sim, the conda env and build caches |
| Model | Gemini 4 Argon through the paid Gemini API (`GEMINI_API_KEY`), or the fake model with no key |

This differs from the [RoboLab guide](../robolab/README.md) in three ways: Isaac Sim 5.1
instead of 6.0.1, a driver below 595 instead of 595 or newer, and no Docker image. RoboDojo's
installer builds its own conda environment, so the two stacks need separate hosts or images.

The commands below were validated on 2026-11-12 from this release's exported tree on a
24 GB RTX 4090 with driver 570.211.01. `setup.sh` took 46 minutes: 28 for clones and RoboDojo's
installer, 18 for the assets. The scripted check moved each arm as commanded, with residuals of 2.5 to 4.5 mm.
Then one Gemini 4 Argon `general_pickup` episode received RoboDojo's own grade, success
(score 1.0), at action 99 of 200, after 3 model calls. Usage was 48,833 input, 568 output and
4,902 thinking tokens, about $0.14 at the recorded price. This single run is a setup check,
not a benchmark score.

The scripts in this folder:

| File | Purpose |
|---|---|
| `setup.sh` | Clones RoboDojo at the pinned revision, runs its installer and asset download, installs Rung into the RoboDojo env, writes the `arx_x5_rung` env_cfg, installs the policy shim. Idempotent per stage |
| `run.sh` | One episode of one task: `run.sh TASK [scripted\|direct] [ENV_CFG]` |
| `policy/model.py` | XPolicyLab `Model` that hands every call to `rung.adapters.robodojo.server.RungRoboDojoModel` |
| `policy/deploy.py` | RoboDojo's per-episode loop for this policy: one observation, one action |
| `policy/deploy.yml` | Policy-server config: one environment, long websocket keepalive while the model thinks |

## How it fits together

```
RoboDojo simulator (Isaac Sim 5.1)          XPolicyLab policy server (same host)
  eval_one_episode (policy/deploy.py)  ws    Model (policy/model.py)
    reset ─────────────────────────────────▶   RungRoboDojoModel: start a harness episode
    update_obs(obs) ───────────────────────▶   RoboDojoCodec: obs dict -> Observation (both arms)
    get_action() ◀─────────────────────────    BimanualDriver: one action per call
    take_action(action), repeat                 run_episode on its own thread: DirectPolicy
                                                (bimanual profile) and the model API
```

Each `get_action` returns one action dict, `{side}_ee_pose` (7: position and `w x y z`
quaternion) and `{side}_ee_joint_state` (1) per arm. The simulator solves IK with cuRobo and
plays it over 0.04 s. While the model thinks, the bridge holds both arms and the simulator is
not stepped (`time_mode=paused`): RoboDojo's `step_lim` counts actions, not seconds.

| Aspect | Mapping |
|---|---|
| Root frame | RoboDojo's end-effector frame: world minus env origin, z up. Arm bases at `(-0.3, -0.45, 0.765)` (left) and `(0.3, -0.45, 0.765)` (right), both facing `+y`; the robots' left is `-x` |
| Cameras | `cam_head` becomes `scene`, `cam_{side}_wrist` becomes `{side}_wrist`. With `arx_x5_rung`: depth, intrinsics and camera-to-world extrinsics (USD axes, converted to OpenCV) |
| TCP | Fingertip midpoint 0.15 m along `link6 +x`, from the X5 URDF and finger mesh. Not touch-calibrated |
| Gripper | `1` opens. `{side}_ee_joint_state` is the commanded value, not a finger measurement, so a closed reading is not proof of a hold |
| Table | Measured from head-camera depth near the declared 0.765 m at episode start; the declared value is kept if the measurement is ambiguous |
| IK failure | A pose cuRobo cannot solve holds that arm. Rung reports it as `blocked` for that arm |
| Grade | RoboDojo's `run_reward` (`> 0.999` ends the episode as a success) in `eval_result/.../_result.json`. The policy server is never told; a model calling `done` is not a grade |

## 1. Install

On a Linux host with the driver and disk above, as root, with Rung checked out at
`/workspace/rung` (or set `RUNG_ROOT`):

```sh
git clone https://github.com/Manda-Robotics/rung.git /workspace/rung
bash /workspace/rung/deploy/robodojo/setup.sh
```

It prints `ROBODOJO_SETUP_DONE` at the end; per-stage logs are in `/workspace/logs`. A fresh
host took 46 minutes in the validated run: submodule clones, RoboDojo's installer, then the
66 GB asset download. Running it means you accept the
[NVIDIA Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)
and RoboDojo's license terms.

Two details the script handles that a hand install tends to miss:

- Rung is installed with `--no-deps`. Rung declares `numpy>=2`; Isaac Lab 2.3 pins `numpy<2`,
  and letting pip resolve Rung breaks Omniverse with "unknown dtype". Rung's adapters run on
  numpy 1.26.
- `utils/update_embodiment_config_path.py` must run after the asset download to write
  `curobo.yml`; without it the simulator cannot build its IK.

## 2. Check the motion with no model

```sh
bash /workspace/rung/deploy/robodojo/run.sh general_pickup scripted
```

`scripted` lifts each arm 5 cm alone, moves both 4 cm forward together, then closes each
gripper, and prints each arm's measured TCP before every step with the receipts of the step
before. Read `/workspace/out/robodojo-general_pickup-scripted/eval.log`: each lift should
show about `+0.05` in z for that arm only, and the paired move `+0.04` in y for both. It does
not try to solve the task, so RoboDojo grades it a failure.

## 3. Run one episode with a model

```sh
export GEMINI_API_KEY="your-api-key"
RUNG_MAX_CALLS=12 bash /workspace/rung/deploy/robodojo/run.sh general_pickup direct
```

`direct` loads [`configs/gemini-4-argon-robodojo.json`](../../configs/gemini-4-argon-robodojo.json)
(override with `RUNG_PROFILE`). Set `"model": "fake"` in a copy of the profile to run the
whole path, including the policy server and the bimanual tools, with no key and no charge.

| Variable | Default | Meaning |
|---|---|---|
| `RUNG_MAX_CALLS` | 12 | Model calls and decisions per episode |
| `RUNG_HORIZON_S` | 8 | Seconds of motion the model is told it has; RoboDojo's `step_lim` is what ends the episode (`general_pickup`: 200 actions = 8 s) |
| `RUNG_PROFILE` | `configs/gemini-4-argon-robodojo.json` | Direct profile; must set `direct_config.bimanual` |
| `RUNG_OUT` | `/workspace/out/robodojo-TASK-POLICY` | `status.json`, `eval.log`, Rung's trace and recording, model requests and responses |
| `RUNG_GRIPPER_ONE_IS_OPEN` | 1 | Gripper sign; 1 was checked in the wrist images |

`status.json` has the episode's end reason, decisions, model calls and metered cost once the
harness episode ends (`episode_done`); RoboDojo may end the episode first, on success. The task
grade is RoboDojo's: look for the newest `_result.json` under `eval_result/`. Calls to a real
model incur provider charges. `RUNG_MAX_CALLS` bounds them; an in-flight request still
completes and is billed.

## What to expect

Turn-by-turn control suits the generalization and open-ended pick tasks. Conveyor and timing
tasks (`*_conveyor*`) need sub-second reaction that a model turn cannot give, even paused,
because the conveyor runs during the motion. Deformable and precision-insertion tasks need
primitives the bimanual profile does not have. Each arm is checked only against its own
reachable box; nothing prevents the arms colliding at the midline. See
[two arms in the interface reference](../../docs/interfaces.md#two-arms-the-bimanual-direct-profile).

Episodes take 3 to 10 minutes of wall time, almost all of it model latency. One environment
per policy server: `eval_batch` stays `false`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Segfault in `librtx.scenedb` at start | Driver 595 or newer with Isaac Sim 5.1. Use a host with a 570-series driver |
| "no suitable CUDA GPU" and every camera capture fails, though `nvidia-smi` works | Missing EGL or Vulkan libraries in the container; `setup.sh` installs them |
| "unknown dtype, size=0" from Omniverse | numpy 2 in the RoboDojo env; reinstall `numpy<2` and Rung with `--no-deps` |
| An arm never moves and its receipts say `blocked` | cuRobo could not solve that pose. The receipt names the arm; the model sees it next turn |
| The workspace sits below the arms | Table measurement fell back without depth: use env_cfg `arx_x5_rung`, not `arx_x5` |
