#!/bin/bash
# Install RoboDojo (Isaac Sim 5.1, Isaac Lab 2.3, cuRobo, XPolicyLab) and Rung's policy
# wrapper on a fresh Linux GPU host. Run as root from anywhere; Rung must be checked out at
# $RUNG_ROOT. Needs about 300 GB of disk (the assets are about 66 GB) and an NVIDIA driver
# below 595 (Isaac Sim 5.1 segfaults at renderer start on 595.x). Each stage is skipped
# when its marker exists, so the script can be re-run after a failure. Logs: $LOGS.
set -euo pipefail
RUNG_ROOT=${RUNG_ROOT:-/workspace/rung}
WORK=${WORK:-/workspace}
LOGS=${LOGS:-$WORK/logs}
ROBODOJO_REF=${ROBODOJO_REF:-f57e1633ffed59761bc8e568cfe7530c4b33381c}
mkdir -p "$LOGS"
export DEBIAN_FRONTEND=noninteractive OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y

# git-lfs for the assets. A fresh container also lacks the EGL and Vulkan libraries; without
# them Isaac Sim reports "no suitable CUDA GPU" and every camera capture fails, even though
# nvidia-smi and torch see the GPU.
apt-get -o DPkg::Lock::Timeout=900 update -qq
apt-get -o DPkg::Lock::Timeout=900 install -y -qq git git-lfs libegl1 libgl1 libglvnd0 \
  libglu1-mesa libxt6 libvulkan1

cd "$WORK"
if [ ! -d RoboDojo ]; then
  git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git
  git -C RoboDojo checkout -q "$ROBODOJO_REF"
  git -C RoboDojo submodule update --init --recursive
fi
cd RoboDojo
if [ ! -f "$LOGS/install.done" ]; then
  bash scripts/install.sh -i > "$LOGS/install.log" 2>&1
  touch "$LOGS/install.done"
fi
if [ ! -f "$LOGS/assets.done" ]; then
  bash scripts/init_assets.sh > "$LOGS/assets.log" 2>&1
  touch "$LOGS/assets.done"
fi

source "$(conda info --base 2>/dev/null || echo /root/miniconda3)/etc/profile.d/conda.sh"
conda activate RoboDojo
# --no-deps: Rung needs numpy and pillow, both already present. Letting pip resolve Rung's
# numpy>=2 would replace Isaac Lab 2.3's numpy<2 and break Omniverse ("unknown dtype").
pip install -q --no-deps -e "$RUNG_ROOT" > "$LOGS/rung-pip.log" 2>&1
pip install -q -e XPolicyLab "numpy<2" >> "$LOGS/rung-pip.log" 2>&1
# The Gemini backend speaks REST over the standard library and needs no extra package.

# Writes Assets/Robots/*/curobo.yml with absolute paths; install.sh and init_assets.sh do not,
# and the simulator cannot build its IK without it.
python utils/update_embodiment_config_path.py < /dev/null > "$LOGS/embodiment-path.log" 2>&1

# The Rung policy: demo_policy's launch scripts with Rung's model, deploy loop and config.
rm -rf XPolicyLab/policy/Rung
cp -r XPolicyLab/policy/demo_policy XPolicyLab/policy/Rung
cp "$RUNG_ROOT"/deploy/robodojo/policy/{model.py,deploy.py,deploy.yml} XPolicyLab/policy/Rung/

# env_cfg arx_x5_rung: arx_x5 (same config_name, so the saved layouts resolve) with depth,
# intrinsics and camera extrinsics on, so Rung can measure the table and triangulate.
sed -e 's/^    depth: false/    depth: true/' -e 's/intrinsic_matrix: false/intrinsic_matrix: true/' \
    -e 's/extrinsic_matrix: false/extrinsic_matrix: true/' \
    -e 's/camera: camera_config/camera: camera_config_rung/' env_cfg/arx_x5.yml > env_cfg/arx_x5_rung.yml
sed -e 's/^    # distance_to_image_plane_capture:/    distance_to_image_plane_capture:/' \
    -e 's/^    #   type: distance_to_image_plane/      type: distance_to_image_plane/' \
    -e 's/^    #   device: cpu/      device: cpu/' env_cfg/camera/camera_config.yml \
    > env_cfg/camera/camera_config_rung.yml
grep -q "depth: true" env_cfg/arx_x5_rung.yml || { echo "depth switch did not apply" >&2; exit 1; }
echo ROBODOJO_SETUP_DONE
