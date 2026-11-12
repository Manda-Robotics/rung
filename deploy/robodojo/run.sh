#!/bin/bash
# One RoboDojo episode with Rung.
#   run.sh TASK [scripted|direct] [ENV_CFG]
# direct reads RUNG_PROFILE (default configs/gemini-4-argon-robodojo.json) and the model key
# from the environment, e.g. GEMINI_API_KEY. Output: $RUNG_OUT (status.json, eval.log, Rung
# traces) and RoboDojo's own eval_result/ (the graded _result.json).
set -uo pipefail
task=${1:-general_pickup}; policy=${2:-scripted}; env_cfg=${3:-arx_x5_rung}
WORK=${WORK:-/workspace}
export RUNG_ROOT=${RUNG_ROOT:-$WORK/rung} RUNG_POLICY=$policy
export RUNG_OUT=${RUNG_OUT:-$WORK/out/robodojo-$task-$policy}
export RUNG_HORIZON_S=${RUNG_HORIZON_S:-8} RUNG_MAX_CALLS=${RUNG_MAX_CALLS:-12}
export EVAL_NUM=${EVAL_NUM:-1} OMNI_KIT_ACCEPT_EULA=Y
mkdir -p "$RUNG_OUT"
cd "$WORK/RoboDojo"
source "$(conda info --base 2>/dev/null || echo /root/miniconda3)/etc/profile.d/conda.sh"
conda activate RoboDojo
# eval.sh BENCH TASK CKPT ENV_CFG ACTION_TYPE SEED POLICY_GPU ENV_GPU POLICY_ENV EVAL_ENV
bash XPolicyLab/policy/Rung/eval.sh RoboDojo "$task" rung "$env_cfg" ee 0 0 0 RoboDojo RoboDojo \
  > "$RUNG_OUT/eval.log" 2>&1
code=$?
python - "$RUNG_OUT/status.json" "$code" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1]); s = json.loads(p.read_text()) if p.exists() else {}
s.update(state="complete" if sys.argv[2] == "0" else "failed", exit_code=int(sys.argv[2]), t=time.time())
p.write_text(json.dumps(s, indent=2))
PY
echo "EVAL_EXIT=$code"
find eval_result -name '*_result.json' -mmin -120 2>/dev/null | tail -3
