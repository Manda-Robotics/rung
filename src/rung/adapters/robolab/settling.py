"""Evaluator-only settling evidence after native scoring has finished.

Does not unfreeze an environment, replace a native grade, or invoke the policy.
Zero commands pass through Verified's existing frozen-environment handling.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def post_success_settling(env, grades, *, seconds, step_s, out, capture=None, zeros=None):
    if not math.isfinite(seconds) or not 0 < seconds <= 3:
        raise ValueError("settling duration must be in (0, 3] seconds")
    if not math.isfinite(step_s) or step_s <= 0:
        raise ValueError("invalid physics step duration")
    if env.num_envs != 1:
        raise ValueError("development settling diagnostic requires one environment")
    original = json.dumps(grades, sort_keys=True, allow_nan=False)
    report = {
        "class": "post_episode_diagnostic_not_policy_or_benchmark_episode",
        "native_grades": json.loads(original),
        "native_grade_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "requested_seconds": seconds,
        "control": "zero commands through unchanged native frozen-env handling",
        "policy_calls": 0,
        "samples": [],
        "limitation": "No retraction; native grade remains the original termination outcome.",
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    path = out / "diagnostic.json"
    if not grades or not all(g.get("success") is True for g in grades):
        report.update(state="skipped_native_failure", steps=0)
        path.write_text(json.dumps(report, indent=2))
        return report
    if not env.all_terminated:
        raise ValueError("native episode must have ended before settling")
    if zeros is None:
        import torch

        def zeros(shape):
            return torch.zeros(shape, device=env.device)

    dim = env.action_manager.total_action_dim
    steps = math.ceil(seconds / step_s)
    for index in range(steps):
        if not env.all_terminated:
            raise RuntimeError("diagnostic must not reactivate an environment")
        raw, *_ = env.step(zeros((env.num_envs, dim)))
        sample = {"elapsed_after_native_end_s": (index + 1) * step_s}
        if capture is not None:
            sample.update(capture(raw, index, sample["elapsed_after_native_end_s"], out))
        report["samples"].append(sample)
    if json.dumps(grades, sort_keys=True, allow_nan=False) != original:
        raise RuntimeError("settling diagnostic changed supplied native grades")
    report.update(state="complete", steps=steps, actual_seconds=steps * step_s)
    path.write_text(json.dumps(report, indent=2))
    return report
