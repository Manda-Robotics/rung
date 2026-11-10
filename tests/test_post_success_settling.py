from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rung.adapters.robolab.settling import post_success_settling


class FinishedEnv:
    num_envs = 1
    all_terminated = True
    action_manager = SimpleNamespace(total_action_dim=7)

    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action.copy())
        return {}, None, None, None, {}


def test_native_grade_preserved_and_only_zero_commands_after_termination(tmp_path):
    env = FinishedEnv()
    grades = [{"env_id": 0, "success": True, "step": 201}]
    original = deepcopy(grades)
    report = post_success_settling(
        env, grades, seconds=0.2, step_s=0.1, out=tmp_path / "diagnostic", zeros=np.zeros
    )
    assert report["state"] == "complete" and report["policy_calls"] == 0
    assert len(env.actions) == 2 and all(not a.any() for a in env.actions)
    assert grades == original == report["native_grades"]


def test_policy_failure_does_not_get_extra_physics_or_replacement_grade(tmp_path):
    env = FinishedEnv()
    report = post_success_settling(
        env,
        [{"success": False}],
        seconds=0.2,
        step_s=0.1,
        out=tmp_path / "diagnostic",
        zeros=np.zeros,
    )
    assert report["state"] == "skipped_native_failure"
    assert not env.actions


def test_cannot_run_before_native_termination(tmp_path):
    env = FinishedEnv()
    env.all_terminated = False
    with pytest.raises(ValueError, match="must have ended"):
        post_success_settling(
            env,
            [{"success": True}],
            seconds=0.2,
            step_s=0.1,
            out=tmp_path / "diagnostic",
            zeros=np.zeros,
        )
    assert not env.actions
