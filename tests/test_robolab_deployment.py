"""Failure/completion semantics of the simulator-free suite supervisor."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def supervisor(monkeypatch):
    deploy = Path(__file__).parents[1] / "deploy" / "robolab"
    monkeypatch.syspath_prepend(str(deploy))
    spec = importlib.util.spec_from_file_location("rung_suite_test", deploy / "suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suite_manifest_covers_all_120_unique_tasks(supervisor):
    manifest = Path(__file__).parents[1] / "tasks" / "robolab120_tasks.txt"
    assert len(supervisor.load_tasks(manifest)) == 120


def test_status_must_match_complete_episode_count(supervisor, tmp_path):
    path = tmp_path / "status.json"
    assert not supervisor.complete_status(path, 2)
    for state, done, expected in [
        ("failed", 2, 2),
        ("running", 2, 2),
        ("complete", 1, 2),
        ("complete", 2, 3),
    ]:
        path.write_text(
            json.dumps(dict(state=state, completed_episodes=done, expected_episodes=expected))
        )
        assert not supervisor.complete_status(path, 2)
    path.write_text(json.dumps(dict(state="complete", completed_episodes=2, expected_episodes=2)))
    assert supervisor.complete_status(path, 2)


def test_zero_exit_without_json_is_a_failed_suite(supervisor, tmp_path, monkeypatch):
    tasklist = tmp_path / "tasks.txt"
    tasklist.write_text("BananaInBowlTask\n")
    out = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        ["suite.py", "--tasks-file", str(tasklist), "--out", str(out), "--episodes", "2"],
    )
    from types import SimpleNamespace

    monkeypatch.setattr(supervisor.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert supervisor.main() == 1
    assert not (out / "run_complete.json").exists()
    assert json.loads((out / "suite_status.json").read_text())["outcomes"] == {
        "BananaInBowlTask": "failed"
    }
