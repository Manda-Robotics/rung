"""Native simulator verdicts, not process exit codes, establish a completed example."""

import importlib.util
import json
from pathlib import Path

import pytest


def example():
    path = Path(__file__).resolve().parents[1] / "examples" / "robolab_episode.py"
    spec = importlib.util.spec_from_file_location("robolab_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_or_failed_qualification_is_not_a_pass(tmp_path):
    path = tmp_path / "probe.json"
    with pytest.raises(FileNotFoundError):
        example().verdict(path)
    path.write_text('{"passed":false}')
    with pytest.raises(ValueError, match="qualification failed"):
        example().verdict(path)
    path.write_text('{"passed":true}')
    assert example().verdict(path)["passed"] is True


def test_completed_task_requires_grade_and_native_recording(tmp_path):
    status = tmp_path / "episode/status"
    status.mkdir(parents=True)
    file = status / "BananaInBowlTask.json"
    file.write_text(
        json.dumps({"state": "complete", "expected_episodes": 1, "completed_episodes": 1})
    )
    results = tmp_path / "episode/episode_results.jsonl"
    results.write_text('{"success":false}\n')
    with pytest.raises(ValueError, match="HDF5"):
        example().episode_result(tmp_path, "BananaInBowlTask")
    # Test fixture bytes establish only that the wrapper demands a nonempty file.
    # Native HDF5 validity is checked separately on the GPU validation path.
    (tmp_path / "episode/run_0.hdf5").write_bytes(b"test recording")
    assert example().episode_result(tmp_path, "BananaInBowlTask")["success"] is False
    file.write_text('{"state":"failed","completed_episodes":1,"expected_episodes":1}')
    with pytest.raises(ValueError, match="did not complete"):
        example().episode_result(tmp_path, "BananaInBowlTask")
