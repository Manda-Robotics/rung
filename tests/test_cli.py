"""Public command behavior, including task failure and invalid episode counts."""

import json

import pytest

from rung.cli import main


def test_scripted_cli_saves_a_successful_episode(tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["--policy", "scripted", "--out", str(out)]) == 0
    assert "1/1 episodes solved" in capsys.readouterr().out
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["interface"] == "scripted"
    assert provenance["notes"]["dollars"] == 0
    episodes = list((out / "episodes").glob("*/result.json"))
    assert len(episodes) == 1
    assert json.loads(episodes[0].read_text())["success"] is True
    assert (out / "trace.jsonl").is_file()


def test_fake_model_completes_tools_but_cli_reports_task_failure(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = tmp_path / "fake"
    assert main(["--policy", "direct", "--model", "fake", "--out", str(out)]) == 1
    output = capsys.readouterr().out
    assert "policy_done" in output
    assert "0/1 episodes solved" in output
    result = json.loads(next((out / "episodes").glob("*/result.json")).read_text())
    assert result["usage"]["calls"] == 5
    assert result["end_reason"] == "policy_done"


@pytest.mark.parametrize("episodes", ["0", "-1"])
def test_invalid_episode_count_fails_before_creating_output(tmp_path, capsys, episodes):
    out = tmp_path / "invalid"
    with pytest.raises(SystemExit) as exc:
        main(["--episodes", episodes, "--out", str(out)])
    assert exc.value.code == 2
    assert "--episodes must be at least 1" in capsys.readouterr().err
    assert not out.exists()
