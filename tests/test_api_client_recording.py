"""Client-owned recording of remote episodes, including interrupted execution."""

import json

import numpy as np
import pytest

from rung.api.client import run_remote_episode
from rung.api.codec import from_wire, to_wire
from rung.contracts.action import TCPTarget
from rung.mock import SCENES


class ScriptedClient:
    def __init__(self, fail_on_step=None):
        self.requests = []
        self.metadata = {"backend": "fake", "model": "test", "key": "private-key-policy"}
        self.fail_on_step = fail_on_step
        self.steps = 0

    def request(self, message):
        self.requests.append(message)
        if message["type"] == "start":
            # A remote session identifier must never become a local path.
            return {"type": "started", "session_id": "../../remote-session"}
        if message["type"] == "end":
            return {
                "type": "ended",
                "success": message.get("success"),
                "end_reason": "policy_done",
                "usage": {"calls": 3},
            }
        self.steps += 1
        if self.steps == self.fail_on_step:
            raise ConnectionError("test disconnect")
        if self.steps == 3:
            return {"type": "finished"}
        obs = from_wire(message["observation"])
        target = TCPTarget(pose=obs.tcp)
        return {"type": "action", "target": to_wire(target)}


def recorded_episode(root):
    return next((root / "episodes").iterdir())


def test_remote_recording_contains_initial_and_final_observations_and_actual_receipts(tmp_path):
    root = tmp_path / "recording"
    client = ScriptedClient()
    summary = run_remote_episode(
        client, SCENES["banana_in_bowl"](), "put banana in bowl", episode_seed=4, record_dir=root
    )
    episode = recorded_episode(root)
    session = json.loads((episode / "session.json").read_text())
    assert session["episode_seed"] == 4
    assert session["dense_control_steps"] is False
    assert session["server_model_conversation"] is False
    server = json.loads((episode / "server.json").read_text())
    assert server["metadata"] == {"backend": "fake", "model": "test"}
    assert summary["targets_executed"] == 2
    observations = sorted(episode.glob("obs_*.json"))
    assert len(observations) == 3
    for index, path in enumerate(observations):
        observation = json.loads(path.read_text())
        assert observation["index"] == index
        assert observation["cameras"]["scene"]["K"]
        assert (episode / f"obs_{index:04d}_scene.png").is_file()
        assert np.load(episode / f"obs_{index:04d}_scene_depth.npy").ndim == 2
    rows = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in rows] == [
        "observation",
        "attempt",
        "execution",
        "observation",
        "attempt",
        "execution",
        "observation",
        "episode_end",
    ]
    assert rows[1]["payload"]["target"]["id"] == rows[2]["payload"]["report"]["target_id"]
    assert json.loads((episode / "result.json").read_text()) == summary
    assert not (tmp_path / "remote-session").exists()


def test_disconnect_preserves_prior_observations_and_execution_without_claiming_completion(
    tmp_path,
):
    root = tmp_path / "partial"
    with pytest.raises(ConnectionError):
        run_remote_episode(
            ScriptedClient(fail_on_step=2), SCENES["banana_in_bowl"](), "test", record_dir=root
        )
    episode = recorded_episode(root)
    assert len(list(episode.glob("obs_*.json"))) == 2
    assert not (episode / "result.json").exists()
    assert (
        json.loads((episode / "interruption.json").read_text())["error_type"] == "ConnectionError"
    )
    rows = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in rows].count("execution") == 1
    assert rows[-1]["kind"] == "execution_error"
    assert "test disconnect" not in (episode / "interruption.json").read_text()


def test_existing_recording_is_refused_before_session_start(tmp_path):
    client = ScriptedClient()
    with pytest.raises(FileExistsError):
        run_remote_episode(client, SCENES["banana_in_bowl"](), "test", record_dir=tmp_path)
    assert client.requests == []


def test_recording_is_optional_and_does_not_change_the_wire_exchange(tmp_path):
    plain = ScriptedClient()
    recorded = ScriptedClient()
    first = run_remote_episode(plain, SCENES["banana_in_bowl"](), "test", episode_seed=0)
    second = run_remote_episode(
        recorded,
        SCENES["banana_in_bowl"](),
        "test",
        episode_seed=0,
        record_dir=tmp_path / "recording",
    )
    assert first == second
    assert [item["type"] for item in plain.requests] == [item["type"] for item in recorded.requests]
    for a, b in zip(plain.requests, recorded.requests, strict=True):
        if a["type"] == "step":
            left, right = from_wire(a["observation"]), from_wire(b["observation"])
            assert np.array_equal(left.scene.rgb, right.scene.rgb)
            assert np.array_equal(left.tcp.pos, right.tcp.pos)
