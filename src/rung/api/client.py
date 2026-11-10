"""Callers of the API: a generic connection, a robot driver and a RoboLab client.

``RungApiClient``            one WebSocket connection, request/reply
``run_remote_episode``       drive any local ``Embodiment`` (the mock, a hardware adapter)
                             through a ``tcp_target`` session
``RungRemoteRoboLabClient``  the same surface as ``RungPolicyClient`` (begin/prepare,
                             ``infer_batch``, reset, ``record_grades``) for RoboLab's
                             evaluator, with the harness on the server. This is the client
                             code an organizer would run against a hosted endpoint.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from websockets.sync.client import connect

from rung.adapters.robolab.compat import instruction_text
from rung.adapters.robolab.translate import to_numpy
from rung.api.codec import MAX_MESSAGE_BYTES, from_wire, packb, to_wire, unpackb
from rung.contracts.trace import TraceWriter
from rung.record import Recorder


class ApiError(RuntimeError):
    def __init__(self, reply: dict):
        super().__init__(f"{reply.get('code')}: {reply.get('message')}")
        self.code = reply.get("code")


class RungApiClient:
    def __init__(self, uri: str, api_key: str, *, open_timeout: float = 30):
        self._connection = connect(
            uri,
            additional_headers={"Authorization": f"Api-Key {api_key}"},
            max_size=MAX_MESSAGE_BYTES,
            open_timeout=open_timeout,
            ping_interval=20,
            ping_timeout=60,
        )
        self.ws = self._connection.__enter__()
        self.metadata = unpackb(self.ws.recv())

    def request(self, message: dict) -> dict:
        self.ws.send(packb(message))
        reply = unpackb(self.ws.recv())
        if reply.get("type") == "error":
            raise ApiError(reply)
        return reply

    def close(self) -> None:
        self._connection.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_remote_episode(
    client: RungApiClient,
    embodiment,
    instruction: str,
    *,
    limits: dict | None = None,
    episode_seed: int | None = None,
    record_dir: str | Path | None = None,
) -> dict:
    """Run an episode, optionally saving target-boundary observations and actions locally.

    ``record_dir`` must not exist. Recordings include the final observation and retain
    partial evidence on errors. They contain TCP targets and execution receipts, not
    the adapter's dense control steps or the server's internal model conversation.
    Use a client context manager so a failed episode also closes its connection.
    """
    recording = _RemoteRecording(record_dir, instruction, episode_seed, limits)
    try:
        return _run_remote_episode(client, embodiment, instruction, limits, episode_seed, recording)
    except BaseException as exc:
        recording.failed(exc)
        raise


def _run_remote_episode(client, embodiment, instruction, limits, episode_seed, recording):
    started = client.request(
        {
            "type": "start",
            "action_space": "tcp_target",
            "instruction": instruction,
            "capabilities": to_wire(embodiment.capabilities),
            "limits": limits or {},
        }
    )
    sid = started["session_id"]
    recording.started(embodiment.capabilities, started, getattr(client, "metadata", {}))
    observation, report, targets = embodiment.reset(episode_seed), None, 0
    recording.observation(observation, targets)
    while True:
        reply = client.request(
            {
                "type": "step",
                "session_id": sid,
                "observation": to_wire(observation),
                "report": None if report is None else to_wire(report),
            }
        )
        if reply["type"] == "finished":
            break
        target = from_wire(reply["target"])
        recording.attempt(target, observation.t, targets)
        report = embodiment.execute(target)
        recording.execution(report, targets)
        observation = embodiment.observe()
        targets += 1
        recording.observation(observation, targets)
    grade = getattr(embodiment, "grade", None)
    end = {"type": "end", "session_id": sid}
    if grade is not None:
        end["success"] = bool(grade()[0])
    ended = client.request(end)
    summary = {**ended, "targets_executed": targets}
    recording.finished(summary, observation.t)
    return summary


class _RemoteRecording:
    """Client-owned evidence; remote session IDs are never used as file paths."""

    def __init__(self, root, instruction, seed, limits):
        self.recorder = None
        self.trace = None
        self.episode = uuid.uuid4().hex
        self.t = 0.0
        if root is None:
            return
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        self.recorder = Recorder(root / "episodes")
        self.trace = TraceWriter(root / "trace.jsonl")
        self.recorder.record_json(
            self.episode,
            "session.json",
            {
                "format": "rung-remote-recording/1",
                "action_space": "tcp_target",
                "instruction": instruction,
                "episode_seed": seed,
                "requested_limits": limits or {},
                "observation_frequency": "initial state and after each TCP target",
                "dense_control_steps": False,
                "server_model_conversation": False,
            },
        )

    def started(self, capabilities, started, metadata):
        if self.recorder:
            self.recorder.record_json(self.episode, "capabilities.json", capabilities)
            self.recorder.record_json(
                self.episode,
                "server.json",
                {
                    "metadata": {
                        key: metadata[key]
                        for key in ("protocol", "server", "backend", "model", "effort")
                        if key in metadata
                    },
                    "effective_limits": started.get("limits"),
                    "time_mode": started.get("time_mode"),
                },
            )

    def observation(self, observation, index):
        self.t = observation.t
        if self.recorder:
            self.recorder.record_observation(self.episode, index, observation)
            self.trace.emit("observation", observation.t, self.episode, index=index)

    def attempt(self, target, t, index):
        if self.trace:
            self.trace.emit("attempt", t, self.episode, index=index, target=target)

    def execution(self, report, index):
        if self.trace:
            self.trace.emit("execution", report.t_end, self.episode, index=index, report=report)

    def finished(self, summary, t):
        if self.recorder:
            self.recorder.record_json(self.episode, "result.json", summary)
            self.trace.emit("episode_end", t, self.episode, **summary)

    def failed(self, exc):
        if self.recorder:
            # Preserve the original exception even if recording itself failed (e.g. disk full).
            try:
                result = {"type": "interrupted", "error_type": type(exc).__name__}
                self.recorder.record_json(self.episode, "interruption.json", result)
                self.trace.emit("execution_error", self.t, self.episode, **result)
            except Exception:
                pass


def _env_slice(raw_obs: dict, env_id: int) -> dict:
    """One environment's terms with a leading batch dimension of 1, as numpy arrays."""
    out = {}
    for group in ("image_obs", "proprio_obs"):
        out[group] = {}
        for key, value in (raw_obs.get(group) or {}).items():
            array = to_numpy(value)
            if array is not None and array.ndim >= 1:
                out[group][key] = array[env_id : env_id + 1]
    return out


class RungRemoteRoboLabClient:
    """RoboLab policy client whose harness runs on a Rung API server.

    Each environment's episode is one ``droid_rel_ik`` session. ``reset`` stops sending
    steps; the session is closed with the evaluator's grade in ``record_grades`` (or
    without one at the next episode or ``close``).
    """

    def __init__(
        self,
        uri: str,
        api_key: str,
        *,
        instruction: str = "",
        limits: dict | None = None,
        time_mode: str = "paused",
        droid: dict | None = None,
    ):
        self.uri, self.api_key = uri, api_key
        self.instruction, self.limits = instruction, limits or {}
        self.time_mode, self.droid = time_mode, droid or {}
        self.conn = RungApiClient(uri, api_key)
        self.episode_idx = None
        self._prepared = None
        self._live: dict[int, str] = {}
        self._ending: dict[int, str] = {}
        self.summaries: dict[tuple[int, int], dict] = {}

    def begin_episode(self, episode_idx: int) -> None:
        self._end_all()
        self.episode_idx = episode_idx

    def prepare_episode(self, episode_idx: int, available_context=None) -> None:
        if self._prepared == episode_idx:
            return
        if self._prepared is not None:
            raise RuntimeError("cannot prepare another episode before resetting the client")
        self.begin_episode(episode_idx)
        self._prepared = episode_idx

    def requested_context_fields(self, available_fields, *, episode_idx: int):
        return ()

    def infer_batch(self, obs, instruction, *, env_ids: list[int]):
        if self.episode_idx is None:
            raise RuntimeError("begin_episode must precede infer_batch")
        if not isinstance(instruction, str):
            instruction = instruction_text(instruction)
        result = {}
        for env_id in env_ids:
            if env_id not in self._live:
                started = self.conn.request(
                    {
                        "type": "start",
                        "action_space": "droid_rel_ik",
                        "instruction": instruction or self.instruction,
                        "limits": self.limits,
                        "time_mode": self.time_mode,
                        "droid": self.droid,
                    }
                )
                self._live[env_id] = started["session_id"]
            reply = self.conn.request(
                {
                    "type": "step",
                    "session_id": self._live[env_id],
                    "observation": _env_slice(obs, env_id),
                }
            )
            result[env_id] = {"action": reply["actions"][0], "viz": None}
        return result

    def reset(self, *, env_id=None):
        ids = list(self._live) if env_id is None else [env_id]
        for eid in ids:
            if eid in self._live:
                self._ending[eid] = self._live.pop(eid)
        if env_id is None:
            self._prepared = None

    def record_grades(self, grades, *, episode_idx: int):
        for grade in grades:
            eid = int(grade["env_id"])
            sid = self._ending.pop(eid, None)
            if sid is None:
                raise RuntimeError(f"no ended session for env {eid}")
            self.summaries[episode_idx, eid] = self.conn.request(
                {"type": "end", "session_id": sid, "success": bool(grade["success"])}
            )

    def _end_all(self):
        self.reset()
        for eid, sid in list(self._ending.items()):
            self.summaries[self.episode_idx, eid] = self.conn.request(
                {"type": "end", "session_id": sid}
            )
            del self._ending[eid]

    def close(self):
        self._end_all()
        self.conn.close()
