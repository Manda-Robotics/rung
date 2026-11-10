"""Traces: one JSONL row per event, numpy-safe, append-only.

Event kinds, in the order they normally appear inside one decision:

* ``observation``   what the policy was given (image references, catalog, memory, prompt hash)
* ``model_call``    request summary, raw response, usage, latency, dollars
* ``decision``      the parsed decision (tool call or JSON) and the interface level
* ``attempt``       the target(s) the harness asked the adapter to execute
* ``intervention``  a clamp, rejection, retreat or override applied by the harness
* ``execution``     the adapter's report for a target
* ``primitive``     result of an L2 primitive (grasp, place, ...) with its reason
* ``episode_end``   which limit ended the episode, the grader's verdict, totals
* ``policy_end``    as ``episode_end``, when no grader is attached and grading is external

Servo and interpolation steps are ``attempt``/``execution`` rows flagged ``servo=True``
so model-call counts are never inflated by them.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

EVENT_KINDS = (
    "observation",
    "model_call",
    "decision",
    "attempt",
    "intervention",
    "execution",
    "primitive",
    "policy_end",
    "execution_error",
    "episode_end",
    # What the harness does between model calls, for the transcript.
    "setup",
    "model_request",
    "model_response",
    "grounding",
    "perception",
    "plan",
    "check",
)


class _Encoder(json.JSONEncoder):
    def default(self, o: Any):  # noqa: ANN401
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, Enum):
            return o.value
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o)
        # Torch tensors, duck-typed so the contracts layer does not import torch. Detach
        # first (tensors requiring grad refuse conversion) and move to CPU (CUDA tensors
        # cannot be read directly).
        if hasattr(o, "detach") and hasattr(o, "cpu") and hasattr(o, "tolist"):
            return o.detach().cpu().tolist()
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        return super().default(o)


def to_jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, cls=_Encoder))


@dataclass
class TraceEvent:
    kind: str
    t_sim: float
    payload: dict[str, Any]
    episode: str
    env_id: int = 0
    t_wall: float = field(default_factory=time.time)
    seq: int = 0

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown trace event kind {self.kind!r}")


class TraceWriter:
    """Append-only JSONL writer. Thread-safe. ``None`` path means events are kept in memory only."""

    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._seq = 0
        self.events: list[TraceEvent] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: TraceEvent) -> TraceEvent:
        with self._lock:
            self._seq += 1
            event.seq = self._seq
            self.events.append(event)
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dataclasses.asdict(event), cls=_Encoder) + "\n")
        return event

    def emit(
        self, kind: str, t_sim: float, episode: str, env_id: int = 0, **payload: Any
    ) -> TraceEvent:
        return self.write(
            TraceEvent(kind=kind, t_sim=t_sim, payload=payload, episode=episode, env_id=env_id)
        )

    def count(self, kind: str, episode: str | None = None, servo: bool | None = None) -> int:
        n = 0
        for e in self.events:
            if e.kind != kind:
                continue
            if episode is not None and e.episode != episode:
                continue
            if servo is not None and bool(e.payload.get("servo", False)) != servo:
                continue
            n += 1
        return n


def read_trace(path: str | os.PathLike) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@dataclass
class RunProvenance:
    """Written once per run next to the traces. Everything needed to reproduce the run."""

    run_id: str
    started_wall: float
    harness_revision: str
    embodiment: str
    embodiment_revision: str | None
    interface: str
    model: str | None
    effort: str | None
    prompt_hashes: dict[str, str]
    perception_version: str | None
    grasp_version: str | None
    task_role: str | None
    limits: dict[str, Any]
    initial_state_audit: dict[str, Any] = field(default_factory=dict)
    prices_version: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def write(self, path: str | os.PathLike) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dataclasses.asdict(self), cls=_Encoder, indent=2), encoding="utf-8")
