"""The policy interface, and what a policy returns.

A policy sees an observation plus whatever feedback the harness produced from its last
decision, and returns one decision. The loop, not the policy, owns budgets, tracing and
termination. Interface levels (L0, L1, L2) are policies; so is the scripted reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from rung.contracts.action import TCPTarget
from rung.contracts.capabilities import EmbodimentCapabilities
from rung.contracts.observation import Observation


class Verdict(StrEnum):
    """How a policy chose to stop. Distinct from why an episode ended; see ``loop.EndReason``."""

    CONTINUE = "continue"
    DONE = "done"
    GIVE_UP = "give_up"


@dataclass
class Decision:
    """One policy step.

    ``targets`` is what to execute, in order, and is empty when the policy stops or only
    looked. ``note`` is the one or two sentences the policy must write for the human
    reading the transcript; the loop records it and shows it live. ``kind`` names what the
    policy did in its own vocabulary ("move_to", "grasp", "point", "take_pic") so failures
    can be grouped without parsing prompts.
    """

    kind: str
    note: str
    targets: list[TCPTarget] = field(default_factory=list)
    verdict: Verdict = Verdict.CONTINUE
    reason: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    """Level-specific detail worth tracing: the raw tool call, the chosen point, the
    grasp candidate id."""


@dataclass
class Feedback:
    """What the harness tells the policy about its last decision.

    ``lines`` are rendered into the next prompt in order. ``structured`` carries the same
    facts for the trace and for programmatic policies.
    """

    lines: list[str] = field(default_factory=list)
    structured: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return "\n".join(self.lines)


@runtime_checkable
class Policy(Protocol):
    """Anything that can drive an embodiment.

    ``name`` appears in traces and result tables. ``interface`` is the rung ("L0", "L1",
    "L2", "scripted"), and is what the controlled track varies.
    """

    @property
    def name(self) -> str: ...

    @property
    def interface(self) -> str: ...

    def reset(self, instruction: str, capabilities: EmbodimentCapabilities) -> None:
        """Start a new episode. Clears any per-episode memory."""
        ...

    def act(self, observation: Observation, feedback: Feedback) -> Decision:
        """Choose the next decision."""
        ...


@dataclass
class ModelUsage:
    """Per-decision accounting, summed by the loop. Backends fill this in; a policy with
    no model (the scripted reference) leaves it at zero."""

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    dollars: float = 0.0
    latency_s: float = 0.0
    calls: int = 0
    unknown_cost_calls: int = 0
    """Attempts whose billable usage is unavailable. Dollars are a lower bound if nonzero."""

    def add(self, other: ModelUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.thinking_tokens += other.thinking_tokens
        self.cached_tokens += other.cached_tokens
        self.dollars += other.dollars
        self.latency_s += other.latency_s
        self.calls += other.calls
        self.unknown_cost_calls += other.unknown_cost_calls
