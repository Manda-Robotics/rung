"""The embodiment interface. Adapters implement this; the core only ever sees it.

An embodiment owns the world and the robot. It does not know what a policy is, what a
model is, or which interface level is driving it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rung.contracts.action import ExecutionReport, TCPTarget
from rung.contracts.capabilities import EmbodimentCapabilities
from rung.contracts.observation import Observation


class EpisodeStopped(Exception):
    """The external evaluator ended the episode while a command was pending."""


@runtime_checkable
class Embodiment(Protocol):
    """Body plus world.

    ``execute`` plays one target to completion, expiry, or supersession, and returns what
    happened. It never raises for a normal failure (unreachable, blocked, expired); those
    are outcomes in the report. It raises only for contract violations and broken rigs.
    """

    @property
    def capabilities(self) -> EmbodimentCapabilities: ...

    def reset(self, episode_seed: int | None = None) -> Observation:
        """Start a new episode. Returns the first observation."""
        ...

    def observe(self) -> Observation:
        """Current observation without advancing the world."""
        ...

    def execute(self, target: TCPTarget) -> ExecutionReport:
        """Play one TCP target. Advances simulated time."""
        ...


@runtime_checkable
class Gradeable(Protocol):
    """An embodiment that can also score the episode it is running.

    Kept separate from ``Embodiment`` because grading is a benchmark concern. The grader's
    verdict never reaches a primitive result or a prompt; see docs/protocol.md, the
    diagnostic boundary rule.
    """

    def grade(self) -> tuple[bool, dict]:
        """Returns ``(success, detail)``."""
        ...

    @property
    def instruction(self) -> str: ...
