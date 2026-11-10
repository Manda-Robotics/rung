"""The model seam: one request shape, one response shape, one method.

Vendor differences live behind this interface and nowhere else. Interface levels never
branch on the model, so a comparison at a fixed level varies only the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rung.policy import ModelUsage


@dataclass
class TextBlock:
    text: str


@dataclass
class ImageBlock:
    data: bytes
    mime: str = "image/png"
    label: str | None = None


Block = TextBlock | ImageBlock


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: list[Block] = field(default_factory=list)

    @staticmethod
    def user(*blocks: Block) -> Message:
        return Message("user", list(blocks))


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema for the arguments object."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""

    @staticmethod
    def parse(name: str, raw: str | dict, call_id: str = "") -> ToolCall:
        if isinstance(raw, dict):
            return ToolCall(name, raw, call_id)
        try:
            args = json.loads(raw or "{}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tool arguments for {name} are not valid JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise ValueError(f"tool arguments for {name} must be a JSON object")
        return ToolCall(name, args, call_id)


@dataclass
class ModelRequest:
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    effort: str | None = None
    max_tokens: int = 2048
    force_json: dict[str, Any] | None = None
    """A JSON Schema, for levels that want a structured answer rather than a tool call."""


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thoughts: str = ""
    """The vendor's summary of the model's reasoning, when offered. Recorded so the trace
    carries the reasoning as well as the model's written note."""
    usage: ModelUsage = field(default_factory=ModelUsage)
    stop_reason: str | None = None
    raw: Any = None
    refusal: str | None = None

    @property
    def call(self) -> ToolCall | None:
        return self.tool_calls[0] if self.tool_calls else None


@runtime_checkable
class ModelBackend(Protocol):
    """One call, one response. Retries and vendor quirks are handled by the backend."""

    @property
    def model(self) -> str: ...

    @property
    def vendor(self) -> str: ...

    def complete(self, request: ModelRequest) -> ModelResponse: ...


class BackendError(RuntimeError):
    """A call that could not be completed. The loop turns this into a policy error."""
