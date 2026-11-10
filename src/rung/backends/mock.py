"""A model backend with no model: scripted or programmable responses.

Used by interface-level tests and by replay. Requests are recorded so a test can assert on
exactly what a policy sent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from rung.backends.base import (
    BackendError,
    ImageBlock,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from rung.policy import ModelUsage


class MockBackend:
    """Replays a fixed sequence of responses, or computes one from the request.

    ``responses`` may hold ``ModelResponse``, ``ToolCall`` (wrapped automatically), or a
    callable taking the request. Running past the end raises: a policy that asks for more
    turns than scripted fails the test.
    """

    vendor = "mock"

    def __init__(
        self,
        responses: Iterable[ModelResponse | ToolCall | Callable[[ModelRequest], ModelResponse]]
        | None = None,
        model: str = "mock-model",
        tokens_per_call: tuple[int, int] = (900, 60),
    ):
        self._responses = list(responses or [])
        self._model = model
        self._tokens = tokens_per_call
        self.requests: list[ModelRequest] = []
        self.calls = 0

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.calls >= len(self._responses):
            raise BackendError(
                f"the mock backend has {len(self._responses)} scripted responses and the "
                f"policy asked for call {self.calls + 1}"
            )
        item = self._responses[self.calls]
        self.calls += 1
        if callable(item):
            response = item(request)
        elif isinstance(item, ToolCall):
            response = ModelResponse(tool_calls=[item])
        else:
            response = item
        if response.usage.calls == 0:
            response.usage = ModelUsage(
                input_tokens=self._tokens[0],
                output_tokens=self._tokens[1],
                calls=1,
                latency_s=0.0,
            )
        return response

    # ---- helpers for assertions ------------------------------------------

    def last_text(self) -> str:
        """The system prompt and every text block of the last request, joined."""
        if not self.requests:
            return ""
        parts = [self.requests[-1].system]
        for message in self.requests[-1].messages:
            parts += [b.text for b in message.content if isinstance(b, TextBlock)]
        return "\n".join(parts)

    def last_images(self) -> list[ImageBlock]:
        if not self.requests:
            return []
        return [
            b
            for message in self.requests[-1].messages
            for b in message.content
            if isinstance(b, ImageBlock)
        ]

    def tool_names(self) -> list[str]:
        return [t.name for t in self.requests[-1].tools] if self.requests else []


def call(name: str, **arguments) -> ToolCall:
    """Terse constructor for scripted tool calls in tests."""
    return ToolCall(name, dict(arguments))
