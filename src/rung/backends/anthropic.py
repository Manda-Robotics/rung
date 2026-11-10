"""Anthropic backend: Claude Fable 5.1 and the Opus family.

API notes:

* Thinking is always on for Fable 5.1. Sending an explicit ``thinking`` configuration is
  rejected, so depth is controlled with ``output_config.effort`` instead.
* Forced tool choice returns a 400 on Fable 5.1. The prompt asks for one tool call and
  ``tool_choice`` stays ``auto``.
* A safety classifier can decline with HTTP 200 and ``stop_reason == "refusal"``, so
  ``stop_reason`` is checked before the content is read, and server-side fallbacks are
  enabled by default.
* Thinking tokens are billed as output and are reported separately, so they are added to
  the output count before the cost is computed.
"""

from __future__ import annotations

import base64
import time

from rung.backends.base import (
    BackendError,
    ImageBlock,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from rung.backends.prices import dollars
from rung.policy import ModelUsage

DEFAULT_MODEL = "claude-fable-5-1"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicBackend:
    vendor = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client=None,
        max_retries: int = 3,
        timeout_s: float = 120.0,
        server_side_fallbacks: bool = True,
    ):
        self._model = model
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.server_side_fallbacks = server_side_fallbacks
        if client is not None:
            self.client = client
        else:
            try:
                import anthropic
            except ImportError as exc:
                raise BackendError(
                    "the anthropic extra is not installed: pip install 'rung[anthropic]'"
                ) from exc
            self.client = anthropic.Anthropic(timeout=timeout_s)

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = {
            "model": self._model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [
                {"role": m.role, "content": [_block(b) for b in m.content]}
                for m in request.messages
            ],
        }
        if request.tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in request.tools
            ]
            payload["tool_choice"] = {"type": "auto"}
        if request.effort:
            payload["output_config"] = {"effort": request.effort}
        if self.server_side_fallbacks:
            payload["betas"] = [FALLBACK_BETA]
            payload["fallbacks"] = "default"

        started = time.time()
        message = self._send(payload)
        latency = time.time() - started

        usage = self._usage(message, latency)
        stop = getattr(message, "stop_reason", None)
        if stop == "refusal":
            details = getattr(message, "stop_details", None)
            return ModelResponse(
                usage=usage,
                stop_reason=stop,
                raw=message,
                refusal=getattr(details, "explanation", None) or "the model declined the request",
            )

        text_parts, thought_parts, calls = [], [], []
        for block in getattr(message, "content", []) or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "thinking" and getattr(block, "thinking", None):
                thought_parts.append(block.thinking)
            elif kind == "tool_use":
                calls.append(ToolCall.parse(block.name, block.input, getattr(block, "id", "")))
        return ModelResponse(
            text="\n".join(text_parts),
            thoughts="\n".join(thought_parts),
            tool_calls=calls,
            usage=usage,
            stop_reason=stop,
            raw=message,
        )

    def _send(self, payload: dict):
        create = (
            self.client.beta.messages.create
            if self.server_side_fallbacks
            else self.client.messages.create
        )
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return create(**payload)
            except Exception as exc:  # retried below unless it is clearly terminal
                last = exc
                if _is_terminal(exc) or attempt == self.max_retries - 1:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise BackendError(f"anthropic call failed: {last}") from last

    def _usage(self, message, latency: float) -> ModelUsage:
        raw = getattr(message, "usage", None)
        inp = int(getattr(raw, "input_tokens", 0) or 0)
        out = int(getattr(raw, "output_tokens", 0) or 0)
        cached = int(getattr(raw, "cache_read_input_tokens", 0) or 0)
        thinking = int(getattr(raw, "thinking_tokens", 0) or 0)
        return ModelUsage(
            input_tokens=inp + cached,
            output_tokens=out,
            thinking_tokens=thinking,
            cached_tokens=cached,
            dollars=dollars(self._model, inp + cached, out + thinking, cached),
            latency_s=latency,
            calls=1,
        )


def _block(block) -> dict:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.mime,
                "data": base64.b64encode(block.data).decode(),
            },
        }
    raise BackendError(f"cannot send a {type(block).__name__} to Anthropic")


def _is_terminal(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in (400, 401, 403, 404)
