"""OpenAI backend: GPT-6 Astra through the Responses API.

API notes:

* Tools are flat in Responses (``{"type": "function", "name": ..., "parameters": ...}``),
  not nested under a ``function`` key as in Chat Completions.
* Images go in as ``input_image`` with a data URL.
* Reasoning depth is ``reasoning={"effort": ...}``; reasoning tokens are billed as output
  and are reported inside ``output_tokens_details``.
* Above 272k input tokens the whole request is repriced. This is not modelled; prompts
  stay far below the threshold, and a long-context run would under-report cost.
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

DEFAULT_MODEL = "gpt-6-astra"


class OpenAIBackend:
    vendor = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client=None,
        max_retries: int = 3,
        timeout_s: float = 120.0,
    ):
        self._model = model
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        if client is not None:
            self.client = client
        else:
            try:
                import openai
            except ImportError as exc:
                raise BackendError(
                    "the openai extra is not installed: pip install 'rung[openai]'"
                ) from exc
            self.client = openai.OpenAI(timeout=timeout_s)

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: ModelRequest) -> ModelResponse:
        content = []
        for message in request.messages:
            for block in message.content:
                content.append(_part(block))
        payload = {
            "model": self._model,
            "instructions": request.system,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": request.max_tokens,
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in request.tools
            ]
            payload["tool_choice"] = "auto"
        if request.effort:
            payload["reasoning"] = {"effort": request.effort}

        started = time.time()
        response = self._send(payload)
        latency = time.time() - started

        text_parts, thought_parts, calls = [], [], []
        for item in getattr(response, "output", []) or []:
            kind = getattr(item, "type", None)
            if kind == "function_call":
                calls.append(
                    ToolCall.parse(item.name, item.arguments, getattr(item, "call_id", ""))
                )
            elif kind == "reasoning":
                for summary in getattr(item, "summary", []) or []:
                    if getattr(summary, "text", None):
                        thought_parts.append(summary.text)
            elif kind == "message":
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        text_parts.append(part.text)
                    elif getattr(part, "type", None) == "refusal":
                        return ModelResponse(
                            usage=self._usage(response, latency),
                            raw=response,
                            refusal=getattr(part, "refusal", "the model declined the request"),
                        )
        return ModelResponse(
            text="\n".join(text_parts),
            thoughts="\n".join(thought_parts),
            tool_calls=calls,
            usage=self._usage(response, latency),
            stop_reason=getattr(response, "status", None),
            raw=response,
        )

    def _send(self, payload: dict):
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.client.responses.create(**payload)
            except Exception as exc:
                last = exc
                if getattr(exc, "status_code", None) in (400, 401, 403, 404):
                    break
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise BackendError(f"openai call failed: {last}") from last

    def _usage(self, response, latency: float) -> ModelUsage:
        raw = getattr(response, "usage", None)
        inp = int(getattr(raw, "input_tokens", 0) or 0)
        out = int(getattr(raw, "output_tokens", 0) or 0)
        details = getattr(raw, "output_tokens_details", None)
        thinking = int(getattr(details, "reasoning_tokens", 0) or 0)
        in_details = getattr(raw, "input_tokens_details", None)
        cached = int(getattr(in_details, "cached_tokens", 0) or 0)
        return ModelUsage(
            input_tokens=inp,
            output_tokens=out - thinking if out >= thinking else out,
            thinking_tokens=thinking,
            cached_tokens=cached,
            dollars=dollars(self._model, inp, out, cached),
            latency_s=latency,
            calls=1,
        )


def _part(block) -> dict:
    if isinstance(block, TextBlock):
        return {"type": "input_text", "text": block.text}
    if isinstance(block, ImageBlock):
        encoded = base64.b64encode(block.data).decode()
        return {"type": "input_image", "image_url": f"data:{block.mime};base64,{encoded}"}
    raise BackendError(f"cannot send a {type(block).__name__} to OpenAI")
