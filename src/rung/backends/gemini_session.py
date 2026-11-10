"""Native, stateless Gemini Interactions session for direct-agent development.

Keep complete provider steps (including opaque thought signatures) and tool receipts.
Only older user images are elided; provider steps and tool transactions stay intact.
No request racing, fallback model, or SDK retry is permitted here.
Prices are supplied explicitly, so an unknown model can never masquerade as free.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from rung.backends.base import ImageBlock, ModelResponse, TextBlock, ToolCall, ToolSpec
from rung.backends.native_session import SessionLedger, TokenPrices
from rung.policy import ModelUsage


class GeminiSession(SessionLedger):
    vendor = "gemini"

    def __init__(
        self,
        model: str,
        prices: TokenPrices,
        *,
        client: Any = None,
        api_key: str | None = None,
        trace_dir: Path | None = None,
        max_tokens: int = 4096,
        effort: str = "low",
        timeout_s: float = 90,
        retain_image_turns: int = 2,
        transport: str = "rest",
    ):
        if retain_image_turns < 1:
            raise ValueError("must retain at least the current image turn")
        injected = client is not None
        if client is None and transport == "rest":
            from rung.backends.native_http import GeminiHTTPClient

            client = GeminiHTTPClient(api_key)
        elif client is None:
            raise ValueError(
                "native policy requires one-shot REST; inject SDK only for diagnostics"
            )
        self.transport = "injected" if injected else transport
        self.client, self.model, self.prices = client, model, prices
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.max_tokens, self.effort, self.timeout_s = max_tokens, effort, timeout_s
        self.retain_image_turns = retain_image_turns
        self.reset()

    def resolve(self, receipts: dict[str, dict]):
        if set(receipts) != {c.id for c in self.pending}:
            raise ValueError("receipt IDs must match all outstanding calls exactly")
        self._write(
            "receipts",
            {"receipts": receipts, "calls": [{"id": c.id, "name": c.name} for c in self.pending]},
        )
        for call in self.pending:
            self.history.append(
                {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "is_error": bool(receipts[call.id].get("error")),
                    "result": [
                        {"type": "text", "text": json.dumps(receipts[call.id], allow_nan=False)}
                    ],
                }
            )
        self.pending = []

    def complete(
        self, system: str, blocks: list[TextBlock | ImageBlock], tools: list[ToolSpec]
    ) -> ModelResponse:
        if self.pending:
            raise ValueError("resolve tool calls before the next model request")
        content = []
        for block in blocks:
            if isinstance(block, TextBlock):
                content.append({"type": "text", "text": block.text})
            else:
                content.append(
                    {
                        "type": "image",
                        "mime_type": block.mime,
                        "data": base64.b64encode(block.data).decode(),
                    }
                )
        user = {"type": "user_input", "content": content}
        history = copy.deepcopy([*self.history, user])
        observation_indices = [i for i, step in enumerate(history) if step["type"] == "user_input"]
        for i in observation_indices[: -self.retain_image_turns]:
            history[i]["content"] = [
                {
                    "type": "text",
                    "text": "Older camera image omitted from active context; "
                    "original retained in request archive. SHA256="
                    + hashlib.sha256(base64.b64decode(part["data"])).hexdigest(),
                }
                if part["type"] == "image"
                else part
                for part in history[i]["content"]
            ]
        request = {
            "model": self.model,
            "store": False,
            "system_instruction": system,
            "input": history,
            "tools": [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ],
            "generation_config": {
                "max_output_tokens": self.max_tokens,
                "thinking_level": self.effort,
            },
        }
        if tools:
            request["generation_config"]["tool_choice"] = "any"
        self.turn += 1
        session_id = self.session_id
        identity = (session_id, self.turn)
        self._write(
            "request",
            {
                "request": request,
                "prices": vars(self.prices),
                "transport": self.transport,
                "retain_image_turns": self.retain_image_turns,
            },
        )
        started = time.monotonic()
        try:
            reply = self.client.interactions.create(**request, timeout=self.timeout_s)
        except Exception as exc:
            # Exceptions may contain request URLs/headers; persist only their class.
            self._write(
                "error",
                {
                    "error_type": type(exc).__name__,
                    "latency_s": time.monotonic() - started,
                    "usage_known": False,
                },
                identity=identity,
            )
            raise
        raw = reply if isinstance(reply, dict) else reply.model_dump(mode="json")
        elapsed = time.monotonic() - started
        if session_id != self.session_id:
            if self.trace_dir:
                path = self.trace_dir / f"{session_id}-late-{uuid.uuid4().hex}.json"
                path.write_text(json.dumps({"response": raw, "latency_s": elapsed}))
            raise ValueError("late response from reset session; cannot execute")
        self._write("response", {"response": raw, "latency_s": elapsed})
        steps = raw.get("steps") or []
        self.history.extend([user, *copy.deepcopy(steps)])
        calls = [
            ToolCall(s["name"], s["arguments"], s["id"])
            for s in steps
            if s["type"] == "function_call"
        ]
        if any(not c.id for c in calls) or len({c.id for c in calls}) != len(calls):
            raise ValueError("provider returned missing or duplicate tool IDs")
        self.pending = calls
        self.last_status = raw.get("status")
        u = raw.get("usage") or {}
        if any(k not in u for k in ("total_input_tokens", "total_output_tokens")):
            raise ValueError("provider usage missing; cost cannot be treated as zero")
        inp, out = u["total_input_tokens"], u["total_output_tokens"]
        cached, thought = u.get("total_cached_tokens", 0), u.get("total_thought_tokens", 0)
        # Interactions separates thought tokens from output tokens. Bill both.
        input_rate, output_rate, cached_rate = self.prices.rates_for(inp)
        dollars = (
            (inp - cached) * input_rate + cached * cached_rate + (out + thought) * output_rate
        ) / 1e6
        usage = ModelUsage(
            input_tokens=inp,
            output_tokens=out,
            thinking_tokens=thought,
            cached_tokens=cached,
            dollars=dollars,
            latency_s=elapsed,
            calls=1,
        )
        return ModelResponse(tool_calls=calls, usage=usage, raw=raw, stop_reason=self.last_status)
