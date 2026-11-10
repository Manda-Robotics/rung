"""Native Responses session with encrypted reasoning and complete tool transactions.

Uses the same DirectPolicy contract as Gemini; it does not claim any particular
Astra release is available through this API. Live model identity must be qualified.
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
from rung.backends.native_http import OpenAIHTTPClient
from rung.backends.native_session import SessionLedger, TokenPrices
from rung.policy import ModelUsage


def strict_parameters(schema):
    """The same tool schema in the form Responses strict mode accepts.

    Strict mode requires every property to be listed in ``required`` and
    ``additionalProperties: false`` on every object. Optional properties (the guarded
    sequence's per-tool step fields) become required-but-nullable, so the model sends
    ``null`` for fields a step does not use; the policy drops nulls before validation.
    Semantics are unchanged; only the wire form differs from the CLI route.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        required = set(out.get("required") or [])
        properties = {}
        for name, prop in out["properties"].items():
            prop = strict_parameters(prop)
            if name not in required:
                prop = {"anyOf": [prop, {"type": "null"}]}
            properties[name] = prop
        out["properties"] = properties
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    if "items" in out:
        out["items"] = strict_parameters(out["items"])
    if "anyOf" in out:
        out["anyOf"] = [strict_parameters(x) for x in out["anyOf"]]
    return out


class OpenAISession(SessionLedger):
    vendor = "openai"

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
    ):
        if retain_image_turns < 1:
            raise ValueError("must retain at least the current image turn")
        self.client = client if client is not None else OpenAIHTTPClient(api_key)
        self.transport = "injected" if client is not None else "native-rest-responses"
        self.model, self.prices = model, prices
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
                    "type": "function_call_output",
                    "call_id": call.id,
                    "output": json.dumps(receipts[call.id], allow_nan=False),
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
                content.append({"type": "input_text", "text": block.text})
            else:
                content.append(
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": f"data:{block.mime};base64,"
                        + base64.b64encode(block.data).decode(),
                    }
                )
        user = {"role": "user", "content": content}
        history = copy.deepcopy([*self.history, user])
        users = [i for i, item in enumerate(history) if item.get("role") == "user"]
        for i in users[: -self.retain_image_turns]:
            history[i]["content"] = [
                {
                    "type": "input_text",
                    "text": "Older camera image omitted from active context; "
                    "original retained in request archive. SHA256="
                    + hashlib.sha256(
                        base64.b64decode(part["image_url"].split(",", 1)[1])
                    ).hexdigest(),
                }
                if part["type"] == "input_image"
                else part
                for part in history[i]["content"]
            ]
        request = {
            "model": self.model,
            "store": False,
            "instructions": system,
            "input": history,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": self.effort},
            "max_output_tokens": self.max_tokens,
            "parallel_tool_calls": False,
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": strict_parameters(t.parameters),
                    "strict": True,
                }
                for t in tools
            ],
        }
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
            reply = self.client.responses.create(**request, timeout=self.timeout_s)
        except Exception as exc:
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
        output = raw.get("output") or []
        self.history.extend([user, *copy.deepcopy(output)])
        calls = [
            ToolCall(s["name"], json.loads(s["arguments"]), s["call_id"])
            for s in output
            if s["type"] == "function_call"
        ]
        if any(not c.id for c in calls) or len({c.id for c in calls}) != len(calls):
            raise ValueError("provider returned missing or duplicate tool IDs")
        self.pending = calls
        self.last_status = raw.get("status")
        u = raw.get("usage") or {}
        if any(k not in u for k in ("input_tokens", "output_tokens")):
            raise ValueError("provider usage missing; cost cannot be treated as zero")
        inp, out = u["input_tokens"], u["output_tokens"]
        cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
        thought = (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        # Responses output_tokens already includes reasoning: do not charge twice.
        dollars = (
            (inp - cached) * self.prices.input_per_million
            + cached * self.prices.cached_per_million
            + out * self.prices.output_per_million
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
