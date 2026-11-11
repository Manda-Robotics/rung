"""Drive Rung with your own model: any OpenAI-compatible chat-completions endpoint.

vLLM, SGLang, Ollama, LM Studio and most hosted open-weight providers serve
``/v1/chat/completions`` with tool calling. This file is a complete backend for them in
about 80 lines. It runs the L0 interface on the mock tabletop:

    # no model needed: a local stub endpoint answers with a fixed tool call
    python examples/own_model.py --selftest

    # your server, for example:
    #   vllm serve Qwen/Qwen3-VL-8B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
    python examples/own_model.py --endpoint http://localhost:8000/v1 \
        --model Qwen/Qwen3-VL-8B-Instruct

Set OWN_MODEL_API_KEY if your endpoint wants a bearer token. Dollars are reported as 0:
the price table only covers hosted frontier models, so pass your own cost if it matters.
See docs/own-model.md for how the seam works.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from rung.backends.base import (
    BackendError,
    ImageBlock,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from rung.contracts.trace import TraceWriter
from rung.interfaces import L0Config, L0Policy
from rung.loop import Limits, run_episode
from rung.mock import SCENES
from rung.policy import ModelUsage


class ChatCompletionsBackend:
    """A ``rung.backends.ModelBackend`` for OpenAI-compatible chat-completions servers."""

    vendor = "openai-compatible"

    def __init__(
        self, endpoint: str, model: str, api_key: str | None = None, timeout_s: float = 120
    ):
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self._model, self.timeout_s = model, timeout_s
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: ModelRequest) -> ModelResponse:
        messages = [{"role": "system", "content": request.system}]
        for message in request.messages:
            parts = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    if block.label:
                        parts.append({"type": "text", "text": block.label})
                    data = base64.b64encode(block.data).decode()
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{block.mime};base64,{data}"},
                        }
                    )
            messages.append({"role": message.role, "content": parts})
        body = {"model": self._model, "messages": messages, "max_tokens": request.max_tokens}
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
            body["tool_choice"] = "required"
        started = time.monotonic()
        try:
            req = urllib.request.Request(
                self.url, json.dumps(body).encode(), self.headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                reply = json.load(response)
        except OSError as exc:  # URLError, HTTPError and timeouts
            raise BackendError(f"endpoint call failed: {exc}") from None
        choice = reply["choices"][0]["message"]
        calls = [
            ToolCall.parse(
                c["function"]["name"], c["function"].get("arguments") or "{}", c.get("id", "")
            )
            for c in choice.get("tool_calls") or []
        ]
        usage = reply.get("usage") or {}
        return ModelResponse(
            text=choice.get("content") or "",
            tool_calls=calls,
            usage=ModelUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_s=time.monotonic() - started,
                calls=1,
            ),
            stop_reason=reply["choices"][0].get("finish_reason"),
            raw=reply,
        )


class _Stub(BaseHTTPRequestHandler):
    """A stand-in endpoint for --selftest: it always ends the trial with `done`."""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        call = {"id": "c1", "type": "function", "function": {"name": "done", "arguments": "{}"}}
        reply = {
            "choices": [
                {
                    "message": {"content": "stub", "tool_calls": [call]},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", help="base URL ending in /v1")
    ap.add_argument("--model", default="stub")
    ap.add_argument("--scene", default="banana_in_bowl", choices=sorted(SCENES))
    ap.add_argument(
        "--selftest", action="store_true", help="serve a local stub endpoint; no model needed"
    )
    a = ap.parse_args()
    if a.selftest:
        server = HTTPServer(("127.0.0.1", 0), _Stub)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        a.endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    if not a.endpoint:
        ap.error("pass --endpoint, or --selftest to try it without a model")
    backend = ChatCompletionsBackend(a.endpoint, a.model, os.environ.get("OWN_MODEL_API_KEY"))
    result = run_episode(
        SCENES[a.scene](),
        L0Policy(backend, L0Config()),
        task=a.scene,
        limits=Limits(horizon_s=120, max_decisions=30),
        trace=TraceWriter(None),
        episode_seed=0,
    )
    print(result.summary())
    if a.selftest:
        print(
            "Self-test: the stub ends every trial at once, so a failed grade is expected. "
            "The backend plumbing works."
        )
        return 0 if result.usage.calls == 1 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
