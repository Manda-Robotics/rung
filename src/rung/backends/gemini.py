"""Google backend: Gemini 3.7 Flash and Gemini Robotics-ER 2.

API notes:

* Function declarations go in ``config.tools`` as ``[{"function_declarations": [...]}]``.
* Thinking depth is ``thinking_config.thinking_level`` (``low``, ``medium``, ``high``).
  Gemini 3.7 Flash rejects ``minimal``. The default is ``medium``, which the Robotics-ER
  preview recommends for spatial reasoning.
* An image costs 1,120 tokens at the default media resolution, so image input, not
  output, dominates cost on this vendor.
* ``thoughts_token_count`` is reported separately and billed as output.
* Every request carries a client timeout. Without one, a stalled call blocks the episode
  indefinitely.
* Function-declaration parameters accept a subset of JSON Schema. ``additionalProperties``
  and ``default`` are rejected with a 400, so schemas are sanitised on the way out and the
  tool surface stays identical across backends.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from rung.backends.base import (
    BackendError,
    ImageBlock,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from rung.backends.prices import dollars, price_of
from rung.policy import ModelUsage

DEFAULT_MODEL = "gemini-3.7-flash"
EFFORT_TO_LEVEL = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
    None: "medium",
}


class GeminiBackend:
    vendor = "google"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client=None,
        max_retries: int = 3,
        media_resolution: str | None = None,
        timeout_s: float = 120.0,
        hedge_after_s: float | None = 0.0,
        rehedge_after_s: float | None = 8.0,
        sdk_retry_attempts: int | None = None,
    ):
        self._model = model
        self.sdk_retry_attempts = sdk_retry_attempts
        self.hedge_after_s = hedge_after_s
        """Delay in seconds before an identical second request; the first answer wins.
        0 races two requests from the start, None sends one.

        Measured with 24 simultaneous pairs of grounding requests: single-request median
        6.0 s, 1 in 6 over 12 s, worst 61 s. Stalls within a pair are uncorrelated
        (r = -0.2), and the faster of a pair never exceeded 7.0 s. A hedge delayed to 8 s
        still allowed 25 s calls. In real-time mode a stalled call is time the robot stands
        still. Every request sent is billed and counted."""
        self.rehedge_after_s = rehedge_after_s
        """Delay in seconds before a third request, sent if neither earlier request has
        answered. Raced acting calls inside episodes occasionally still take 20 to 58 s;
        a fresh request sent while both are stalled usually completes sooner.

        Each request uses its own client and its own copy of the config. Over 14 raced
        acting calls per configuration, a shared client gave a median of 6.5 s and
        separate clients 4.5 s."""
        self._clients: list = []
        self.max_retries = max_retries
        self.media_resolution = media_resolution
        self.timeout_s = timeout_s
        self._injected = client is not None
        if client is not None:
            self.client = client
        else:
            try:
                from google import genai
            except ImportError as exc:
                raise BackendError(
                    "the gemini extra is not installed: pip install 'rung[gemini]'"
                ) from exc
            from google.genai import types as _types

            if sdk_retry_attempts is not None and not hasattr(_types, "HttpRetryOptions"):
                raise BackendError("single-request profile requires SDK HttpRetryOptions support")
            self.client = genai.Client(
                http_options=_types.HttpOptions(
                    timeout=int(timeout_s * 1000),
                    retry_options=(
                        _types.HttpRetryOptions(attempts=sdk_retry_attempts)
                        if sdk_retry_attempts is not None
                        else None
                    ),
                )
            )

    @classmethod
    def single_request(cls, model: str = DEFAULT_MODEL, *, client=None, timeout_s: float = 120.0):
        """Explicit research profile: no racing, backend retries or SDK retries.

        Injected clients must provide their own one-request transport guarantee.
        This profile does not alter the historical latency-oriented defaults.
        """
        if price_of(model) is None:
            raise BackendError("single-request profile needs an explicit known API price")
        return cls(
            model,
            client=client,
            timeout_s=timeout_s,
            max_retries=1,
            hedge_after_s=None,
            rehedge_after_s=None,
            sdk_retry_attempts=1,
        )

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: ModelRequest) -> ModelResponse:
        from google.genai import types

        parts = []
        for message in request.messages:
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append(types.Part.from_text(text=block.text))
                elif isinstance(block, ImageBlock):
                    parts.append(types.Part.from_bytes(data=block.data, mime_type=block.mime))
                else:
                    raise BackendError(f"cannot send a {type(block).__name__} to Gemini")

        config: dict = {
            "system_instruction": request.system,
            "max_output_tokens": request.max_tokens,
            "thinking_config": types.ThinkingConfig(
                thinking_level=EFFORT_TO_LEVEL.get(request.effort, "medium"),
                include_thoughts=True,
            ),
        }
        if request.tools:
            # Function calls are parsed here. With automatic calling on, the SDK would
            # invoke them itself, unlike every other backend.
            config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )
            config["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name,
                            description=t.description,
                            parameters=_supported_schema(t.parameters),
                        )
                        for t in request.tools
                    ]
                )
            ]
        if self.media_resolution:
            config["media_resolution"] = self.media_resolution

        started = time.time()
        response, sent = self._send_hedged(parts, types.GenerateContentConfig(**config))
        latency = time.time() - started

        text_parts, thought_parts, calls = [], [], []
        for candidate in getattr(response, "candidates", []) or []:
            for part in getattr(getattr(candidate, "content", None), "parts", []) or []:
                fn = getattr(part, "function_call", None)
                if fn is not None:
                    calls.append(ToolCall.parse(fn.name, dict(fn.args or {})))
                elif getattr(part, "text", None):
                    # With include_thoughts the reasoning summary arrives as text parts
                    # flagged thought=True; it is not the answer and must not be read as one.
                    (thought_parts if getattr(part, "thought", False) else text_parts).append(
                        part.text
                    )
        return ModelResponse(
            text="\n".join(text_parts),
            thoughts="\n".join(thought_parts),
            tool_calls=calls,
            usage=self._usage(response, latency, calls=sent),
            stop_reason=_finish_reason(response),
            raw=response,
        )

    def _client(self, i: int):
        """Client for request slot i: the main client for slot 0, then one lazily built
        client per extra slot. An injected client (tests) serves every slot."""
        if i == 0 or self._injected:
            return self.client
        while len(self._clients) < i:
            from google import genai
            from google.genai import types as _types

            self._clients.append(
                genai.Client(
                    http_options=_types.HttpOptions(
                        timeout=int(self.timeout_s * 1000),
                        retry_options=_types.HttpRetryOptions(attempts=self.sdk_retry_attempts)
                        if self.sdk_retry_attempts is not None
                        else None,
                    )
                )
            )
        return self._clients[i - 1]

    def _send_hedged(self, parts, config):
        """Send requests on the hedge schedule and return (response, requests sent).

        The SDK cannot cancel a request, so losing requests finish in daemon workers and
        their answers are discarded."""
        if self.hedge_after_s is None:
            return self._send(parts, config), 1
        import copy

        schedule = [0.0, float(self.hedge_after_s)]
        if self.rehedge_after_s is not None and self.rehedge_after_s > self.hedge_after_s:
            schedule.append(float(self.rehedge_after_s))
        pool = ThreadPoolExecutor(max_workers=len(schedule))
        start = time.monotonic()
        pending: set = set()
        sent = 0
        error: BaseException | None = None
        try:
            while True:
                while sent < len(schedule) and time.monotonic() - start >= schedule[sent]:
                    pending.add(
                        pool.submit(self._send, parts, copy.deepcopy(config), self._client(sent))
                    )
                    sent += 1
                if not pending:
                    raise error if error is not None else BackendError("no request was sent")
                timeout = (
                    None
                    if sent == len(schedule)
                    else max(0.0, schedule[sent] - (time.monotonic() - start))
                )
                done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                for future in done:
                    if future.exception() is None:
                        return future.result(), sent
                    error = future.exception()
                if not pending and sent == len(schedule):
                    raise error
        finally:
            pool.shutdown(wait=False)

    def _send(self, parts, config, client=None):
        client = client or self.client
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return client.models.generate_content(
                    model=self._model, contents=parts, config=config
                )
            except Exception as exc:
                last = exc
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if code in (400, 401, 403, 404) or attempt == self.max_retries - 1:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise BackendError(f"gemini call failed: {last}") from last

    def _usage(self, response, latency: float, calls: int = 1) -> ModelUsage:
        raw = getattr(response, "usage_metadata", None)
        inp = int(getattr(raw, "prompt_token_count", 0) or 0)
        out = int(getattr(raw, "candidates_token_count", 0) or 0)
        thinking = int(getattr(raw, "thoughts_token_count", 0) or 0)
        cached = int(getattr(raw, "cached_content_token_count", 0) or 0)
        return ModelUsage(
            input_tokens=inp,
            output_tokens=out,
            thinking_tokens=thinking,
            cached_tokens=cached,
            # Every raced request is billed. Losing requests report no usage, so each is
            # charged at the winner's.
            dollars=calls * dollars(self._model, inp, out + thinking, cached),
            latency_s=latency,
            calls=calls,
        )


UNSUPPORTED_SCHEMA_KEYS = frozenset({"additionalProperties", "default", "$schema", "title"})


def _supported_schema(schema):
    """Drop JSON Schema keys the Gemini function-declaration parser rejects.

    Recursive, because the keys appear inside ``properties`` and ``items`` too. Everything
    that carries meaning for the model (types, descriptions, required, enum) is kept.
    """
    if isinstance(schema, dict):
        return {
            k: _supported_schema(v) for k, v in schema.items() if k not in UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_supported_schema(v) for v in schema]
    return schema


def _finish_reason(response) -> str | None:
    for candidate in getattr(response, "candidates", []) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            return str(reason)
    return None
