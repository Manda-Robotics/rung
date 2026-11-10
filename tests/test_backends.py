"""Backends: what a vendor response turns into. No network; the clients are faked."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from rung.backends.base import Message, ModelRequest, TextBlock


def _gemini_response():
    parts = [
        NS(
            function_call=None,
            thought=True,
            text="The onion sits by the bowl rim; a top grasp fits.",
        ),
        NS(
            function_call=NS(name="grasp", args={"object_id": "onion_0", "note": "Pick it up."}),
            text=None,
            thought=False,
        ),
    ]
    return NS(
        candidates=[NS(content=NS(parts=parts), finish_reason="STOP")],
        usage_metadata=NS(
            prompt_token_count=1000,
            candidates_token_count=40,
            thoughts_token_count=300,
            cached_content_token_count=0,
        ),
    )


def test_gemini_thought_summaries_are_kept_apart_from_the_answer():
    pytest.importorskip("google.genai")
    from rung.backends.gemini import GeminiBackend

    sent = {}

    class Models:
        def generate_content(self, model, contents, config):
            sent["config"] = config
            return _gemini_response()

    backend = GeminiBackend("gemini-3.7-flash", client=NS(models=Models()), hedge_after_s=None)
    response = backend.complete(ModelRequest(system="s", messages=[Message.user(TextBlock("hi"))]))
    assert sent["config"].thinking_config.include_thoughts is True
    assert "top grasp fits" in response.thoughts
    assert response.text == ""
    assert response.call.name == "grasp"
    assert response.usage.thinking_tokens == 300


def test_the_trace_keeps_per_call_usage_and_thoughts():
    import dataclasses

    from rung.loop import _traceable
    from rung.policy import ModelUsage

    payload = {
        "usage": ModelUsage(calls=1, dollars=0.01, latency_s=2.5),
        "tool": "grasp",
        "thoughts": "because",
    }
    out = _traceable(payload)
    assert out["usage"] == dataclasses.asdict(ModelUsage(calls=1, dollars=0.01, latency_s=2.5))
    assert out["thoughts"] == "because"


def _slow_then_fast_models(delays):
    import threading
    import time

    lock = threading.Lock()
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            with lock:
                n = len(calls)
                calls.append(n)
            time.sleep(delays[min(n, len(delays) - 1)])
            return _gemini_response()

    return Models(), calls


def test_a_stalled_gemini_call_is_hedged_and_both_requests_are_billed():
    pytest.importorskip("google.genai")
    import time

    from rung.backends.gemini import GeminiBackend

    models, calls = _slow_then_fast_models([2.0, 0.05])
    backend = GeminiBackend("gemini-3.7-flash", client=NS(models=models), hedge_after_s=0.1)
    started = time.time()
    response = backend.complete(ModelRequest(system="s", messages=[Message.user(TextBlock("hi"))]))
    assert time.time() - started < 1.0
    assert len(calls) == 2
    assert response.usage.calls == 2
    single = GeminiBackend(
        "gemini-3.7-flash", client=NS(models=_slow_then_fast_models([0.0])[0]), hedge_after_s=None
    )
    one = single.complete(ModelRequest(system="s", messages=[Message.user(TextBlock("hi"))]))
    assert response.usage.dollars == pytest.approx(2 * one.usage.dollars)


def test_a_prompt_gemini_call_is_not_hedged():
    pytest.importorskip("google.genai")
    from rung.backends.gemini import GeminiBackend

    models, calls = _slow_then_fast_models([0.0])
    backend = GeminiBackend("gemini-3.7-flash", client=NS(models=models), hedge_after_s=1.0)
    response = backend.complete(ModelRequest(system="s", messages=[Message.user(TextBlock("hi"))]))
    assert len(calls) == 1 and response.usage.calls == 1


def test_a_zero_hedge_races_two_requests_from_the_start():
    pytest.importorskip("google.genai")
    from rung.backends.gemini import GeminiBackend

    models, calls = _slow_then_fast_models([0.3, 0.3])
    backend = GeminiBackend("gemini-3.7-flash", client=NS(models=models), hedge_after_s=0.0)
    response = backend.complete(ModelRequest(system="s", messages=[Message.user(TextBlock("hi"))]))
    assert len(calls) == 2 and response.usage.calls == 2


def test_a_third_request_goes_out_when_both_raced_requests_stall():
    pytest.importorskip("google.genai")
    import time

    from rung.backends.gemini import GeminiBackend

    models, calls = _slow_then_fast_models([2.0, 2.0, 0.05])
    backend = GeminiBackend(
        "gemini-3.7-flash", client=NS(models=models), hedge_after_s=0.0, rehedge_after_s=0.2
    )
    started = time.time()
    response = backend.complete(ModelRequest(system="s", messages=[Message.user(TextBlock("hi"))]))
    assert time.time() - started < 1.0
    assert len(calls) == 3 and response.usage.calls == 3


def test_single_request_gemini_profile_never_retries_or_races():
    pytest.importorskip("google.genai")
    from rung.backends.base import BackendError
    from rung.backends.gemini import GeminiBackend

    sent = []

    class Models:
        def generate_content(self, **kwargs):
            sent.append(kwargs)
            raise RuntimeError("transient service failure")

    backend = GeminiBackend.single_request("gemini-4-argon", client=NS(models=Models()))
    assert backend.hedge_after_s is None and backend.rehedge_after_s is None
    assert backend.sdk_retry_attempts == 1
    with pytest.raises(BackendError):
        backend.complete(ModelRequest(system="public", messages=[]))
    assert len(sent) == 1
    with pytest.raises(BackendError, match="known API price"):
        GeminiBackend.single_request("unpriced-model", client=NS(models=Models()))
    assert len(sent) == 1


def test_argon_price_includes_thinking_and_implicit_cache():
    from rung.backends.prices import dollars

    assert dollars("gemini-4-argon", 10000, 2000, 1000) == pytest.approx(0.018 + 0.020 + 0.0001)


def test_single_request_profile_disables_sdk_transport_retries(monkeypatch):
    genai = pytest.importorskip("google.genai")
    from rung.backends.gemini import GeminiBackend

    captured = {}

    def client(**kwargs):
        captured.update(kwargs)
        return NS()

    monkeypatch.setattr(genai, "Client", client)
    GeminiBackend.single_request("gemini-4-argon", timeout_s=17)
    assert captured["http_options"].retry_options.attempts == 1
    assert captured["http_options"].timeout == 17000
