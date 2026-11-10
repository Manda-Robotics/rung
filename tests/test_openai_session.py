from copy import deepcopy
from types import SimpleNamespace

import pytest

from rung.backends.base import ImageBlock, TextBlock
from rung.backends.native_session import TokenPrices
from rung.backends.openai_session import OpenAISession
from rung.interfaces.direct import tools


def response():
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs1", "summary": [], "encrypted_content": "opaque-cipher"},
            {
                "type": "function_call",
                "id": "fc1",
                "call_id": "call1",
                "name": "observe",
                "arguments": '{"note":"look"}',
            },
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 30,
            "input_tokens_details": {"cached_tokens": 40},
            "output_tokens_details": {"reasoning_tokens": 20},
        },
    }


def test_responses_preserves_encrypted_reasoning_and_call_id(tmp_path):
    requests = []

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        return response()

    s = OpenAISession(
        "qualified-model-placeholder",
        TokenPrices(1, 2, 0.1, "test"),
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
        trace_dir=tmp_path,
    )
    for _ in range(3):
        r = s.complete("system", [TextBlock("camera"), ImageBlock(b"fixture")], tools())
        assert r.usage.dollars == pytest.approx((60 + 4 + 60) / 1e6)
        assert r.tool_calls[0].id == "call1"
        s.resolve({"call1": {"executed": True}})
    wire = requests[-1]
    assert wire["include"] == ["reasoning.encrypted_content"]
    assert wire["parallel_tool_calls"] is False
    assert wire["input"][1] == response()["output"][0]
    assert wire["input"][3]["type"] == "function_call_output"
    assert wire["input"][3]["call_id"] == "call1"
    assert wire["input"][0]["content"][1]["type"] == "input_text"
    assert s.history[0]["content"][1]["type"] == "input_image"
    s.reset()
    assert s.pending == [] and s.history == []


def test_openai_rejects_late_reply(tmp_path):
    def create(**kwargs):
        s.reset()
        return response()

    s = OpenAISession(
        "placeholder",
        TokenPrices(1, 2, 0.1, "test"),
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
        trace_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="late response"):
        s.complete("system", [TextBlock("camera")], tools())
    assert s.history == [] and s.pending == []


def test_strict_schema_keeps_optional_step_fields_nullable_and_required():
    from rung.backends.openai_session import strict_parameters

    spec = next(t for t in tools("both", action_sequence=True) if t.name == "execute_sequence")
    strict = strict_parameters(spec.parameters)
    assert strict["required"] == ["steps", "note"] and strict["additionalProperties"] is False
    step = strict["properties"]["steps"]["items"]
    assert set(step["required"]) == set(step["properties"]) and not step["additionalProperties"]
    assert step["properties"]["tool"]["type"] == "string"  # originally required
    assert step["properties"]["delta_xyz_m"]["anyOf"][1] == {"type": "null"}  # optional
    # The CLI route still gets the original schema.
    assert spec.parameters["properties"]["steps"]["items"]["required"] == ["tool"]
    requests = []

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        return response()

    s = OpenAISession(
        "placeholder",
        TokenPrices(1, 2, 0.1, "test"),
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    s.complete("system", [TextBlock("camera")], tools("both", action_sequence=True))
    wire = next(t for t in requests[0]["tools"] if t["name"] == "execute_sequence")
    assert wire["strict"] and wire["parameters"] == strict
