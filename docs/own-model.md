# Bring your own model

Rung talks to models through one seam, `rung.backends.ModelBackend`. It has three members:

```python
class ModelBackend(Protocol):
    @property
    def model(self) -> str: ...  # recorded in traces and provenance
    @property
    def vendor(self) -> str: ...
    def complete(self, request: ModelRequest) -> ModelResponse: ...
```

A `ModelRequest` carries a system prompt, messages made of text and image blocks, the tool
schemas of the interface level, and a token limit. A `ModelResponse` returns text, tool
calls and usage. The interface levels never branch on the model, so a comparison at a
fixed level varies only the model. Retries and vendor quirks belong inside your backend.

## OpenAI-compatible servers (vLLM, SGLang, Ollama, LM Studio)

[`examples/own_model.py`](../examples/own_model.py) is a complete backend for any
`/v1/chat/completions` endpoint with tool calling, and runs it on the mock tabletop:

```bash
python examples/own_model.py --selftest          # a local stub endpoint, no model needed
python examples/own_model.py --endpoint http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct
```

Copy `ChatCompletionsBackend` into your code and pass it to a policy:

```python
from rung.interfaces import L0Config, L0Policy

policy = L0Policy(ChatCompletionsBackend(url, model), L0Config())  # absolute TCP poses
```

`L2Policy` (semantic primitives) takes a backend the same way. Your model needs image input
and reliable tool calling. Run a few mock episodes before a simulator; a model that ignores
the tools, or calls them with malformed arguments, shows up there in the trace.

## Anything else

For a model behind another API, or in-process weights, write a class with the three members
above. `rung.backends.MockBackend` is the smallest working example, and the Anthropic,
Gemini and OpenAI backends in `src/rung/backends/` show the vendor versions.

## What to know

- **Cost.** Dollar figures come from `rung.backends.prices`, which covers hosted frontier
  models only. Set `ModelUsage.dollars` in your backend if you want spend caps to apply;
  otherwise your model is metered at $0.
- **The direct interface.** `--policy direct`, the control loop behind our RoboLab
  leaderboard run, is built on the Gemini interactions API, with sessions for Gemini and
  OpenAI in `src/rung/backends/*_session.py`. A new vendor there means a new session
  class, not a `ModelBackend`. L0 and L2 are the levels to start with.
- **Comparisons.** To compare your model with ours, keep the interface level, the profile
  and the episode seeds fixed, and change only the backend.
