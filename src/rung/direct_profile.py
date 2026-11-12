"""Direct-interface profiles and the scripted fake Gemini endpoint.

A profile is a JSON file ``{"model", "effort", "max_tokens", "timeout_s",
"direct_config": {...}, "droid": {...}}``. ``direct_config`` keys must be ``DirectConfig``
fields exactly, so the policy that runs is the one the profile names and nothing is
silently dropped.

``FakeInteractions`` stands in for the Gemini Interactions endpoint with a fixed tool-call
script, so the direct loop runs end to end with no API key.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

from rung.backends.gemini_session import GeminiSession
from rung.backends.native_session import TokenPrices
from rung.backends.prices import PRICES
from rung.interfaces.direct import DirectConfig, DirectPolicy

DEFAULT_MODEL = "gemini-4-argon"
FAKE_MODEL = "fake"

FAKE_SCRIPT = (
    ("observe", {"note": "look at the scene"}),
    (
        "move_relative",
        {
            "delta_xyz_m": [0.0, 0.0, -0.03],
            "rotation_vector_rad": [0.0, 0.0, 0.0],
            "note": "descend",
        },
    ),
    ("set_gripper", {"action": "close", "note": "grasp"}),
    (
        "move_relative",
        {"delta_xyz_m": [0.0, 0.0, 0.03], "rotation_vector_rad": [0.0, 0.0, 0.0], "note": "lift"},
    ),
    ("done", {"note": "scripted fake model finished"}),
)

_ZERO_ROT = [0.0, 0.0, 0.0]

FAKE_SCRIPT_BIMANUAL = (
    ("observe", {"note": "look at the scene"}),
    (
        "move_relative",
        {
            "arm": "left",
            "delta_xyz_m": [0.0, 0.0, -0.03],
            "rotation_vector_rad": _ZERO_ROT,
            "note": "left arm descends",
        },
    ),
    ("set_gripper", {"arm": "left", "action": "close", "note": "left grasp"}),
    (
        "move_both",
        {
            "left": {"delta_xyz_m": [0.0, 0.0, 0.03], "rotation_vector_rad": _ZERO_ROT},
            "right": {"delta_xyz_m": [0.0, 0.0, 0.03], "rotation_vector_rad": _ZERO_ROT},
            "note": "both arms lift together",
        },
    ),
    ("done", {"note": "scripted fake model finished"}),
)
"""The fake's script for the bimanual profile: per-arm tools and one ``move_both``."""


@dataclass
class DirectProfile:
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    max_tokens: int = 8192
    timeout_s: float = 600.0
    direct_config: DirectConfig = field(default_factory=DirectConfig)
    droid: dict = field(default_factory=dict)
    """Camera layout and history for RoboLab DROID episodes."""


def load_profile(path: str | Path) -> DirectProfile:
    data = json.loads(Path(path).read_text())
    known = {f.name for f in fields(DirectConfig)}
    unknown = set(data.get("direct_config", {})) - known
    if unknown:
        raise ValueError(f"unknown DirectConfig fields {sorted(unknown)}")
    kwargs = {k: data[k] for k in ("model", "effort", "max_tokens", "timeout_s") if k in data}
    return DirectProfile(
        **kwargs,
        direct_config=DirectConfig(**data.get("direct_config", {})),
        droid=dict(data.get("droid", {})),
    )


def token_prices(model: str) -> TokenPrices:
    price = PRICES[model]
    cached = price.cached_input_per_mtok
    return TokenPrices(
        price.input_per_mtok,
        price.output_per_mtok,
        price.input_per_mtok if cached is None else cached,
        f"rung.backends.prices:{model}",
    )


class FakeInteractions:
    """A fixed tool-call script in place of the Gemini Interactions endpoint.

    Everything else on the path is real: GeminiSession's transaction and accounting,
    DirectPolicy's tools, receipts and bounds, and the episode loop. It tests plumbing,
    not manipulation; the script ignores the images.
    """

    def __init__(self, script=FAKE_SCRIPT, latency_s: float = 0.0):
        self.script, self.latency_s = list(script), latency_s
        self.calls = 0
        self._ids = itertools.count(1)

    def create(self, **body):
        if self.latency_s:
            time.sleep(self.latency_s)
        name, args = (
            self.script[self.calls]
            if self.calls < len(self.script)
            else ("give_up", {"note": "script exhausted"})
        )
        self.calls += 1
        return {
            "status": "completed",
            "steps": [
                {
                    "type": "function_call",
                    "id": f"fake{next(self._ids)}",
                    "name": name,
                    "arguments": args,
                }
            ],
            "usage": {
                "total_input_tokens": 1000,
                "total_output_tokens": 50,
                "total_thought_tokens": 0,
                "total_cached_tokens": 0,
            },
        }


class FakeClient:
    def __init__(self, interactions):
        self.interactions = interactions


def build_direct_policy(
    profile: DirectProfile,
    *,
    api_key: str | None = None,
    trace_dir: Path | None = None,
    fake_latency_s: float = 0.0,
    budgets: dict | None = None,
) -> DirectPolicy:
    """A fresh DirectPolicy for one episode over the paid Gemini API, or the fake.

    ``profile.model == "fake"`` uses the scripted endpoint and needs no key; a bimanual
    profile gets the two-arm script. Any other
    model goes to the Gemini API with ``api_key`` or ``GEMINI_API_KEY`` from the
    environment, and must have a price on record in ``rung.backends.prices``.
    """
    if profile.model == FAKE_MODEL:
        session = GeminiSession(
            "fake-" + DEFAULT_MODEL,
            token_prices(DEFAULT_MODEL),
            client=FakeClient(
                FakeInteractions(
                    FAKE_SCRIPT_BIMANUAL if profile.direct_config.bimanual else FAKE_SCRIPT,
                    latency_s=fake_latency_s,
                )
            ),
            effort=profile.effort,
            max_tokens=profile.max_tokens,
            trace_dir=trace_dir,
        )
    else:
        if profile.model not in PRICES:
            raise ValueError(f"no price on record for {profile.model}; refusing to meter blind")
        session = GeminiSession(
            profile.model,
            token_prices(profile.model),
            api_key=api_key,
            effort=profile.effort,
            max_tokens=profile.max_tokens,
            timeout_s=profile.timeout_s,
            trace_dir=trace_dir,
            transport="rest",
        )
    return DirectPolicy(session, profile.direct_config, budgets=budgets)
