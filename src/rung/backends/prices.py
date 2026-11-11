"""Per-model prices, versioned, so a run's dollar figure can be reproduced later.

A trace that records dollars must also record the table that produced them; ``VERSION``
identifies it.

Existing entries checked 2026-09-09; Gemini4 Argon added 2026-10-16 from Google:
* Anthropic, platform.claude.com/docs/en/about-claude/pricing
* Google, ai.google.dev/gemini-api/docs/pricing
* OpenAI, developers.openai.com/api/docs/models/gpt-6-astra
"""

from __future__ import annotations

from dataclasses import dataclass

VERSION = "prices-2026-10-16"


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None
    note: str = ""


PRICES: dict[str, Price] = {
    # Anthropic. Thinking tokens are billed as output.
    "claude-fable-5-1": Price(10.0, 50.0, 0.25),
    "claude-opus-5": Price(5.0, 25.0, 0.50),
    "claude-sonnet-5": Price(2.0, 10.0, 0.20),
    "claude-haiku-4-5": Price(1.0, 5.0, 0.10),
    # Google. Gemini 3.7 Flash is at an introductory rate that doubles on 2027-01-01.
    "gemini-4-argon": Price(2.0, 10.0, 0.10, "introductory rate as of 2026-10-16"),
    "gemini-3.7-flash": Price(0.75, 3.75, 0.075, "introductory rate through 2026-12-31"),
    "gemini-robotics-er-2-preview": Price(1.0, 5.0, None, "preview rate through 2026"),
    "gemini-3.1-pro-preview": Price(2.0, 12.0, 0.20, "prompts up to 200k tokens"),
    # OpenAI. Above 272k input tokens the whole request reprices; not modelled here.
    "gpt-6-astra": Price(10.0, 50.0, 1.0),
    "gpt-5.6-sol": Price(1.25, 10.0, 0.125),
}


def price_of(model: str) -> Price | None:
    if model in PRICES:
        return PRICES[model]
    for known, price in PRICES.items():  # tolerate a dated or suffixed id
        if model.startswith(known):
            return price
    return None


def dollars(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """Cost of one call. Returns 0.0 for an unpriced model rather than guessing.

    Thinking tokens are counted as output by every vendor here, so callers must add them to
    ``output_tokens`` before calling.
    """
    price = price_of(model)
    if price is None:
        return 0.0
    billed_input = max(0, input_tokens - cached_tokens)
    total = billed_input * price.input_per_mtok + output_tokens * price.output_per_mtok
    if cached_tokens and price.cached_input_per_mtok is not None:
        total += cached_tokens * price.cached_input_per_mtok
    elif cached_tokens:
        total += cached_tokens * price.input_per_mtok
    return total / 1_000_000
