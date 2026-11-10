"""Construct backends and grounding from model names.

Shared by the CLI and the RoboLab connector so both resolve a model flag identically.
"""

from __future__ import annotations

from rung.backends.base import ModelBackend

VENDORS = {
    "gemini": ("rung.backends.gemini", "GeminiBackend"),
    "anthropic": ("rung.backends.anthropic", "AnthropicBackend"),
    "openai": ("rung.backends.openai", "OpenAIBackend"),
}

PREFIX_TO_VENDOR = {"gemini": "gemini", "claude": "anthropic", "gpt": "openai"}

DEFAULT_GROUNDING_MODEL = "gemini-3.7-flash"
"""Grounding is held fixed while the acting model changes, so the controlled track can say
two policies saw the same catalog. Gemini is the default because it is the cheapest model
that returns boxes in a documented convention."""


def vendor_of(model: str) -> str:
    for prefix, vendor in PREFIX_TO_VENDOR.items():
        if model.startswith(prefix):
            return vendor
    raise ValueError(
        f"cannot tell which vendor serves {model!r}; pass a vendor explicitly "
        f"({', '.join(sorted(VENDORS))})"
    )


def build_backend(model: str, vendor: str | None = None, **kwargs) -> ModelBackend:
    vendor = vendor or vendor_of(model)
    if vendor not in VENDORS:
        raise ValueError(f"unknown vendor {vendor!r}; have {sorted(VENDORS)}")
    module_name, class_name = VENDORS[vendor]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)(model=model, **kwargs)


def build_grounding(model: str | None = None, vendor: str | None = None, **kwargs):
    """The perception source for a real rig, where there is no oracle."""
    from rung.perception.box_grounding import BoxGrounding

    return BoxGrounding(build_backend(model or DEFAULT_GROUNDING_MODEL, vendor), **kwargs)
