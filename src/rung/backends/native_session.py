"""Provider-neutral session contract and append-only accounting archive."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rung.backends.base import ImageBlock, ModelResponse, TextBlock, ToolCall, ToolSpec


def encode_body(body: dict) -> bytes:
    """Shared REST serializer; archives can reconstruct the exact unauthenticated body."""
    return json.dumps(body, allow_nan=False).encode()


def _pack_images(value, folder: Path, *, image_content: bool = False):
    if isinstance(value, list):
        return [_pack_images(v, folder) for v in value]
    if not isinstance(value, dict):
        return value
    result = {
        k: v if k in ("arguments", "parameters") else _pack_images(v, folder)
        for k, v in value.items()
    }
    if value.get("type") == "user_input" or value.get("role") == "user":
        result["content"] = [_pack_images(v, folder, image_content=True) for v in value["content"]]
    key, prefix = None, ""
    if image_content and value.get("type") == "image" and isinstance(value.get("data"), str):
        key = "data"
        mime = value.get("mime_type", "application/octet-stream")
        data = base64.b64decode(value[key], validate=True)
    elif (
        image_content
        and value.get("type") == "input_image"
        and isinstance(value.get("image_url"), str)
    ):
        key = "image_url"
        prefix, encoded = value[key].split(",", 1)
        prefix += ","
        mime = prefix.split(";", 1)[0].removeprefix("data:")
        data = base64.b64decode(encoded, validate=True)
    if key is not None:
        digest = hashlib.sha256(data).hexdigest()
        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(
            mime, ".bin"
        )
        folder.mkdir(exist_ok=True)
        path = folder / (digest + suffix)
        try:
            with path.open("xb") as f:
                f.write(data)
        except FileExistsError:
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("corrupt image blob; refusing archive") from None
        result[key] = {"$rung_blob": path.name, "sha256": digest, "prefix": prefix}
    return result


def read_request_archive(path: Path) -> dict:
    """Verify image blobs and reconstruct a logical request (including native steps)."""
    path = Path(path)
    value = json.loads(path.read_text())

    def restore(v):
        if isinstance(v, list):
            return [restore(x) for x in v]
        if isinstance(v, dict):
            if "$rung_blob" in v:
                name = v["$rung_blob"]
                if Path(name).name != name:
                    raise ValueError("invalid blob path")
                data = (path.parent / "blobs" / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != v["sha256"]:
                    raise ValueError("image blob hash mismatch")
                return v["prefix"] + base64.b64encode(data).decode()
            return {k: x if k in ("arguments", "parameters") else restore(x) for k, x in v.items()}
        return v

    result = restore(value)
    digest = result.get("canonical_body_sha256")
    if digest and hashlib.sha256(encode_body(result["request"])).hexdigest() != digest:
        raise ValueError("reconstructed request hash mismatch")
    return result


@dataclass(frozen=True)
class TokenPrices:
    input_per_million: float
    output_per_million: float
    cached_per_million: float
    source: str
    long_context_threshold: int | None = None
    long_context_rates: tuple[float, float, float] | None = None

    def __post_init__(self):
        rates = (self.input_per_million, self.output_per_million, self.cached_per_million)
        if not self.source or any(not math.isfinite(v) or v < 0 for v in rates):
            raise ValueError("finite nonnegative prices and a provenance source are required")
        if (self.long_context_threshold is None) != (self.long_context_rates is None):
            raise ValueError("context threshold and rates must be supplied together")
        if self.long_context_rates is not None:
            if (
                self.long_context_threshold <= 0
                or len(self.long_context_rates) != 3
                or any(not math.isfinite(v) or v < 0 for v in self.long_context_rates)
            ):
                raise ValueError("invalid long context prices")

    def rates_for(self, input_tokens):
        if self.long_context_threshold is not None and input_tokens > self.long_context_threshold:
            return self.long_context_rates
        return self.input_per_million, self.output_per_million, self.cached_per_million


class NativeSession(Protocol):
    model: str
    vendor: str
    turn: int
    pending: list[ToolCall]

    def reset(self) -> None: ...
    def resolve(self, receipts: dict[str, dict]) -> None: ...
    def complete(
        self, system: str, blocks: list[TextBlock | ImageBlock], tools: list[ToolSpec]
    ) -> ModelResponse: ...


class SessionLedger:
    trace_dir: Path | None

    def reset(self):
        self.history: list[dict] = []
        self.pending: list[ToolCall] = []
        self.session_id = uuid.uuid4().hex
        self.turn = 0
        self.last_status: str | None = None

    def _write(self, suffix: str, value: dict, *, identity: tuple[str, int] | None = None):
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        session_id, turn = identity or (self.session_id, self.turn)
        path = self.trace_dir / f"{session_id}-{turn:04d}-{suffix}.json"
        if suffix == "request":
            value = {
                **value,
                "canonical_body_sha256": hashlib.sha256(encode_body(value["request"])).hexdigest(),
                "archive_format": "rung-native-v1",
            }
            value = _pack_images(value, self.trace_dir / "blobs")
        with path.open("x") as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
