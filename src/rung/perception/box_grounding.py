"""Model grounding: a vision model names and boxes the objects, the harness measures them.

The grounding model is configured independently of the acting model, so perception can be
held fixed while the policy varies. The prompt requests every object on the table, not only
those in the instruction, so the policy also learns about obstacles.

Boxes follow Gemini's convention, ``[y0, x0, y1, x1]`` normalised to 0-1000; other vendors
are prompted for the same contract. Offline imports may select pixel units or XYXY order
explicitly; neither is inferred from image dimensions.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass

import numpy as np

from rung.backends.base import ImageBlock, Message, ModelBackend, ModelRequest, TextBlock
from rung.contracts.observation import Observation
from rung.perception.types import Detection
from rung.policy import ModelUsage

PROMPT = """List every distinct object on the table in this image.

Include containers (bins, bowls, crates, plates) and anything a robot arm might have to
work around, not only the objects the task mentions. Do not list the table, the robot, or
the background.

Reply with JSON only: a list of objects, each with
  "label": a short noun phrase, lower case, unique within the list
  "box_2d": [y0, x0, y1, x1], normalised to 0-1000, top-left origin

Example: [{"label": "banana", "box_2d": [412, 233, 498, 401]}]"""


@dataclass
class BoxGroundingConfig:
    max_objects: int = 12
    effort: str | None = "low"
    max_tokens: int = 4096
    """Thinking tokens count against this budget. At 1024, a scene with about ten objects
    can spend nearly all of it thinking and return a truncated list, which parses as no
    boxes."""
    reuse_for_steps: int = 0
    """Reuse the last grounding for this many further steps. Zero re-grounds every step,
    since objects move; higher values trade accuracy for cost. Grounding calls are counted
    in the trace either way."""


class BoxGrounding:
    """Boxes from a vision model, converted to detections in pixel coordinates."""

    def __init__(self, backend: ModelBackend, config: BoxGroundingConfig | None = None):
        self.backend = backend
        self.cfg = config or BoxGroundingConfig()
        self.calls = 0
        self.failures: list[str] = []
        self.last_usage = ModelUsage()
        self.total_usage = ModelUsage()
        """Grounding is priced like any other model call; a step that grounds and then acts
        makes two calls."""
        self._cache: list[Detection] | None = None
        self._cache_age = 0
        self.last_prompt: str = ""
        self.last_response_text: str = ""
        self.last_detections: list[Detection] = []
        self.last_reused = False
        """The last detect() prompt, raw response and parsed detections, recorded so the
        transcript shows the grounding call like any other model call."""

    @property
    def name(self) -> str:
        return f"boxes/{self.backend.model}"

    def detect(self, observation: Observation, instruction: str = "") -> list[Detection]:
        self.last_usage = ModelUsage()
        if self._cache is not None and self._cache_age < self.cfg.reuse_for_steps:
            self._cache_age += 1
            self.last_reused = True
            return self._cache
        self.last_reused = False

        cam = observation.scene
        request = ModelRequest(
            system="You locate objects in images for a robot. You answer with JSON only.",
            messages=[
                Message.user(
                    TextBlock(PROMPT),
                    _png_block(cam.rgb),
                    TextBlock(f"The robot's task, for context only: {instruction}")
                    if instruction
                    else TextBlock("Locate every object."),
                )
            ],
            effort=self.cfg.effort,
            max_tokens=self.cfg.max_tokens,
        )
        self.last_prompt = (
            PROMPT
            + "\n\n"
            + (
                f"The robot's task, for context only: {instruction}"
                if instruction
                else "Locate every object."
            )
        )
        response = self.backend.complete(request)
        self.last_response_text = response.text or ""
        self.calls += 1
        self.last_usage = response.usage
        self.total_usage.add(response.usage)
        detections = parse_boxes(
            response.text, cam.width, cam.height, self.name, self.cfg.max_objects
        )
        if not detections:
            self.failures.append(
                f"the grounding model returned no usable boxes: {response.text[:120]!r}"
            )
        self._cache, self._cache_age = detections, 0
        self.last_detections = detections
        return detections


def parse_boxes(
    text: str,
    width: int,
    height: int,
    source: str = "boxes",
    max_objects: int = 12,
    *,
    coordinates: str = "normalized_1000",
    order: str = "yxyx",
) -> list[Detection]:
    """Parse boxes with explicit units and axis order, independent of image resolution."""
    payload = _extract_json(text)
    if payload is None:
        return []
    out: list[Detection] = []
    seen: set[str] = set()
    for item in payload[:max_objects]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("name") or "").strip().lower()
        raw = item.get("box_2d") or item.get("box") or item.get("bbox")
        if not label or not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        try:
            values = [float(v) for v in raw]
        except (TypeError, ValueError):
            continue
        box = _to_pixels(values, width, height, coordinates=coordinates, order=order)
        if box is None:
            continue
        if label in seen:
            label = f"{label} {sum(1 for d in out if d.label.startswith(label)) + 1}"
        seen.add(label)
        out.append(Detection(label=label, box=box, score=1.0, source=source))
    return out


def _to_pixels(
    values: list[float],
    width: int,
    height: int,
    *,
    coordinates: str = "normalized_1000",
    order: str = "yxyx",
):
    if coordinates not in ("normalized_1000", "pixels") or order not in ("yxyx", "xyxy"):
        raise ValueError("declare coordinates as normalized_1000/pixels and order as yxyx/xyxy")
    if not all(np.isfinite(v) for v in values):
        return None
    if coordinates == "normalized_1000" and any(v < 0 or v > 1000 for v in values):
        return None
    if order == "yxyx":
        y0, x0, y1, x1 = values
    else:
        x0, y0, x1, y1 = values
    sx, sy = (width / 1000, height / 1000) if coordinates == "normalized_1000" else (1, 1)
    box = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
    if not _plausible(box, width, height):
        return None
    left, top, right, bottom = box
    return (
        max(0.0, left),
        max(0.0, top),
        min(float(width - 1), right),
        min(float(height - 1), bottom),
    )


def _plausible(box, width: int, height: int) -> bool:
    left, top, right, bottom = box
    if not all(np.isfinite(v) for v in box):
        return False
    if right <= left or bottom <= top:
        return False
    if right < 0 or bottom < 0 or left > width or top > height:
        return False
    area = (right - left) * (bottom - top)
    return area >= 16 and area <= width * height * 0.95


def _extract_json(text: str):
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    body = fenced.group(1) if fenced else text
    start, end = body.find("["), body.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(body[start : end + 1])
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, list) else None


def _png_block(rgb) -> ImageBlock:
    """Encode the scene image for the grounding model as JPEG at quality 92, despite the
    name; see interfaces.l2._image_block."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB").save(
        buf, format="JPEG", quality=92
    )
    return ImageBlock(buf.getvalue(), "image/jpeg", "scene")
