"""What perception produces: detections, per-object geometry, and the catalog.

A ``Detection`` is a grounding claim in image space (this pixel region is "the banana").
An ``ObjectDescriptor`` is the metric geometry derived from it once depth is applied.
The catalog is the compact text a policy is shown: metric size, jaw fit and reach are
stated explicitly rather than left to be inferred from pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class Detection:
    """A grounding claim in the scene image.

    ``box`` is ``(u0, v0, u1, v1)`` in pixels of the delivered image. ``mask`` is optional
    and takes precedence when present. ``source`` names the producer and is recorded in the
    trace, so runs can verify that policies saw the same catalog.
    """

    label: str
    box: tuple[float, float, float, float]
    score: float = 1.0
    mask: npt.NDArray[np.bool_] | None = None
    source: str = "unknown"


@dataclass
class ObjectDescriptor:
    """One object's geometry, in the robot root frame, metres.

    ``length`` and ``width`` are the footprint sides, longer first. ``axis`` is the
    horizontal principal axis, the direction the jaw must close across. ``visible_fraction``
    is the share of the detection's pixels with usable depth; a low value means occlusion
    or missing depth, and the remaining fields are provisional.
    """

    object_id: str
    label: str
    centroid: npt.NDArray[np.float64]
    top_z: float
    bottom_z: float
    length: float
    width: float
    height: float
    axis: npt.NDArray[np.float64]
    grasp_length: float
    grasp_width: float
    grasp_centre: npt.NDArray[np.float64]
    n_points: int
    visible_fraction: float
    uncertain: bool = False
    uncertainty_reason: str | None = None
    points: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    detection: Detection | None = field(default=None, repr=False)

    @property
    def min_footprint(self) -> float:
        """The narrowest horizontal span at grasp height, which is what the jaw must span.

        Measured at grasp height, not over the whole cloud; see cloud.cross_section.
        """
        return float(min(self.grasp_length, self.grasp_width))

    def describe(self, jaw_m: float) -> str:
        """One prompt line stating what the image does not show: metric size and position."""
        bits = [
            f"{self.object_id} ({self.label})",
            f"at x={self.grasp_centre[0]:.2f} y={self.grasp_centre[1]:.2f} m",
            f"{self.length * 100:.0f} x {self.width * 100:.0f} x {self.height * 100:.0f} cm",
            f"top at {self.top_z * 100:.0f} cm",
            f"{self.grasp_width * 100:.0f} cm across at grasp height",
        ]
        if self.min_footprint > jaw_m:
            bits.append(
                f"narrowest side {self.min_footprint * 100:.0f} cm, wider than the jaw "
                "across its body (propose_grasps may still find a way to hold it)"
            )
        if self.uncertain:
            bits.append(f"uncertain: {self.uncertainty_reason}")
        return ", ".join(bits)


@dataclass
class Catalog:
    """Everything perception knows this step, plus its own provenance."""

    objects: list[ObjectDescriptor]
    source: str
    version: str
    t: float
    failures: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, object_id: str) -> ObjectDescriptor | None:
        for o in self.objects:
            if o.object_id == object_id:
                return o
        return None

    def ids(self) -> list[str]:
        return [o.object_id for o in self.objects]

    def render(self, jaw_m: float) -> str:
        if not self.objects:
            return "No objects were found in this view."
        return "\n".join(f"- {o.describe(jaw_m)}" for o in self.objects)
