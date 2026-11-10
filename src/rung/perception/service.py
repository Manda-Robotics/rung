"""The perception service: detections in, versioned catalog out.

For policies to share perception, the catalog must be reproducible from the observation
and grounding alone; every trace records the version that produced it.

Grounding sources are pluggable:

* ``OracleGrounding`` reads the mock's ground-truth index map. Diagnostics only: it tests
  the geometry without a model and separates grounding errors from geometry errors.
* ``BoxGrounding`` requests boxes from a vision model. Used by real runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from rung.contracts.observation import CameraFrame, Observation
from rung.perception.cloud import (
    LINK_M,
    TABLE_CUT_M,
    above_table,
    connected_component,
    cross_section,
    extents,
    footprint_centre,
    grasp_point,
    local_span,
    mask_from_box,
    points_from_mask,
    principal_axis_xy,
    top_slab,
    trusted_width_direction,
    unseen_fraction,
    view_direction_xy,
    view_elevation,
)
from rung.perception.types import Catalog, Detection, ObjectDescriptor

MIN_POINTS = 30
LOCAL_NARROWING_M = 0.005
"""The local width replaces the whole-object width only when narrower by more than this,
which indicates a real taper or curve rather than sampling noise."""
LOW_VISIBILITY = 0.5
HAND_NEAR_M = 0.12
"""Within this horizontal distance the hand exclusion zone removes part of the object."""


@runtime_checkable
class Grounding(Protocol):
    """Turns an image into labelled regions. Knows nothing about depth or robots."""

    @property
    def name(self) -> str: ...

    def detect(self, observation: Observation, instruction: str) -> list[Detection]: ...


@dataclass
class PerceptionConfig:
    link_m: float = LINK_M
    min_points: int = MIN_POINTS
    exclude_hand: bool = True
    max_objects: int = 12
    table_z: float | None = 0.0
    """Height of the table plane in the root frame. Points at or just above it are
    removed before clustering; see cloud.TABLE_CUT_M. None disables the cut."""
    jaw_m: float = 0.085
    """What the jaw opens to, so the width measurement can prefer the silhouette
    whenever it fits; see cloud.trusted_width_direction."""


class PerceptionService:
    """Detections plus depth to a catalog of metric objects.

    The same observation and the same grounding must give the same catalog, so nothing
    here is random and nothing depends on call order.
    """

    version = "perception-0.1"

    def __init__(self, grounding: Grounding, config: PerceptionConfig | None = None):
        self.grounding = grounding
        self.cfg = config or PerceptionConfig()

    @property
    def source(self) -> str:
        return self.grounding.name

    def build(
        self,
        observation: Observation,
        instruction: str = "",
        table_z: float | None = None,
        jaw_m: float | None = None,
        detections: list | None = None,
    ) -> Catalog:
        """Measure every detection in depth.

        ``detections`` re-measures earlier boxes without querying the grounding model
        (see Executor.refresh)."""
        cam = observation.scene
        self._table_z = self.cfg.table_z if table_z is None else table_z
        self._jaw_m = self.cfg.jaw_m if jaw_m is None else jaw_m
        failures: list[str] = []
        if not cam.has_geometry:
            return Catalog(
                [],
                self.source,
                self.version,
                observation.t,
                failures=["scene camera has no depth or calibration"],
            )

        reused = detections is not None
        if detections is None:
            detections = self.grounding.detect(observation, instruction)
        objects: list[ObjectDescriptor] = []
        for i, det in enumerate(detections[: self.cfg.max_objects]):
            desc = self._describe(cam, det, observation, index=i)
            if desc is None:
                if not reused:  # a reused box may cover a spot now vacated
                    failures.append(f"{det.label}: no usable depth inside the detection")
                continue
            objects.append(desc)
        catalog = Catalog(objects, self.source, self.version, observation.t, failures=failures)
        if reused:
            catalog.extra["reused_boxes"] = True
        usage = None if reused else getattr(self.grounding, "last_usage", None)
        if usage is not None and usage.calls:
            catalog.extra["grounding_usage"] = usage
        failures.extend(getattr(self.grounding, "failures", [])[-1:] if not objects else [])
        return catalog

    def _describe(
        self, cam: CameraFrame, det: Detection, obs: Observation, index: int
    ) -> ObjectDescriptor | None:
        mask = det.mask if det.mask is not None else mask_from_box(det.box, cam.height, cam.width)
        exclude = obs.tcp.pos if self.cfg.exclude_hand else None
        pts, visible = points_from_mask(cam, mask, exclude_near=exclude)
        table_z = getattr(self, "_table_z", self.cfg.table_z)
        if table_z is not None:
            pts = above_table(pts, table_z)
        if len(pts) < self.cfg.min_points:
            return None

        seed = _seed_point(pts, _central(pts, det.box, cam))
        cluster = connected_component(pts, seed, link_m=self.cfg.link_m)
        if len(cluster) < self.cfg.min_points:
            cluster = pts
        axis = principal_axis_xy(cluster)
        top, bottom, length, width, height = extents(cluster, axis)
        if table_z is not None and bottom < table_z + TABLE_CUT_M + 0.006:
            # The table cut also removes the object's lowest centimetre. The object rests
            # on the table, so its bottom is the table plane. Leaving it at the cut
            # underestimates height (a 4 cm banana measures 2.9 cm) and makes derived
            # grasp heights too shallow, so the object can pivot out of the hand.
            bottom = float(table_z)
            height = float(top - bottom)

        # What the jaw has to span is the object's width where the jaw closes, not the
        # span of every point the camera saw. Measured in a slab below the top face.
        slab = cross_section(cluster, top)
        crown = top_slab(cluster, top)
        centre_xy = footprint_centre(crown if len(crown) >= 8 else slab)
        cam_pos = cam.pose.pos if cam.pose is not None else None
        view = view_direction_xy(cam_pos, centre_xy)
        elevation = view_elevation(cam_pos, np.array([centre_xy[0], centre_xy[1], top]))
        grasp_width, grasp_length, closing, bounded = trusted_width_direction(
            slab,
            crown,
            view,
            fits_m=getattr(self, "_jaw_m", self.cfg.jaw_m) - 0.016,
            elevation_rad=elevation,
        )
        # Prefer the width under the pads: a curved banana spans 7.7 cm over its length
        # and 4.5 cm under the pads.
        aim = grasp_point(crown) if len(crown) >= 8 else centre_xy
        local = local_span(slab, aim, closing)
        if local is not None:
            if bounded:
                local *= 1.0 + unseen_fraction(elevation)
            else:
                top_local = local_span(crown, aim, closing)
                if top_local is not None:
                    local = max(local, top_local)
            if 0 < local < grasp_width - LOCAL_NARROWING_M:
                grasp_width = float(local)
        axis = np.array([-closing[1], closing[0]])  # the jaw closes across the long axis
        grasp_centre = np.array([centre_xy[0], centre_xy[1], top])

        uncertain = False
        reason = None
        hand_dist = float(np.linalg.norm(obs.tcp.pos[:2] - grasp_centre[:2]))
        if visible < LOW_VISIBILITY:
            uncertain, reason = True, f"only {visible * 100:.0f} percent of it returned depth"
        elif bounded and grasp_width > 0.06:
            uncertain, reason = (
                True,
                (
                    f"its width along the camera's line of sight is an estimate from a partial "
                    f"view, at most {grasp_width * 100:.1f} cm"
                ),
            )
        elif len(cluster) < len(pts) * 0.35:
            uncertain, reason = True, "the detection spans more than one surface"
        elif hand_dist < HAND_NEAR_M:
            # Points around the grasp point were removed as hand, so the remaining cloud
            # understates the width (a 12 cm object can measure 3 cm). It is re-measured
            # once the hand is clear.
            uncertain, reason = (
                True,
                (
                    "the gripper is over it, so its size is measured from a partial view; "
                    "move the hand away to measure it properly"
                ),
            )

        return ObjectDescriptor(
            object_id=f"{_slug(det.label)}_{index}",
            label=det.label,
            centroid=cluster.mean(axis=0),
            top_z=top,
            bottom_z=bottom,
            length=length,
            width=width,
            height=height,
            grasp_length=grasp_length,
            grasp_width=grasp_width,
            grasp_centre=grasp_centre,
            axis=axis if axis is not None else np.array([1.0, 0.0]),
            n_points=int(len(cluster)),
            visible_fraction=visible,
            uncertain=uncertain,
            uncertainty_reason=reason,
            points=cluster,
            detection=det,
        )


def _seed_point(points: np.ndarray, central: np.ndarray | None = None) -> np.ndarray:
    """Flood-fill seed: the highest point of the largest component in the box's centre.

    A detection box is centred on the object it names, so its middle is mostly that object
    and its edges are neighbours. Simpler rules fail in clutter: the horizontal median of
    all box points can fall between two objects once the table is removed, and the highest
    central point can belong to a taller neighbour whose rim crosses the box.
    """
    if central is not None and central.sum() >= 5:
        pool = points[central]
        largest = _largest_component(pool)
        return largest[int(np.argmax(largest[:, 2]))]
    centre = np.median(points[:, :2], axis=0)
    near = points[np.linalg.norm(points[:, :2] - centre, axis=1) < 0.04]
    pool = near if len(near) >= 5 else points
    return pool[int(np.argmax(pool[:, 2]))]


def _largest_component(points: np.ndarray, link_m: float = LINK_M) -> np.ndarray:
    from rung.perception.cloud import component_labels

    if len(points) == 0:
        return points
    labels = component_labels(points, link_m)
    if labels is not None:
        return points[labels == np.bincount(labels).argmax()]
    remaining = points
    best = points[:1]
    while len(remaining):
        seed = remaining[int(np.argmax(remaining[:, 2]))]
        part = connected_component(remaining, seed, link_m=link_m)
        if len(part) == 0:
            part = seed[None]
        if len(part) > len(best):
            best = part
        taken = {tuple(np.round(q, 6)) for q in part}
        keep = np.array([tuple(np.round(q, 6)) not in taken for q in remaining], dtype=bool)
        remaining = remaining[keep]
        if len(best) > len(remaining):
            break
    return best


def _central(points: np.ndarray, box, cam: CameraFrame, fraction: float = 0.5) -> np.ndarray:
    """Mask of the points that project into the middle ``fraction`` of the box."""
    from rung.contracts.frames import project

    if len(points) == 0 or cam.K is None or cam.pose is None:
        return np.zeros(len(points), dtype=bool)
    u0, v0, u1, v1 = box
    du, dv = (u1 - u0) * (1 - fraction) / 2, (v1 - v0) * (1 - fraction) / 2
    uv, z = project(points, cam.K, cam.pose)
    return (
        (z > 0)
        & (uv[:, 0] >= u0 + du)
        & (uv[:, 0] <= u1 - du)
        & (uv[:, 1] >= v0 + dv)
        & (uv[:, 1] <= v1 - dv)
    )


def _slug(label: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in label.strip()]
    return "".join(keep).strip("_")[:24] or "object"


class OracleGrounding:
    """Ground-truth regions from the mock's index map. Diagnostics only.

    Never available to a policy under test. Running the same episode with oracle and with
    model grounding isolates the grounding's contribution to a failure.
    """

    name = "oracle"

    def __init__(self, embodiment, include_containers: bool = True):
        self.emb = embodiment
        self.include_containers = include_containers

    def detect(self, observation: Observation, instruction: str = "") -> list[Detection]:
        from rung.mock.world import gripper_bodies

        hand = gripper_bodies(self.emb.robot.tcp, self.emb.robot.travel, self.emb.gripper_model)
        _, _, idx = self.emb.scene.render(self.emb.scene_cam, extra=hand)
        out: list[Detection] = []
        for i, body in enumerate(self.emb.scene.bodies):
            if body.container and not self.include_containers:
                continue
            m = idx == i
            if m.sum() < 20:
                continue
            vs, us = np.nonzero(m)
            out.append(
                Detection(
                    label=body.name,
                    box=(float(us.min()), float(vs.min()), float(us.max()), float(vs.max())),
                    score=1.0,
                    mask=m,
                    source=self.name,
                )
            )
        return out
