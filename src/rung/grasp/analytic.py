"""The analytic grasp planner: candidates from a point cloud, with every check applied.

The object is scanned at several heights rather than only at the top face, so a tall
bottle can be taken at its neck and a wide box at a narrow part. Whether a grasp exists
depends on the cross-section at each height, not on a single bounding box.

Side grasps are horizontal approaches below the top face, offered alongside the top-down
candidates and ranked slightly below them for tall or top-heavy objects. They pass the
same checks and are executed through the same full-orientation target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rung.contracts.capabilities import EmbodimentCapabilities
from rung.contracts.frames import Pose, quat_from_axis_angle, quat_mul, rot_to_quat
from rung.grasp.checks import (
    check_approach_path,
    check_finger_collision,
    check_fingers_above_table,
    check_opening,
    check_reach,
    check_retreat,
    check_table,
)
from rung.grasp.types import GraspCandidate, GraspPlan, closing_yaw
from rung.perception.cloud import (
    cross_section,
    footprint_centre,
    grasp_point,
    local_span,
    top_slab,
    trusted_width_direction,
    unseen_fraction,
)
from rung.perception.types import ObjectDescriptor

STANDOFF_M = 0.10
LIFT_M = 0.15
SLAB_M = 0.025
JAW_MARGIN_M = 0.005
RIM_SLAB_M = 0.012
RIM_DEPTH_M = 0.02
"""A rim pinch closes this far below the rim: deep enough that the pads hold the wall,
shallow enough that the inner finger stays clear of a mug's floor."""
RIM_LIP_M = 0.006
"""The lip: how far below an object's top the rim circle is fitted."""
GRASP_FLOOR_M = 0.015
"""The lowest a top-down grasp puts the open fingertips above the table. The Robotiq
fingers swing down as they close: at 1 cm open-hand height they scrape the table, at
1.7 cm they do not."""


@dataclass
class PlannerConfig:
    heights: int = 5
    """How many cross-sections to try, from the top face downward."""
    standoff_m: float = STANDOFF_M
    lift_m: float = LIFT_M
    slab_m: float = SLAB_M
    jaw_margin_m: float = JAW_MARGIN_M
    min_grasp_depth_m: float = 0.008
    """The fingers must sit at least this far below the object's top, or they slip off."""
    allow_side: bool = True


def top_down_pose(pos: npt.ArrayLike, yaw: float = 0.0) -> Pose:
    """TCP pointing down (+z local along -z world) with the closing axis at ``yaw``."""
    down = quat_from_axis_angle([1.0, 0.0, 0.0], np.pi)
    about_z = quat_from_axis_angle([0.0, 0.0, 1.0], yaw)
    return Pose(np.asarray(pos, dtype=np.float64).reshape(3), quat_mul(about_z, down))


def _within_angle(angles: npt.NDArray, centre: float, half_deg: float) -> npt.NDArray:
    return np.abs((angles - centre + np.pi) % (2 * np.pi) - np.pi) < np.radians(half_deg)


def fit_rim_circle(points: npt.ArrayLike, iterations: int = 4):
    """``(centre_xy, radius, outlier_points)`` of a ring of points, or None.

    An algebraic least-squares circle, refitted with the points far from it removed.
    What is removed is what sticks out of a round rim: a handle, a spout, a neighbour's
    edge caught in the same box.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 8:
        return None
    keep = np.ones(len(pts), dtype=bool)
    cx = cy = r = 0.0
    for _ in range(iterations):
        xy = pts[keep, :2]
        if len(xy) < 8:
            return None
        A = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
        b = (xy**2).sum(axis=1)
        (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
        r2 = c + cx**2 + cy**2
        if r2 <= 0:
            return None
        r = float(np.sqrt(r2))
        resid = np.abs(np.linalg.norm(pts[:, :2] - [cx, cy], axis=1) - r)
        spread = float(np.median(resid[keep])) if keep.any() else 0.0
        keep = resid <= max(0.006, 3.0 * spread)
    resid = np.abs(np.linalg.norm(pts[:, :2] - [cx, cy], axis=1) - r)
    outliers = pts[resid > 0.012]
    return np.array([cx, cy]), r, outliers


def side_pose(pos: npt.ArrayLike, approach: npt.ArrayLike, closing: npt.ArrayLike) -> Pose:
    """A horizontal TCP: +z local along ``approach``, +y local along ``closing``.

    The tool frame's convention is that the fingers extend back along -z and close along
    y, so a horizontal grasp is a rotation that lays +z flat. The remaining axis, the pad
    width direction, then points up or down; the sign is chosen so the frame stays
    right-handed.
    """
    z = np.asarray(approach, dtype=np.float64).reshape(3)
    z = z / np.linalg.norm(z)
    y = np.asarray(closing, dtype=np.float64).reshape(3)
    y = y - z * float(y @ z)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    rot = np.stack([x, y, z], axis=1)
    return Pose(np.asarray(pos, dtype=np.float64).reshape(3), rot_to_quat(rot))


def _compass(direction: npt.ArrayLike) -> str:
    d = np.asarray(direction, dtype=np.float64)
    # ``direction`` is the way the hand moves in; it arrives from the opposite side.
    names = [
        ("the robot's side", (1.0, 0.0)),
        ("the far side", (-1.0, 0.0)),
        ("the left", (0.0, -1.0)),
        ("the right", (0.0, 1.0)),
    ]
    return max(names, key=lambda n: float(d[0] * n[1][0] + d[1] * n[1][1]))[0]


class AnalyticGraspPlanner:
    """Cross-section scanning plus the geometric checks. No learned model, no GPU."""

    planner = "analytic"
    version = "analytic-0.1"

    def __init__(self, capabilities: EmbodimentCapabilities, config: PlannerConfig | None = None):
        self.caps = capabilities
        self.cfg = config or PlannerConfig()

    def plan(
        self,
        obj: ObjectDescriptor,
        obstacles: npt.ArrayLike | None = None,
        view_dir: npt.ArrayLike | None = None,
        view_elevation: float | None = None,
    ) -> GraspPlan:
        """``view_dir`` is the horizontal direction from the scene camera to the object
        and ``view_elevation`` how steeply it looks down. With them, candidates close
        across the camera's line of sight where the object allows, and a width measured
        along it is bounded rather than trusted."""
        self._view_elevation = view_elevation
        obstacles = np.zeros((0, 3)) if obstacles is None else np.asarray(obstacles, float)
        candidates: list[GraspCandidate] = []
        if obj.points is None or len(obj.points) < 10:
            return GraspPlan(obj.object_id, [], self.planner, self.version)

        for i, height in enumerate(self._heights(obj)):
            cand = self._top_candidate(obj, height, obstacles, index=i, view_dir=view_dir)
            if cand is not None:
                candidates.append(cand)

        if self.cfg.allow_side:
            side = self._side_candidate(obj, obstacles, index=len(candidates), view_dir=view_dir)
            if side is not None:
                if any(c.feasible for c in candidates):
                    # Ranked below a feasible top-down grasp: a vertical lift is the simpler
                    # motion and the more thoroughly validated one.
                    side.score *= 0.9
                candidates.append(side)

        if not any(c.feasible for c in candidates):
            rim = self._rim_candidate(obj, obstacles, index=len(candidates))
            if rim is not None:
                candidates.append(rim)

        return GraspPlan(obj.object_id, candidates, self.planner, self.version)

    def _rim_candidate(
        self, obj: ObjectDescriptor, obstacles: npt.NDArray, index: int
    ) -> GraspCandidate | None:
        """Pinch the wall of an open container: one finger inside, one outside.

        A mug is wider than the jaw across its body but has a wall a few millimetres thick
        at its rim. Offered only when no whole-body grasp fits and the object is open at
        the top: its top slab is a ring around a hollow whose visible floor is well below
        the rim.
        """
        pts = obj.points
        crown = top_slab(pts, obj.top_z, RIM_SLAB_M)
        if len(crown) < 20 or obj.height < 0.04:
            return None
        rough = fit_rim_circle(crown)
        if rough is None:
            return None
        centre, radius, _ = rough
        if radius < 0.02 or radius > 0.15:
            return None
        radial = np.linalg.norm(crown[:, :2] - centre, axis=1)
        ringlike = float(np.mean(np.abs(radial - radius) < 0.35 * radius)) > 0.7
        inner = pts[np.linalg.norm(pts[:, :2] - centre, axis=1) < 0.6 * radius]
        hollow = len(inner) == 0 or float(np.median(inner[:, 2])) < obj.top_z - 0.025
        if not (ringlike and hollow):
            return None
        # The rim itself: a circle fitted to the lip, the top few millimetres of the object,
        # with outliers trimmed. A handle protrudes from that circle; a plain percentile of
        # radii would include it and can place the pinch outside the wall, closing on air.
        lip = crown[crown[:, 2] >= obj.top_z - RIM_LIP_M]
        fit = fit_rim_circle(lip if len(lip) >= 12 else crown)
        if fit is None:
            return None
        centre, radius, _ = fit
        if radius < 0.02 or radius > 0.15:
            return None
        # Protrusions are looked for over the whole height the fingers will occupy, not only
        # on the lip: a mug's handle usually starts below the rim, and the outer finger
        # comes down right beside it.
        height = max(obj.top_z - RIM_DEPTH_M, self.caps.workspace.table_z + GRASP_FLOOR_M)
        band = pts[(pts[:, 2] >= height - 0.01) & (pts[:, 2] <= obj.top_z + 1e-6)]
        outliers = band[np.linalg.norm(band[:, :2] - centre, axis=1) > radius + 0.012]
        # The wall to pinch: the side nearest the robot's base that has a clean lip, and no
        # handle or other protrusion within 35 degrees of it. The nearest side is the
        # shortest reach; a handle there is the one place the wall cannot be pinched.
        toward_base = -centre / max(1e-9, float(np.linalg.norm(centre)))
        base_angle = float(np.arctan2(toward_base[1], toward_base[0]))
        lip_angles = (
            np.arctan2(lip[:, 1] - centre[1], lip[:, 0] - centre[0]) if len(lip) else np.zeros(0)
        )
        bad_angles = (
            np.arctan2(outliers[:, 1] - centre[1], outliers[:, 0] - centre[0])
            if len(outliers)
            else np.zeros(0)
        )
        chosen = None
        for offset in sorted(np.radians(np.arange(-180, 180, 15)), key=abs):
            a = base_angle + offset
            if np.count_nonzero(_within_angle(bad_angles, a, 35)):
                continue
            if np.count_nonzero(_within_angle(lip_angles, a, 35)) < 4:
                continue
            chosen = a
            break
        if chosen is None:
            return None
        direction = np.array([np.cos(chosen), np.sin(chosen)])
        rim_xy = centre + direction * radius
        closing = direction  # across the wall
        wall = crown[np.linalg.norm(crown[:, :2] - rim_xy, axis=1) < 0.012]
        thickness = float(np.ptp(wall[:, :2] @ closing)) if len(wall) >= 3 else 0.01
        pos = np.array([rim_xy[0], rim_xy[1], height])
        yaw = closing_yaw(closing)
        pose = top_down_pose(pos, yaw)
        pre = top_down_pose(pos + np.array([0.0, 0.0, self.cfg.standoff_m]), yaw)
        opening = self.caps.gripper.max_opening_m
        side = int(round(np.degrees((chosen - base_angle + np.pi) % (2 * np.pi) - np.pi)))
        cand = GraspCandidate(
            candidate_id=f"{obj.object_id}#rim{index}",
            object_id=obj.object_id,
            approach="rim",
            pose=pose,
            pre_pose=pre,
            width_m=max(thickness, 0.005),
            opening_m=opening,
            height_m=height,
            clearance_m=float("inf"),
            score=0.0,
            notes=[
                f"pinches the rim wall {abs(side)} degrees "
                f"{'from' if side else 'on'} the side nearest the robot"
                + (" (clear of a handle or other protrusion)" if len(outliers) else "")
                + f", {RIM_DEPTH_M * 100:.0f} cm below the rim of a "
                f"{2 * radius * 100:.0f} cm opening"
            ],
        )
        reason = (
            check_reach(pose, self.caps)
            or check_table(pose, self.caps)
            or check_retreat(pose, self.cfg.lift_m, self.caps)
        )
        if reason is None:
            reason, clearance = check_finger_collision(pose, opening, self.caps, obstacles)
            cand.clearance_m = clearance
            if reason is None:
                reason = check_approach_path(pre, pose, opening, self.caps, obstacles)
        cand.rejected = reason
        cand.score = self._score(obj, cand) * 0.8 if reason is None else -1.0
        return cand

    def _heights(self, obj: ObjectDescriptor) -> list[float]:
        """Grasp heights from just below the top face downward.

        The first is the shallow top grasp that works for a flat object. The rest walk down
        the body, which is how a bottle's neck or a mug's waist gets found.
        """
        top = obj.top_z
        floor = self.caps.workspace.table_z + GRASP_FLOOR_M
        deepest = max(obj.bottom_z + 0.01, floor, top - min(0.12, obj.height * 0.8))
        n = max(1, self.cfg.heights)
        first = max(
            top - max(self.cfg.min_grasp_depth_m, min(0.02, obj.height * 0.35)),
            min(floor, top - self.cfg.min_grasp_depth_m),
        )
        if n == 1 or deepest >= first - 1e-6:
            return [first]
        return list(np.linspace(first, deepest, n))

    def _top_candidate(
        self,
        obj: ObjectDescriptor,
        height: float,
        obstacles: npt.NDArray,
        index: int,
        view_dir: npt.ArrayLike | None = None,
    ) -> GraspCandidate | None:
        """A top-down candidate, with width measured over the band the fingers will occupy.

        The fingertips sit at ``height`` and the fingers extend upward, so the jaw must clear
        everything between the grasp height and the object's top. A thin slab at one height
        is insufficient: from a single camera it holds only the two visible side walls,
        whose outline understates the width (a 7 cm bottle measures 4.9 cm).
        """
        pts = obj.points
        band_top = min(obj.top_z, height + self.caps.gripper.finger_length_m)
        band = pts[(pts[:, 2] >= height - self.cfg.slab_m / 2) & (pts[:, 2] <= band_top)]
        if len(band) < 8:
            return None
        sees_top = band_top >= obj.top_z - 1e-6
        # The centre comes from the top slab, area-weighted: the part of the object a single
        # camera sees completely, weighted so denser near-side sampling does not pull the
        # aim toward the camera. The width comes from a direction the camera can measure.
        crown = top_slab(pts, obj.top_z)
        width, length, closing, bounded = trusted_width_direction(
            band,
            crown,
            view_dir,
            fits_m=self._fits_m(),
            elevation_rad=getattr(self, "_view_elevation", None),
        )
        centre_xy = grasp_point(crown) if len(crown) >= 8 else footprint_centre(band)
        width = self._local_width(band, crown, centre_xy, closing, width, bounded)
        pos = np.array([centre_xy[0], centre_xy[1], height])
        yaw = closing_yaw(closing)
        pose = top_down_pose(pos, yaw)
        # Current adapters open fully. Check the actual finger volume, not a narrower
        # requested gap that the binary gripper cannot realize.
        opening = self.caps.gripper.max_opening_m
        pre = top_down_pose(pos + np.array([0.0, 0.0, self.cfg.standoff_m]), yaw)

        cand = GraspCandidate(
            candidate_id=f"{obj.object_id}#top{index}",
            object_id=obj.object_id,
            approach="top",
            pose=pose,
            pre_pose=pre,
            width_m=width,
            opening_m=opening,
            height_m=height,
            clearance_m=float("inf"),
            score=0.0,
            notes=[f"cross-section {length * 100:.0f} by {width * 100:.1f} cm at this height"],
        )

        too_wide = check_opening(width, self.caps, self.cfg.jaw_margin_m)
        if too_wide and bounded:
            too_wide += (
                " (an upper bound: this side runs along the camera's line of "
                "sight and was only partly seen)"
            )
        if bounded and not too_wide:
            cand.notes.append(
                f"width along the line of sight is an estimate, at most {width * 100:.1f} cm"
            )
        reason = (
            self._single_view_limit(sees_top, height)
            or too_wide
            or check_reach(pose, self.caps)
            or check_table(pose, self.caps)
            or check_retreat(pose, self.cfg.lift_m, self.caps)
        )
        if reason is None:
            reason, clearance = check_finger_collision(pose, opening, self.caps, obstacles)
            cand.clearance_m = clearance
            if reason is None:
                reason = check_approach_path(pre, pose, opening, self.caps, obstacles)
            if reason is not None:
                turned = self._turn_clear(
                    obj, band, crown, centre_xy, closing, pos, opening, obstacles, view_dir
                )
                if turned is not None:
                    cand.pose, cand.pre_pose, cand.width_m, cand.clearance_m, note = turned
                    cand.notes.append(note)
                    reason = None
        cand.rejected = reason
        cand.score = self._score(obj, cand)
        return cand

    def _turn_clear(self, obj, band, crown, centre_xy, closing, pos, opening, obstacles, view_dir):
        """Another jaw angle at the same place and height, when the first one collides.

        The first angle is the best measured, not the only option: a bottle standing 3 cm
        from a pail can be taken with the jaw turned away from the pail."""
        base = float(np.arctan2(closing[1], closing[0]))
        unseen = 1.0 + unseen_fraction(getattr(self, "_view_elevation", None))
        view = None
        if view_dir is not None:
            view = np.asarray(view_dir, dtype=float) / max(1e-9, np.linalg.norm(view_dir))
        best = None
        for deg in (90, 60, -60, 30, -30, 120, -120, 150):
            a = base + np.radians(deg)
            d = np.array([np.cos(a), np.sin(a)])
            span = local_span(band, centre_xy, d)
            if span is None:
                continue
            if view is not None and abs(float(d @ view)) > np.cos(np.radians(55)):
                span *= unseen  # partly seen along the line of sight: an upper bound
            if check_opening(span, self.caps, self.cfg.jaw_margin_m):
                continue
            yaw = closing_yaw(d)
            pose = top_down_pose(pos, yaw)
            pre = top_down_pose(pos + np.array([0.0, 0.0, self.cfg.standoff_m]), yaw)
            reason, clearance = check_finger_collision(pose, opening, self.caps, obstacles)
            if reason is None:
                reason = check_approach_path(pre, pose, opening, self.caps, obstacles)
            if reason is None and (best is None or clearance > best[3]):
                best = (
                    pose,
                    pre,
                    float(span),
                    float(clearance),
                    f"jaw turned {deg:+d} degrees to clear a neighbour",
                )
        return best

    def _single_view_limit(self, sees_top: bool, height: float) -> str | None:
        """Reject a grasp whose width cannot be measured from one camera.

        From a single elevated view, the band above a mid-body grasp contains only the two
        visible side walls, whose outline understates the width (a 7 cm bottle measures
        4.9 cm). A band that reaches the object's top face is fully observed. The limit is
        a property of single-view sensing; a second viewpoint or a learned grasp backend
        that infers unseen geometry removes it.
        """
        if sees_top:
            return None
        return (
            f"cannot measure the width at {height * 100:.0f} cm from one camera: the far "
            "side of the object is hidden at this height, so the jaw opening would be a "
            "guess. Grasp nearer the top, or use a different object"
        )

    def _side_candidate(
        self,
        obj: ObjectDescriptor,
        obstacles: npt.NDArray,
        index: int,
        view_dir: npt.ArrayLike | None = None,
    ) -> GraspCandidate | None:
        """A horizontal approach: the fingers come in level with the object and close across it.

        The jaw only spans the object's width rather than clearing its top, which suits a
        tall bottle, jug or mug whose band above the grasp height is wider than the jaw.
        The approach comes from the side nearer the robot's base, the finger boxes are
        checked against the table, other objects and the workspace as for top-down
        candidates, and the lift is vertical.
        """
        pts = obj.points
        height = obj.top_z - min(0.04, obj.height * 0.3)
        floor = self.caps.workspace.table_z + self.caps.gripper.pad_width_m / 2 + 0.006
        height = max(height, floor)
        if height > obj.top_z - 0.005:
            return None
        crown = top_slab(pts, obj.top_z)
        slab = cross_section(pts, height + self.cfg.slab_m / 2, self.cfg.slab_m)
        if len(slab) < 8:
            return None
        width, _, closing, bounded = trusted_width_direction(
            slab,
            crown,
            view_dir,
            fits_m=self._fits_m(),
            elevation_rad=getattr(self, "_view_elevation", None),
        )
        centre_xy = grasp_point(crown) if len(crown) >= 8 else footprint_centre(slab)
        width = self._local_width(slab, crown, centre_xy, closing, width, bounded)
        pos = np.array([centre_xy[0], centre_xy[1], height])
        # Two horizontal approaches are perpendicular to the closing axis; take the one
        # whose standoff is nearer the robot's base and inside the workspace.
        options = []
        for sign in (1.0, -1.0):
            approach = sign * np.array([-closing[1], closing[0], 0.0])
            standoff = pos - approach * self.cfg.standoff_m
            options.append((float(np.linalg.norm(standoff[:2])), approach, standoff))
        options.sort(key=lambda o: o[0])
        approach, standoff = options[0][1], options[0][2]
        pose = side_pose(pos, approach, np.array([closing[0], closing[1], 0.0]))
        pre = Pose(standoff, pose.quat)
        opening = self.caps.gripper.max_opening_m
        cand = GraspCandidate(
            candidate_id=f"{obj.object_id}#side{index}",
            object_id=obj.object_id,
            approach="side",
            pose=pose,
            pre_pose=pre,
            width_m=width,
            opening_m=opening,
            height_m=height,
            clearance_m=float("inf"),
            score=0.0,
            notes=[
                f"horizontal, from {_compass(approach)}, jaw across "
                f"{width * 100:.1f} cm at {height * 100:.0f} cm up"
            ],
        )
        too_wide = check_opening(width, self.caps, self.cfg.jaw_margin_m)
        if too_wide and bounded:
            too_wide += (
                " (an upper bound: this side runs along the camera's line of "
                "sight and was only partly seen)"
            )
        reason = (
            too_wide
            or check_reach(pose, self.caps)
            or check_reach(pre, self.caps)
            or check_fingers_above_table(pose, opening, self.caps)
            or check_retreat(pose, self.cfg.lift_m, self.caps)
        )
        if reason is None:
            reason, clearance = check_finger_collision(pose, opening, self.caps, obstacles)
            cand.clearance_m = clearance
            if reason is None:
                reason = check_approach_path(pre, pose, opening, self.caps, obstacles)
        cand.rejected = reason
        cand.score = self._score(obj, cand)
        return cand

    def _local_width(self, band, crown, centre_xy, closing, width, bounded) -> float:
        """The width over the stretch the pads touch, with the single-view treatment of
        ``trusted_width_direction``: an upper bound where the direction was only partly
        seen, otherwise at least the top face's span."""
        local = local_span(band, centre_xy, closing)
        if local is None:
            return width
        if bounded:
            local *= 1.0 + unseen_fraction(getattr(self, "_view_elevation", None))
        top_local = local_span(crown, centre_xy, closing)
        if top_local is not None and not bounded:
            local = max(local, top_local)
        return float(local) if 0 < local < width - 0.005 else width

    def _fits_m(self) -> float:
        return self.caps.gripper.max_opening_m - 2 * self.cfg.jaw_margin_m

    def _score(self, obj: ObjectDescriptor, cand: GraspCandidate) -> float:
        """Prefer a confident, roomy, high grasp on a well-seen object.

        Deliberately simple: every term can be read off the candidate, so disagreement
        between a policy's choice and this ranking is interpretable.
        """
        if not cand.feasible:
            return -1.0
        jaw = self.caps.gripper.max_opening_m
        fit = 1.0 - min(1.0, cand.width_m / jaw)
        clearance = min(1.0, (cand.clearance_m if np.isfinite(cand.clearance_m) else 0.5) / 0.05)
        depth = min(1.0, max(0.0, (obj.top_z - cand.height_m) / 0.04))
        seen = min(1.0, obj.visible_fraction / 0.8)
        return float(0.4 * fit + 0.25 * clearance + 0.2 * depth + 0.15 * seen)


def obstacles_for(target_id: str, catalog, table_z: float | None = None) -> npt.NDArray[np.float64]:
    """Every catalogued point that is not the target, for collision checks."""
    chunks = [
        o.points for o in catalog.objects if o.object_id != target_id and o.points is not None
    ]
    if not chunks:
        return np.zeros((0, 3))
    pts = np.concatenate(chunks, axis=0)
    if table_z is not None:
        pts = pts[pts[:, 2] > table_z + 0.005]
    return pts
