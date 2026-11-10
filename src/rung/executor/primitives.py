"""L2 primitives: grasp, place, move_to, inspect, observe.

Each primitive is closed loop and returns a structured result with a reason. Invariants:

* No unverified success. ``grasp`` lifts and checks that the object came with the hand
  before reporting ``grasped``; finger travel alone cannot distinguish a grasp from an
  obstructed close.
* No privileged state. Results derive only from observations available to the policy,
  never from the grader.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np

from rung.contracts.action import (
    ExecutionOutcome,
    GripperAction,
    GripperCommand,
    TCPTarget,
)
from rung.contracts.frames import Pose, depth_image_to_points, quat_to_rot
from rung.embodiment import Embodiment
from rung.grasp.analytic import AnalyticGraspPlanner, obstacles_for, top_down_pose
from rung.grasp.types import GraspCandidate, GraspPlan
from rung.perception.cloud import HAND_EXCLUSION_M, rim_model, support_height
from rung.perception.memory import ObjectMemory
from rung.perception.service import PerceptionService
from rung.perception.types import Catalog, ObjectDescriptor
from rung.policy import ModelUsage

LIFT_CHECK_M = 0.05
"""Lift height before verifying the grasp."""

LIFT_CONFIRM_M = 0.02
"""Minimum drop of the surface at the vacated spot for the object to count as lifted."""
CLOSED_TRAVEL = 0.95
THIN_HOLD_M = 0.015
"""Below this expected grasp width, jaw travel cannot distinguish a hold from a miss."""
COLUMN_HEADROOM_M = 0.02
"""Height above an object's previous top still searched by the vacated-spot check."""
LIFT_DONE_FRACTION = 0.7
"""A lift reaching less than this fraction of its target is anchored by what the jaw holds."""

TABLE_CLEARANCE_M = 0.025
"""Depth returns within this height of the table are treated as table, not a held object."""

MOTION_RETRIES = 3
"""Maximum issues of one motion while it is still converging on its target.

Measured convergence leaves under 2 cm after one full time budget, and each retry starts
where the last stopped, so the residual shrinks quickly. A blocked motion exits on the
progress test, not by exhausting this bound.
"""

MOTION_PROGRESS_M = 0.005
"""Minimum travel per attempt for a re-issue to be worthwhile."""

PLACE_CLEARANCE_M = 0.03
"""Height of the held object's underside above a container's rim at release.

Larger values (6 cm) let objects bounce off the rim. 3 cm clears the rim under
measurement noise; a lowering that contacts the rim is handled in ``Executor.place``.
"""

COMFORTABLE_REACH_M = 0.50
MIN_COMFORTABLE_REACH_M = 0.30
"""Preferred and minimum horizontal distance from the base for a pointing-down Franka hand.
Measured: a step aimed at 0.17 m stalled; steps to 0.31 m and beyond completed."""

BESIDE = {
    "left_of": np.array([0.0, 1.0]),
    "right_of": np.array([0.0, -1.0]),
    "in_front_of": np.array([-1.0, 0.0]),
    "behind": np.array([1.0, 0.0]),
}
"""Unit directions in the robot base frame (xy) for relational placements such as
'left of the bowl'."""
BESIDE_MARGIN_M = 0.04
BESIDE_DROP_M = 0.015

ASIDE_CLEARANCE_M = 0.06
"""Minimum distance between the step-aside path and any other object."""

CARRY_CLEARANCE_M = 0.05
"""Clearance between a carried object's underside and everything below its path."""

WAYPOINT_TOLERANCE_M = 0.02
"""Arrival tolerance for pre-grasp, carry and retreat poses."""

GRASP_TOLERANCE_M = 0.01
"""Arrival tolerance for the fingertips at grasp or release height."""

LOWER_RELEASE_ANYWAY_M = 0.04
"""A lowering that stops within this distance of the release height releases anyway: the
object is over the container and resting on its rim, so opening drops it in."""

CARRY_HEIGHT_M = 0.22

FREE_SPEED_MPS = 0.25
CONTACT_SPEED_MPS = 0.08
LIFT_SPEED_MPS = 0.12
"""Planned TCP speeds, realised by the bridge after tracking-gain compensation.

Free-space moves use full speed. Moves that may end in contact (final descent, lowering
into a container) stay slow because relative IK pushes a blocked arm by the commanded step,
and a compensated step is three to four times larger. Lifts are moderate so acceleration
does not shake the object out of the fingers."""


class Outcome(StrEnum):
    GRASPED = "grasped"
    UNKNOWN = "unknown"
    CLOSURE_OBSTRUCTED = "closure_obstructed"
    MISSED = "missed"
    REJECTED = "rejected"
    PLACED = "placed"
    BLOCKED = "blocked"
    CLEARANCE_VIOLATED = "clearance_violated"
    MOVED = "moved"
    OBSERVED = "observed"
    NOT_HOLDING = "not_holding"
    DROPPED = "dropped"


@dataclass
class PrimitiveResult:
    primitive: str
    outcome: Outcome
    object_id: str | None = None
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome in (Outcome.GRASPED, Outcome.PLACED, Outcome.MOVED, Outcome.OBSERVED)

    def render(self) -> str:
        head = f"{self.primitive}"
        if self.object_id:
            head += f"({self.object_id})"
        line = f"{head} -> {self.outcome.value}"
        return f"{line}: {self.reason}" if self.reason else line


class MotionIncomplete(RuntimeError):
    def __init__(self, report):
        self.outcome = report.outcome
        self.report = report
        super().__init__(report.reason or f"motion {report.outcome.value}; primitive stopped")


class Executor:
    """Runs primitives against an embodiment, using perception it re-reads each time."""

    def __init__(
        self,
        embodiment: Embodiment,
        perception: PerceptionService,
        planner: AnalyticGraspPlanner,
        memory: ObjectMemory | None = None,
        instruction: str = "",
        trace=None,
        episode: str = "",
    ):
        self.emb = embodiment
        self.perception = perception
        self.planner = planner
        self.memory = memory or ObjectMemory()
        self.instruction = instruction
        self.catalog: Catalog | None = None
        self.last_seen: dict[str, ObjectDescriptor] = {}
        """Latest measurement of every object ever seen, by stable id. Grounding recall
        varies between looks, so place targets can fall back to this."""
        self.plans: dict[str, GraspPlan] = {}
        self.trace = trace
        self.episode = episode
        self.motion_handler = None
        self.pending_usage = ModelUsage()
        self.unverified_hold = False
        self.held_overhang_m: float | None = None
        self.hold_threshold = CLOSED_TRAVEL
        self.jaw_informative = True
        """Whether jaw travel is a usable hold signal for the current grasp.
        ``hold_threshold`` is the travel above which the fingers closed on nothing; it is
        set per grasp from the expected width (a 5 mm wall stops the jaw near 0.94)."""

    # ---- transcript ---------------------------------------------------------

    def log(self, kind: str, **payload) -> None:
        """Emit one harness step to the transcript.

        Covers grounding calls, perception measurements, identity changes, grasp plans and
        the checks behind each verdict. Stamped with simulated time; the trace adds wall time.
        """
        if self.trace is None:
            return
        try:
            t = float(self.emb.observe().t)
        except Exception:  # never let logging stop a primitive
            t = 0.0
        try:
            self.trace.emit(kind, t, self.episode, **payload)
        except Exception:  # an unloggable payload is a transcript gap, not a robot failure
            pass

    # ---- perception -------------------------------------------------------

    def refresh(self, reuse_boxes: bool = False) -> Catalog:
        """Re-read the scene so no primitive acts on a stale view.

        ``reuse_boxes`` re-measures the previous detections in a fresh depth image instead
        of calling the grounding model. Used after a verified grasp, where only the held
        object has moved; a grounding call costs 3 to 12 s of simulated time.
        """
        obs = self.emb.observe()
        boxes = (
            list(getattr(self.perception.grounding, "last_detections", None) or [])
            if reuse_boxes
            else None
        )
        catalog = self.perception.build(
            obs,
            self.instruction,
            table_z=self.emb.capabilities.workspace.table_z,
            jaw_m=self.emb.capabilities.gripper.max_opening_m,
            detections=boxes or None,
        )
        if catalog.extra.get("grounding_usage") is not None:
            self.pending_usage.add(catalog.extra["grounding_usage"])
        if catalog.extra.get("reused_boxes"):
            self.log(
                "perception",
                reused_boxes=True,
                note="re-measured the last look's boxes in a fresh depth image; no grounding call",
            )
        else:
            self._log_grounding(catalog)
        if (
            not catalog.objects
            and catalog.failures
            and self.catalog is not None
            and self.catalog.objects
        ):
            # An empty result on a previously populated table is a grounding failure, not
            # an empty scene. Keep the last catalog, flag it as stale, and leave memory
            # untouched so no object is marked missing.
            self.catalog.failures = [
                "perception returned nothing this turn (" + catalog.failures[-1] + "); "
                "the catalog below is from the previous look and may be out of date"
            ]
            self.log(
                "perception",
                stale=True,
                failures=list(self.catalog.failures),
                note="kept the previous catalog because this look found nothing",
            )
            self.plans.clear()
            return self.catalog
        self.catalog = self.memory.update(catalog, tcp_pos=obs.tcp.pos)
        for obj in self.catalog.objects:
            self.last_seen[obj.object_id] = obj
        self._log_perception(self.catalog)
        self.plans.clear()
        return self.catalog

    def _log_grounding(self, catalog: Catalog) -> None:
        g = self.perception.grounding
        usage = catalog.extra.get("grounding_usage")
        self.log(
            "grounding",
            source=getattr(g, "name", type(g).__name__),
            reused=bool(getattr(g, "last_reused", False)),
            prompt=getattr(g, "last_prompt", None),
            response_text=getattr(g, "last_response_text", None),
            detections=[
                {"label": d.label, "box_px": [round(float(v), 1) for v in d.box]}
                for d in (getattr(g, "last_detections", None) or [])
            ],
            usage=dataclasses.asdict(usage) if usage is not None else None,
        )

    def _log_perception(self, catalog: Catalog) -> None:
        jaw = self.emb.capabilities.gripper.max_opening_m
        self.log(
            "perception",
            table_z=self.emb.capabilities.workspace.table_z,
            failures=list(catalog.failures),
            objects=[
                {
                    "object_id": o.object_id,
                    "label": o.label,
                    "at": [round(float(o.grasp_centre[0]), 3), round(float(o.grasp_centre[1]), 3)],
                    "size_cm": [
                        round(o.length * 100, 1),
                        round(o.width * 100, 1),
                        round(o.height * 100, 1),
                    ],
                    "top_cm": round(o.top_z * 100, 1),
                    "bottom_cm": round(o.bottom_z * 100, 1),
                    "grasp_width_cm": round(o.grasp_width * 100, 1),
                    "points": o.n_points,
                    "uncertain": o.uncertainty_reason if o.uncertain else None,
                    "line": o.describe(jaw),
                }
                for o in catalog.objects
            ],
            memory=list(getattr(self.memory, "last_changes", []) or []),
        )

    def catalog_or_refresh(self) -> Catalog:
        return self.catalog if self.catalog is not None else self.refresh()

    def object(self, object_id: str) -> ObjectDescriptor | None:
        return self.catalog_or_refresh().get(object_id)

    def plan_for(self, object_id: str) -> GraspPlan | None:
        if object_id in self.plans:
            return self.plans[object_id]
        catalog = self.catalog_or_refresh()
        obj = catalog.get(object_id)
        if obj is None:
            return None
        from rung.perception.cloud import view_direction_xy, view_elevation

        cam = self.emb.observe().scene
        cam_pos = cam.pose.pos if cam.pose is not None else None
        plan = self.planner.plan(
            obj,
            obstacles_for(object_id, catalog, self.emb.capabilities.workspace.table_z),
            view_dir=view_direction_xy(cam_pos, obj.grasp_centre),
            view_elevation=view_elevation(cam_pos, obj.grasp_centre),
        )
        self.plans[object_id] = plan
        self.log(
            "plan",
            object_id=object_id,
            rendered=plan.render(),
            candidates=[
                {
                    "id": c.candidate_id,
                    "approach": c.approach,
                    "feasible": c.feasible,
                    "height_cm": round(c.height_m * 100, 1),
                    "width_cm": round(c.width_m * 100, 1),
                    "clearance_cm": (
                        round(c.clearance_m * 100, 1) if np.isfinite(c.clearance_m) else None
                    ),
                    "score": round(float(c.score), 3),
                    "rejected": c.rejected,
                    "notes": list(c.notes),
                    "pose": [round(float(v), 3) for v in c.pose.pos],
                }
                for c in plan.candidates
            ],
        )
        return plan

    # ---- primitives -------------------------------------------------------

    def propose_grasps(self, object_id: str) -> PrimitiveResult:
        plan = self.plan_for(object_id)
        if plan is None:
            return self._unknown("propose_grasps", object_id)
        return PrimitiveResult(
            "propose_grasps",
            Outcome.OBSERVED,
            object_id,
            reason=None,
            detail={
                "candidates": [c.candidate_id for c in plan.feasible],
                "rendered": plan.render(),
            },
        )

    def inspect(self, object_id: str) -> PrimitiveResult:
        obj = self.object(object_id)
        if obj is None:
            return self._unknown("inspect", object_id)
        plan = self.plan_for(object_id)
        return PrimitiveResult(
            "inspect",
            Outcome.OBSERVED,
            object_id,
            detail={
                "description": obj.describe(self.emb.capabilities.gripper.max_opening_m),
                "visible_fraction": obj.visible_fraction,
                "uncertain": obj.uncertain,
                "grasps": plan.render() if plan else "no grasp plan",
            },
        )

    def grasp(self, object_id: str, candidate_id: str | None = None) -> PrimitiveResult:
        """Approach, close, lift, and verify. Only a verified hold reports ``grasped``."""
        if self.unverified_hold:
            return PrimitiveResult(
                "grasp",
                Outcome.REJECTED,
                object_id,
                reason="attachment is uncertain; release before another grasp",
            )
        if self.memory.held is not None:
            return PrimitiveResult(
                "grasp",
                Outcome.REJECTED,
                object_id,
                reason=f"already holding {self.memory.held}; place or release it first",
            )
        plan = self.plan_for(object_id)
        if plan is None:
            return self._unknown("grasp", object_id)
        cand = plan.get(candidate_id) if candidate_id else plan.best()
        if cand is None:
            reasons = sorted({c.rejected for c in plan.candidates if c.rejected})
            return PrimitiveResult(
                "grasp",
                Outcome.REJECTED,
                object_id,
                reason=(
                    f"no feasible grasp: {'; '.join(reasons)}"
                    if reasons
                    else "no grasp candidates could be built from the visible geometry"
                ),
                detail={"rendered": plan.render()},
            )
        if not cand.feasible:
            return PrimitiveResult(
                "grasp",
                Outcome.REJECTED,
                object_id,
                reason=f"{cand.candidate_id} was rejected: {cand.rejected}",
            )

        before = self.object(object_id)
        before_top = before.top_z if before else None
        before_xy = before.grasp_centre[:2].copy() if before else None

        open_cmd = GripperCommand(GripperAction.OPEN, opening_m=cand.opening_m)
        try:
            self._go(
                cand.pre_pose,
                open_cmd,
                meta={"phase": "pre_grasp", "cand": cand.candidate_id},
                tolerance_m=WAYPOINT_TOLERANCE_M,
            )
        except MotionIncomplete as exc:
            return PrimitiveResult(
                "grasp",
                Outcome.BLOCKED,
                object_id,
                reason=f"could not reach the approach pose: {exc}",
            )

        try:
            self._go(
                cand.pose,
                GripperCommand(GripperAction.HOLD),
                meta={"phase": "descend", "cand": cand.candidate_id},
                tolerance_m=GRASP_TOLERANCE_M,
                speed_mps=CONTACT_SPEED_MPS,
            )
        except MotionIncomplete as exc:
            # Fingertips stopped above grasp height: they rest on the object or a neighbour,
            # so the object is wider or taller than measured. Retreat before reporting so
            # the hand is out of the next look.
            short = float(exc.report.final_tcp.pos[2] - cand.pose.pos[2]) * 100
            self._retreat(cand.pre_pose, phase="retreat_after_block")
            self.memory.record(
                "grasp", object_id, Outcome.BLOCKED.value, f"descent stopped {short:.1f} cm high"
            )
            return PrimitiveResult(
                "grasp",
                Outcome.BLOCKED,
                object_id,
                reason=(
                    f"the descent stopped {short:.1f} cm above the grasp height: {exc}. "
                    "The fingers most likely came down on the object or a neighbour, "
                    "so it is wider or nearer the hand than measured. The hand has "
                    "moved back up; look again before choosing"
                ),
                detail={"execution": exc.outcome.value, "stopped_high_m": short / 100},
            )

        self.unverified_hold = True
        opening = self.emb.capabilities.gripper.max_opening_m
        expected = 1.0 - float(cand.width_m) / max(opening, 1e-6)
        # Midway between the expected stopping travel and fully closed, never below
        # CLOSED_TRAVEL.
        self.hold_threshold = max(CLOSED_TRAVEL, min(0.99, expected + 0.5 * (1.0 - expected)))
        # On thin walls (e.g. a 5 mm mug wall) a Robotiq closes fully, as on air, so jaw
        # travel is uninformative and the scene checks decide.
        self.jaw_informative = float(cand.width_m) >= THIN_HOLD_M
        self._go(
            self.emb.observe().tcp,
            GripperCommand(GripperAction.CLOSE),
            meta={"phase": "close", "cand": cand.candidate_id},
        )

        travel = self.emb.observe().gripper.travel
        self.log(
            "check",
            check="jaw after close",
            object_id=object_id,
            candidate=cand.candidate_id,
            travel=round(float(travel), 3),
            closed_on_nothing_above=round(float(self.hold_threshold), 3),
            verdict=(
                "jaw cannot tell on a thin wall; lifting to check the scene"
                if not self.jaw_informative
                else "closed on nothing"
                if travel > self.hold_threshold
                else "stopped on something"
            ),
        )
        if travel > self.hold_threshold and self.jaw_informative:
            self.unverified_hold = False
            self.memory.record("grasp", object_id, Outcome.MISSED.value, "closed on nothing")
            return PrimitiveResult(
                "grasp",
                Outcome.MISSED,
                object_id,
                reason="the fingers met with nothing between them",
                detail={"travel": travel, "candidate": cand.candidate_id},
            )

        # Lift, then step aside if needed. After a pure lift the hand and any held object
        # remain over the spot, so the camera sees something tall there either way.
        lift_to = Pose(cand.pose.pos + np.array([0.0, 0.0, LIFT_CHECK_M]), cand.pose.quat)
        # Lift and step are best effort: the checks read the achieved motion, so a motion
        # that stops slightly short is not treated as a failed grasp.
        rise_from = float(self.emb.observe().tcp.pos[2])
        self._go_best_effort(lift_to, meta={"phase": "lift_check"}, speed_mps=LIFT_SPEED_MPS)
        rose = float(self.emb.observe().tcp.pos[2]) - rise_from
        lifted = rose >= LIFT_DONE_FRACTION * max(1e-6, float(lift_to.pos[2]) - rise_from)
        lift_match = self._attachment_match(before, cand.pose)
        self.log(
            "check",
            check="lift",
            object_id=object_id,
            rose_cm=round(rose * 100, 1),
            wanted_cm=round((float(lift_to.pos[2]) - rise_from) * 100, 1),
            lifted=bool(lifted),
            surfaces_match_hand=lift_match,
            jaw=round(float(self.emb.observe().gripper.travel), 3),
        )
        if not lifted and (
            not self.jaw_informative or self.emb.observe().gripper.travel <= self.hold_threshold
        ):
            # The jaw holds something anchored (a container rim, the table edge, a pinned
            # neighbour), not the target. Release and retreat.
            self._go(
                self.emb.observe().tcp,
                GripperCommand(GripperAction.OPEN),
                meta={"phase": "release_anchored"},
            )
            self._retreat(cand.pre_pose, phase="retreat_after_anchored")
            self.unverified_hold = False
            self.refresh()
            self.memory.record(
                "grasp",
                object_id,
                Outcome.CLOSURE_OBSTRUCTED.value,
                "closed on something that would not lift",
            )
            return PrimitiveResult(
                "grasp",
                Outcome.CLOSURE_OBSTRUCTED,
                object_id,
                reason=(
                    f"the fingers closed on something that would not lift (the hand rose "
                    f"{rose * 100:.1f} cm of {LIFT_CHECK_M * 100:.0f}): most likely the rim "
                    "or edge of something next to the object. The hand has let go and "
                    "moved back up; try a candidate whose fingers stay clear of it"
                ),
                detail={"travel": travel, "rose_m": rose, "candidate": cand.candidate_id},
            )

        # Read the vacated spot from the lift pose first; step aside only if inconclusive,
        # since a step aside costs a round trip of simulated time and displaces the object.
        # Only a clear drop is trusted here: the check reads a vertical column at the old
        # position, and an object lifted straight up is still in that column, reading as a
        # risen surface. None, small or negative readings fall through to the step aside.
        aside_match: bool | None = None
        aside_cleared = True
        dropped = self._surface_dropped(before_xy, before_top)
        self.log(
            "check",
            check="vacated spot after lift",
            object_id=object_id,
            fell_cm=None if dropped is None else round(dropped * 100, 1),
            needs_cm=round(LIFT_CONFIRM_M * 100, 1),
            verdict=(
                "unmeasurable"
                if dropped is None
                else "emptied"
                if dropped >= LIFT_CONFIRM_M
                else "inconclusive, stepping aside"
            ),
        )
        if dropped is None or dropped < LIFT_CONFIRM_M:
            # Move the smallest distance that clears the old footprint and look again.
            aside = Pose(
                lift_to.pos
                + self._aside_offset(
                    lift_to.pos, before.length if before else 0.0, object_id=object_id
                ),
                cand.pose.quat,
            )
            wanted = float(np.linalg.norm(aside.pos - self.emb.observe().tcp.pos))
            report = self._go_best_effort(aside, meta={"phase": "step_aside"})
            aside_cleared = wanted < 1e-6 or report.executed_lin_m >= 0.7 * wanted
            aside_match = self._attachment_match(before, cand.pose)
            dropped = self._surface_dropped(before_xy, before_top, hand_left=aside_cleared)
            if not aside_cleared and dropped is not None and dropped < -LIFT_CONFIRM_M:
                # The hand never left the measured column, so the risen surface is the
                # hand itself. No evidence either way.
                dropped = None
            self.log(
                "check",
                check="vacated spot after step aside",
                object_id=object_id,
                step_cm=round(wanted * 100, 1),
                moved_cm=round(float(report.executed_lin_m) * 100, 1),
                cleared=bool(aside_cleared),
                surfaces_match_hand=aside_match,
                fell_cm=None if dropped is None else round(dropped * 100, 1),
            )

        # The fingers hold something (a closure on nothing returned above). The target is
        # held if its spot emptied and the jaw is still stopped after the motion: a jaw
        # whose object was knocked loose keeps closing toward the command. The jaw needs no
        # line of sight. Surface matches only grade confidence, since the hand often
        # occludes the held object from the scene camera.
        still_holding = (
            self.emb.observe().gripper.travel <= self.hold_threshold or not self.jaw_informative
        )
        place_emptied = dropped is not None and dropped >= LIFT_CONFIRM_M
        attached = place_emptied and still_holding
        corroborated = sum(1 for m in (lift_match, aside_match) if m is True)
        self.log(
            "check",
            check="grasp verdict inputs",
            object_id=object_id,
            jaw=round(float(self.emb.observe().gripper.travel), 3),
            still_holding=bool(still_holding),
            place_emptied=bool(place_emptied),
            corroborated=corroborated,
        )
        confidence = (
            "confirmed"
            if corroborated == 2
            else "corroborated once"
            if corroborated == 1
            else "inferred from the emptied place alone; the hand hides what it "
            "holds from this camera"
        )
        if attached and dropped is not None and dropped >= LIFT_CONFIRM_M:
            self.unverified_hold = False
            self.memory.set_held(object_id)  # before the refresh, so the id survives the lift
            # Distance the object hangs below the fingertips, needed to set release height.
            self.held_overhang_m = self._overhang(cand, before)
        self.refresh(reuse_boxes=attached and dropped is not None and dropped >= LIFT_CONFIRM_M)

        if attached and dropped is not None and dropped >= LIFT_CONFIRM_M:
            self.memory.record(
                "grasp", object_id, Outcome.GRASPED.value, f"its place fell {dropped * 100:.0f} cm"
            )
            return PrimitiveResult(
                "grasp",
                Outcome.GRASPED,
                object_id,
                reason=(
                    f"lifted and held: where it was standing is now {dropped * 100:.1f} cm "
                    f"lower and the jaw stopped at {travel:.2f} rather than closing, so the "
                    f"object came with the hand ({confidence})"
                ),
                detail={
                    "travel": travel,
                    "dropped_m": dropped,
                    "candidate": cand.candidate_id,
                    "lift_match": lift_match,
                    "aside_match": aside_match,
                    "confidence": confidence,
                },
            )

        if place_emptied and not still_holding:
            # It left its place but the jaw ran shut behind it: knocked away, not carried.
            self.unverified_hold = False
            self.memory.record("grasp", object_id, Outcome.MISSED.value, "knocked it away")
            return PrimitiveResult(
                "grasp",
                Outcome.MISSED,
                object_id,
                reason=(
                    "the object left its place but the fingers closed the rest of the "
                    "way afterwards, so it was knocked aside rather than picked up"
                ),
                detail={"travel": travel, "dropped_m": dropped, "candidate": cand.candidate_id},
            )

        if dropped is None and still_holding and lift_match is not False:
            # Spot unmeasurable, but the jaw stayed stopped through the lift; an object left
            # on the table would have let it close. Report a weak hold; the carry re-checks
            # the jaw before release.
            self.unverified_hold = False
            self.memory.set_held(object_id)
            self.held_overhang_m = self._overhang(cand, before)
            self.refresh()
            self.memory.record("grasp", object_id, Outcome.GRASPED.value, "jaw only")
            return PrimitiveResult(
                "grasp",
                Outcome.GRASPED,
                object_id,
                reason=(
                    "probably held: the jaw stopped at "
                    f"{travel:.2f} and stayed there through the lift, but the spot the "
                    "object came from could not be seen, so this rests on the jaw alone"
                ),
                detail={
                    "travel": travel,
                    "dropped_m": None,
                    "candidate": cand.candidate_id,
                    "lift_match": lift_match,
                    "aside_match": aside_match,
                    "confidence": "jaw only",
                },
            )

        if dropped is None:
            # Spot unmeasurable and no jaw evidence: attachment unknown.
            self.unverified_hold = True
            return PrimitiveResult(
                "grasp",
                Outcome.UNKNOWN,
                object_id,
                reason=(
                    "the fingers are holding something, but the spot the object came "
                    "from could not be measured, so what they hold is unconfirmed; "
                    "release before another grasp"
                ),
                detail={
                    "travel": travel,
                    "dropped_m": dropped,
                    "lift_match": lift_match,
                    "aside_match": aside_match,
                },
            )
        self.memory.record(
            "grasp", object_id, Outcome.CLOSURE_OBSTRUCTED.value, "it stayed where it was"
        )
        return PrimitiveResult(
            "grasp",
            Outcome.CLOSURE_OBSTRUCTED,
            object_id,
            reason=(
                "the fingers stopped on something but the object stayed where it was"
                + (f" (its place fell only {dropped * 100:.1f} cm)" if dropped is not None else "")
                + ", so it is not held. Try a different grasp candidate or a different object"
            ),
            detail={"travel": travel, "dropped_m": dropped, "candidate": cand.candidate_id},
        )

    def _attachment_match(self, obj: ObjectDescriptor, grasp_pose: Pose) -> bool | None:
        """Check observed surfaces at the object's predicted TCP-relative positions.

        Uses only object points below the fingertip plane, excluding fingers and palm.
        Geometric evidence of attachment, not identity; occluded geometry returns None.
        """
        obs = self.emb.observe()
        if self.jaw_informative and not (0.05 < obs.gripper.travel < self.hold_threshold):
            return False
        if obj is None or obj.points is None or not obs.scene.has_geometry:
            return None
        local = (obj.points - grasp_pose.pos) @ quat_to_rot(grasp_pose.quat)
        local = local[local[:, 2] > 0.008]
        if len(local) < 5:
            return None
        predicted = local[:: max(1, len(local) // 200)] @ quat_to_rot(obs.tcp.quat).T + obs.tcp.pos
        points, _ = depth_image_to_points(obs.scene.depth, obs.scene.K, obs.scene.pose, stride=2)
        if len(points) == 0:
            return None
        # Exclude table returns: a short object lifted 5 cm predicts surfaces within
        # tolerance of the table, which would verify a grasp of nothing.
        table_z = self.emb.capabilities.workspace.table_z
        points = points[points[:, 2] > table_z + TABLE_CLEARANCE_M]
        if len(points) == 0:
            return None
        region = points[
            np.all(
                (points >= predicted.min(axis=0) - 0.015)
                & (points <= predicted.max(axis=0) + 0.015),
                axis=1,
            )
        ]
        if len(region) < 5:
            return None
        distances = np.linalg.norm(predicted[:, None] - region[None, :], axis=2).min(axis=1)
        return bool(np.count_nonzero(distances < 0.012) >= max(5, len(predicted) * 0.25))

    def _aside_offset(
        self, pos: np.ndarray, object_length_m: float = 0.0, object_id: str | None = None
    ) -> np.ndarray:
        """Horizontal step for the lift check, inside the workspace.

        The step must exceed half the object's length, or a long object still overhangs its
        old footprint and reads as not moved. It must also keep the hand within comfortable
        reach, since the arm cannot hold a pointing-down hand close to its base. Among valid
        directions it prefers unobstructed paths, then lateral over inward, then more
        clearance, then reach nearest ``COMFORTABLE_REACH_M``.
        """
        ws = self.emb.capabilities.workspace
        # Half the length clears the object's centre, where the drop is measured; the
        # constant adds room for the hand. Longer steps cost convergence time.
        step = float(np.clip(object_length_m * 0.5 + 0.06, 0.08, 0.25))
        # Directions relative to the base. Outward is excluded: the forearm would cover the
        # vacated spot and read as the object.
        radial = np.asarray(pos[:2], dtype=np.float64)
        radial = radial / max(1e-9, float(np.linalg.norm(radial)))
        lateral = np.array([-radial[1], radial[0]])
        # Prefer the side away from neighbours so the held object is not pushed into them.
        others = self._neighbour_points(object_id)
        options = []
        for rank, direction in ((0, lateral), (0, -lateral), (1, -radial)):
            d = np.array([direction[0] * step, direction[1] * step, 0.0])
            p = pos + d
            if not ws.contains(p, margin=0.02):
                continue
            reach = float(np.linalg.norm(p[:2]))
            if reach < MIN_COMFORTABLE_REACH_M:
                continue
            clear = 1.0
            if len(others):
                path = np.stack([pos[:2] + d[:2] * f for f in (0.35, 0.7, 1.0)])
                clear = float(np.min(np.linalg.norm(others[:, None, :2] - path[None], axis=2)))
            blocked = clear < ASIDE_CLEARANCE_M
            options.append((blocked, rank, -min(clear, 0.15), abs(reach - COMFORTABLE_REACH_M), d))
        if not options:
            return np.zeros(3)
        return min(options, key=lambda o: o[:4])[4]

    def _overhang(self, cand, before) -> float:
        """Distance the held object extends below the fingertips: grasp height minus the
        object's underside (not grasp height alone, which assumes a table at z = 0)."""
        bottom = before.bottom_z if before is not None else self.emb.capabilities.workspace.table_z
        return float(max(0.0, cand.pose.pos[2] - bottom))

    def _neighbour_points(self, exclude_id: str | None) -> np.ndarray:
        """Catalogued points of every other object that stands clear of the table."""
        catalog = self.catalog
        if catalog is None:
            return np.zeros((0, 3))
        table_z = self.emb.capabilities.workspace.table_z
        chunks = [
            o.points
            for o in catalog.objects
            if o.object_id != exclude_id
            and o.object_id != self.memory.held
            and o.points is not None
            and len(o.points)
        ]
        if not chunks:
            return np.zeros((0, 3))
        pts = np.concatenate(chunks)
        return pts[pts[:, 2] > table_z + 0.01]

    def _surface_dropped(
        self, xy, before_top: float | None, hand_left: bool = False
    ) -> float | None:
        """Drop of the surface at the object's previous position, in metres.

        Points near the gripper are excluded as the gripper itself, so a held object's
        height is unmeasurable. The check instead measures the vacated spot: if its top
        surface is now much lower, the object came with the hand. Returns None when
        depth is missing or occluded.
        """
        if xy is None or before_top is None:
            return None
        obs = self.emb.observe()
        cam = obs.scene
        if not cam.has_geometry:
            return None
        pts, _ = depth_image_to_points(cam.depth, cam.K, cam.pose, stride=2)
        if len(pts) == 0:
            return None
        near_hand = np.linalg.norm(pts - obs.tcp.pos, axis=1) < HAND_EXCLUSION_M
        near_place = np.linalg.norm(pts[:, :2] - np.asarray(xy), axis=1) < 0.03
        # After a step aside, returns well above the old top are the gripper body or wrist
        # reaching over the spot, outside the fingertip exclusion, so they are dropped.
        # Not applied straight after the lift: the held object is still in its column,
        # and truncating it would read as a false drop.
        below_old_top = (
            pts[:, 2] <= before_top + COLUMN_HEADROOM_M
            if hand_left
            else np.ones(len(pts), dtype=bool)
        )
        pool = pts[near_place & ~near_hand & below_old_top]
        if len(pool) < 5:
            # Missing depth or occlusion is not evidence that the object moved.
            return None
        return float(before_top - float(np.percentile(pool[:, 2], 95)))

    def place(self, target_id: str, relation: str = "in") -> PrimitiveResult:
        """Carry what is held over the target and release above it."""
        held = self.memory.held
        if held is None:
            return PrimitiveResult(
                "place",
                Outcome.NOT_HOLDING,
                target_id,
                reason="you are not holding anything; grasp something first",
            )
        catalog = self.catalog_or_refresh()
        target = catalog.get(target_id)
        remembered = False
        if target is None:
            target = self._remembered_target(target_id, held)
            if target is None:
                return self._unknown("place", target_id)
            remembered = True

        if relation in BESIDE:
            spot = self._beside_pose(target, relation, held)
            if isinstance(spot, str):
                return PrimitiveResult("place", Outcome.REJECTED, target_id, reason=spot)
            release_z, centre = spot
        else:
            release_z, centre = self._release_pose(target, relation)
        start = self.emb.observe().tcp
        carry_z = self._carry_height(start.pos, centre, release_z, held, target_id)
        self.log(
            "plan",
            plan="place",
            held=held,
            target_id=target_id,
            relation=relation,
            release_at=[
                round(float(centre[0]), 3),
                round(float(centre[1]), 3),
                round(float(release_z), 3),
            ],
            carry_height_cm=round(carry_z * 100, 1),
            overhang_cm=None
            if self.held_overhang_m is None
            else round(self.held_overhang_m * 100, 1),
        )
        # Rise vertically, carry at a height that clears everything on the path, then
        # lower, so the held object does not swing through neighbouring rims.
        rise = Pose(np.array([start.pos[0], start.pos[1], carry_z]), start.quat)
        carry = Pose(np.array([centre[0], centre[1], carry_z]), start.quat)
        try:
            self._go(
                rise,
                GripperCommand(GripperAction.HOLD),
                meta={"phase": "rise"},
                tolerance_m=WAYPOINT_TOLERANCE_M,
                speed_mps=LIFT_SPEED_MPS,
            )
            dropped = self._dropped_in_transit(held, target_id, "lifting it")
            if dropped is not None:
                return dropped
            self._go(
                carry,
                GripperCommand(GripperAction.HOLD),
                meta={"phase": "carry"},
                tolerance_m=WAYPOINT_TOLERANCE_M,
            )
        except MotionIncomplete as exc:
            return PrimitiveResult(
                "place",
                Outcome.BLOCKED,
                target_id,
                reason=f"could not carry to above {target_id}: {exc}",
            )

        dropped = self._dropped_in_transit(held, target_id, "carrying it")
        if dropped is not None:
            return dropped

        down = Pose(np.array([centre[0], centre[1], release_z]), self.emb.observe().tcp.quat)
        note = None
        try:
            self._go(
                down,
                GripperCommand(GripperAction.HOLD),
                meta={"phase": "lower"},
                tolerance_m=GRASP_TOLERANCE_M,
                speed_mps=CONTACT_SPEED_MPS,
            )
        except MotionIncomplete as exc:
            high = float(exc.report.final_tcp.pos[2] - release_z)
            if high > LOWER_RELEASE_ANYWAY_M:
                self._retreat(carry, phase="retreat_after_block")
                return PrimitiveResult(
                    "place",
                    Outcome.BLOCKED,
                    target_id,
                    reason=(
                        f"lowering stopped {high * 100:.1f} cm above the release height: "
                        f"{exc}. Still holding {held}; the hand has moved back up"
                    ),
                )
            note = f"released {high * 100:.1f} cm higher than planned because lowering stopped"
        dropped = self._dropped_in_transit(held, target_id, "lowering it")
        if dropped is not None:
            return dropped
        self._go(
            self.emb.observe().tcp, GripperCommand(GripperAction.OPEN), meta={"phase": "release"}
        )
        back_up = Pose(
            np.array([centre[0], centre[1], CARRY_HEIGHT_M]), self.emb.observe().tcp.quat
        )
        self._go_best_effort(back_up, meta={"phase": "retreat"})

        # Relocate in memory before refreshing so the object keeps its id.
        landed = np.array([centre[0], centre[1], release_z])
        self.memory.relocate(held, landed)
        self.memory.set_held(None)
        self.held_overhang_m = None
        catalog = self.refresh()
        placed = catalog.get(held)
        detail = {"target": target_id, "release_z": release_z}
        if remembered:
            detail["target_from_memory"] = True
            note = ((note + "; ") if note else "") + (
                f"{target_id} was not in the latest look, so its last measured position was used"
            )
        if placed is not None:
            over = float(np.linalg.norm(placed.centroid[:2] - target.centroid[:2]))
            detail["offset_m"] = over
            self.memory.record(
                "place", held, Outcome.PLACED.value, f"{over * 100:.0f} cm from centre"
            )
            return PrimitiveResult(
                "place",
                Outcome.PLACED,
                target_id,
                reason=f"released {held} {over * 100:.0f} cm from the centre of {target_id}"
                + (f" ({note})" if note else ""),
                detail=detail,
            )
        self.memory.record("place", held, Outcome.PLACED.value, "released, now out of view")
        return PrimitiveResult(
            "place",
            Outcome.PLACED,
            target_id,
            reason=f"released {held} over {target_id}; it is no longer separately visible"
            + (f" ({note})" if note else ""),
            detail=detail,
        )

    def _carry_height(
        self, start_pos, centre, release_z: float, held: str, target_id: str
    ) -> float:
        """Fingertip carry height that keeps the held object clear of everything along the
        straight path to the target, including the target's rim."""
        overhang = self.held_overhang_m if self.held_overhang_m is not None else 0.10
        z = max(CARRY_HEIGHT_M, release_z + 0.10)
        others = self._neighbour_points(held)
        if len(others):
            a = np.asarray(start_pos, dtype=float)[:2]
            b = np.asarray(centre, dtype=float)[:2]
            ab = b - a
            t = np.clip(((others[:, :2] - a) @ ab) / max(1e-9, float(ab @ ab)), 0.0, 1.0)
            near = np.linalg.norm(others[:, :2] - (a + t[:, None] * ab), axis=1) < 0.10
            if near.any():
                z = max(z, float(np.max(others[near, 2])) + overhang + CARRY_CLEARANCE_M)
        top = float(self.emb.capabilities.workspace.max_xyz[2]) - 0.02
        return float(min(z, top))

    def _dropped_in_transit(self, held: str, target_id: str, doing: str) -> PrimitiveResult | None:
        """Return a DROPPED result if the jaw has closed past the hold threshold, else None.

        The jaw cannot be occluded, so it is checked at each transport phase to avoid
        reporting a placement for an object lost in transit.
        """
        jaw = float(self.emb.observe().gripper.travel)
        if not self.jaw_informative:
            self.log(
                "check",
                check=f"jaw while {doing}",
                held=held,
                jaw=round(jaw, 3),
                verdict="cannot tell: the hold is on a wall too thin for the jaw to read",
            )
            return None
        self.log(
            "check",
            check=f"jaw while {doing}",
            held=held,
            jaw=round(jaw, 3),
            verdict="still holding" if jaw <= self.hold_threshold else "fell out",
        )
        if jaw <= self.hold_threshold:
            return None
        self.memory.set_held(None)
        self.held_overhang_m = None
        self.unverified_hold = False
        self.refresh()
        self.memory.record("place", held, Outcome.DROPPED.value, f"fell out while {doing}")
        return PrimitiveResult(
            "place",
            Outcome.DROPPED,
            target_id,
            reason=(
                f"{held} fell out of the hand while {doing}: the jaw closed fully. It is "
                "somewhere below the path from where it was picked up, probably on the "
                "table; look before grasping it again, and prefer a deeper grasp"
            ),
            detail={"held": held, "phase": doing},
        )

    def move_to(self, pose: Pose, note: str = "") -> PrimitiveResult:
        report = self._go(pose, GripperCommand(GripperAction.HOLD), meta={"phase": "move_to"})
        outcome = {
            ExecutionOutcome.REACHED: Outcome.MOVED,
            ExecutionOutcome.CLAMPED: Outcome.MOVED,
            ExecutionOutcome.EXPIRED: Outcome.MOVED,
            ExecutionOutcome.BLOCKED: Outcome.BLOCKED,
        }.get(report.outcome, Outcome.MOVED)
        return PrimitiveResult(
            "move_to",
            outcome,
            None,
            reason=report.reason
            or (
                None
                if report.outcome is ExecutionOutcome.REACHED
                else f"{report.residual_m * 100:.1f} cm short on {report.residual_axis}"
            ),
            detail={"outcome": report.outcome.value, "residual_m": report.residual_m},
        )

    def release(self) -> PrimitiveResult:
        held = self.memory.held
        self._go(
            self.emb.observe().tcp, GripperCommand(GripperAction.OPEN), meta={"phase": "release"}
        )
        self.memory.set_held(None)
        self.unverified_hold = False
        self.refresh()
        return PrimitiveResult("release", Outcome.MOVED, held, reason="opened the gripper")

    # ---- internals --------------------------------------------------------

    def _beside_pose(
        self, target: ObjectDescriptor, relation: str, held: str
    ) -> tuple[float, np.ndarray] | str:
        """Free table spot beside the target and the release height, or a rejection reason.

        Base frame: left is +y, right is -y, in front of is toward the base, behind is away.
        The spot clears the target's extent in that direction by the held object's half size
        plus a margin, and moves outward if another object is in the way.
        """
        direction = BESIDE[relation]
        pts = target.points if target.points is not None and len(target.points) else None
        c = np.asarray(target.centroid[:2], dtype=float)
        reach = (
            float(np.max((pts[:, :2] - c) @ direction))
            if pts is not None
            else 0.5 * max(target.length, target.width)
        )
        tracked = self.memory.tracked.get(held)
        held_half = 0.5 * (tracked.size_m if tracked is not None and tracked.size_m else 0.06)
        others = self._neighbour_points(held)
        if len(others):
            others = others[np.linalg.norm(others[:, :2] - c, axis=1) > 0]
            if pts is not None:
                mask = np.ones(len(others), dtype=bool)
                tp = {tuple(np.round(q, 5)) for q in pts[::5]}
                mask = np.array([tuple(np.round(q, 5)) not in tp for q in others])
                others = others[mask]
        ws = self.emb.capabilities.workspace
        for extra in (0.0, 0.03, 0.06, 0.09):
            xy = c + direction * (reach + held_half + BESIDE_MARGIN_M + extra)
            if not ws.contains(np.array([xy[0], xy[1], ws.table_z + 0.1]), margin=0.02):
                continue
            if len(others) and float(np.min(np.linalg.norm(others[:, :2] - xy, axis=1))) < (
                held_half + 0.02
            ):
                continue
            overhang = self.held_overhang_m if self.held_overhang_m is not None else 0.05
            release_z = ws.table_z + overhang + BESIDE_DROP_M
            return float(release_z), np.array([xy[0], xy[1], ws.table_z])
        return (
            f"no free spot {relation.replace('_', ' ')} {target.object_id} inside the "
            "workspace: the table there is occupied or out of reach"
        )

    def _release_pose(self, target: ObjectDescriptor, relation: str) -> tuple[float, np.ndarray]:
        """Release height and xy centre, from the target's rim rather than its centroid."""
        pts = target.points if target.points is not None else np.zeros((0, 3))
        model = rim_model(pts)
        table_z = self.emb.capabilities.workspace.table_z
        if model is not None and relation == "in":
            centre, rim = model
            floor = support_height(pts, table_z)
            # Release when the held object's underside just clears the rim, so long objects
            # land inside rather than across the rim. With no measured overhang, fall back
            # to a fixed 10 cm above the floor to avoid releasing too low.
            overhang = self.held_overhang_m
            if overhang is None:
                return max(rim + PLACE_CLEARANCE_M, floor + 0.10), centre
            return max(rim + PLACE_CLEARANCE_M + overhang, floor + overhang + 0.02), centre
        centre = np.array([target.centroid[0], target.centroid[1], target.top_z])
        return target.top_z + PLACE_CLEARANCE_M, centre

    def _go(
        self,
        pose: Pose,
        gripper: GripperCommand,
        meta: dict | None = None,
        tolerance_m: float | None = None,
        speed_mps: float = FREE_SPEED_MPS,
    ):
        """Execute one motion inside a primitive; raise ``MotionIncomplete`` if it falls short.

        Traced with ``servo=True``: sub-steps of one decision do not count as decisions
        (see docs/protocol.md on budgets). ``tolerance_m`` overrides the controller's arrival
        tolerance, so waypoints do not spend time converging on the last millimetres.
        """
        caps = self.emb.capabilities
        # Timeout scales with distance; a flat budget expires long moves short of the
        # target. See EmbodimentCapabilities.time_to_converge.
        distance = float(np.linalg.norm(pose.pos - self.emb.observe().tcp.pos))
        target = TCPTarget(
            pose=pose,
            gripper=gripper,
            max_lin_speed_mps=min(speed_mps, caps.max_lin_speed_mps),
            max_ang_speed_rps=min(1.0, caps.max_ang_speed_rps),
            timeout_s=min(45.0, max(6.0, caps.time_to_converge(distance))),
            meta=meta or {},
        )
        report = self._play(target, meta)
        if report.outcome is ExecutionOutcome.REACHED:
            return report
        if (
            tolerance_m is not None
            and report.residual_m <= tolerance_m
            and report.outcome in (ExecutionOutcome.EXPIRED, ExecutionOutcome.BLOCKED)
        ):
            return replace(report, outcome=ExecutionOutcome.REACHED)
        raise MotionIncomplete(report)

    def _go_best_effort(
        self, pose: Pose, meta: dict | None = None, speed_mps: float = FREE_SPEED_MPS
    ):
        """Execute a motion and return its report without raising when it falls short."""
        try:
            return self._go(
                pose, GripperCommand(GripperAction.HOLD), meta=meta, speed_mps=speed_mps
            )
        except MotionIncomplete as exc:
            return exc.report

    def _retreat(self, pose: Pose, phase: str):
        try:
            self._go(
                pose,
                GripperCommand(GripperAction.HOLD),
                meta={"phase": phase},
                tolerance_m=WAYPOINT_TOLERANCE_M,
            )
        except MotionIncomplete:
            pass

    def _play(self, target: TCPTarget, meta: dict | None):
        """Issue a target, re-issuing while it expires but is still converging.

        An expiry with progress is an unfinished motion, not a block, and the policy cannot
        resume a motion inside a primitive. Expired approaches typically end with under
        1.5 cm to go. Retries cost simulated time only and stop after ``MOTION_RETRIES`` or
        when an attempt travels less than ``MOTION_PROGRESS_M``. Applies to both the
        episode-loop handler and direct execution.
        """
        attempts = 0
        while True:
            report = self._issue(target, meta)
            attempts += 1
            if report.outcome is ExecutionOutcome.REACHED or attempts >= MOTION_RETRIES:
                return report
            if report.outcome is not ExecutionOutcome.EXPIRED:
                return report
            # No progress means the arm is against a limit; retrying only spends time.
            if report.executed_lin_m < MOTION_PROGRESS_M:
                return report

    def _issue(self, target: TCPTarget, meta: dict | None):
        """Run one attempt through the loop's motion handler if set, else the embodiment."""
        if self.motion_handler is not None:
            return self.motion_handler(target, servo=True)
        if self.trace is not None:
            self.trace.emit(
                "attempt",
                self.emb.observe().t,
                self.episode,
                servo=True,
                target_id=target.id,
                pose=target.pose.pos,
                quat=target.pose.quat,
                gripper=target.gripper.action.value,
                phase=(meta or {}).get("phase"),
            )
        report = self.emb.execute(target)
        if self.trace is not None:
            self.trace.emit(
                "execution",
                report.t_end,
                self.episode,
                servo=True,
                target_id=report.target_id,
                outcome=report.outcome.value,
                residual_m=report.residual_m,
                residual_axis=report.residual_axis,
                reason=report.reason,
                phase=(meta or {}).get("phase"),
            )
        return report

    def emit_primitive(self, result: PrimitiveResult) -> PrimitiveResult:
        """Record a finished primitive. Called by the interface that invoked it."""
        if self.trace is not None:
            self.trace.emit(
                "primitive",
                self.emb.observe().t,
                self.episode,
                primitive=result.primitive,
                outcome=result.outcome.value,
                object_id=result.object_id,
                reason=result.reason,
                detail={k: v for k, v in result.detail.items() if k != "rendered"},
            )
        return result

    def _remembered_target(self, target_id: str, held: str) -> ObjectDescriptor | None:
        """Last measurement of a place target missing from the latest look.

        Destinations only: an object not grasped or moved since it was last seen is more
        likely a missed detection than gone. Grasp targets require a fresh sighting.
        """
        obj = self.last_seen.get(target_id)
        if obj is None or target_id == held or target_id in self.memory.just_moved:
            return None
        if any(
            a.object_id == target_id and a.primitive == "grasp" and a.result == "grasped"
            for a in self.memory.attempts
        ):
            return None
        self.log(
            "perception",
            note="place target not in the latest look; using its last measurement",
            object_id=target_id,
            missing_looks=self.memory.missing(target_id),
        )
        return obj

    def _unknown(self, primitive: str, object_id: str) -> PrimitiveResult:
        known = ", ".join(self.catalog_or_refresh().ids()) or "nothing"
        return PrimitiveResult(
            primitive,
            Outcome.REJECTED,
            object_id,
            reason=f"there is no object called {object_id} in this view; visible: {known}",
        )


def hover_over(obj: ObjectDescriptor, height_m: float = 0.12) -> Pose:
    return top_down_pose([obj.centroid[0], obj.centroid[1], obj.top_z + height_m])


def candidate_of(plan: GraspPlan, candidate_id: str | None) -> GraspCandidate | None:
    return plan.get(candidate_id) if candidate_id else plan.best()
