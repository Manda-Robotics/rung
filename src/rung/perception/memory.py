"""Object memory: stable identities, what is held, and what has been tried.

Identity is tracked in the harness rather than left to the prompt, since a stateless
policy otherwise switches objects between calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rung.perception.types import Catalog, ObjectDescriptor

MATCH_M = 0.06
"""Two detections in consecutive frames are the same object if their centroids are within
this distance. Wider than the noise, tighter than the gap between neighbouring objects."""

HELD_MATCH_M = 0.15
"""A held object is allowed to match further away: the harness knows it is in the hand, so
its position comes from the gripper rather than from proximity to where it used to be."""

CROSS_LABEL_M = 0.03
"""A detection this close to a tracked object's last position continues that object even
if the grounding label changed (e.g. "onion" to "red onion"). Otherwise the relabelled
object gets a new id and the policy sees a duplicate. Two distinct objects cannot both
stand within 3 cm of the same spot."""

HAND_SHADOW_M = 0.15
"""Hand distance within which an object's measured centre is not trusted for identity.

Reaching over an object is enough: the gripper occludes part of its cloud and the visible
centroid shifts, possibly past the match distance, which would give a stationary object a
new id. Wider than the gripper because its occlusion shadow in a depth view is wider.
"""


@dataclass
class Attempt:
    step: int
    primitive: str
    object_id: str
    result: str
    reason: str | None = None

    def render(self) -> str:
        line = f"step {self.step}: {self.primitive}({self.object_id}) -> {self.result}"
        return f"{line} ({self.reason})" if self.reason else line


@dataclass
class Tracked:
    object_id: str
    label: str
    centroid: np.ndarray
    last_seen_t: float
    last_seen_step: int
    missing_steps: int = 0
    size_m: float = 0.0
    """Longest horizontal side.

    A large object's measured centre moves under partial occlusion or when something lands
    inside it (about 10 cm for a crate), so its match tolerance scales with its size; see
    ``match_distance``.
    """

    @property
    def match_distance(self) -> float:
        """Half the object's own size, floored at ``MATCH_M``.

        A static object whose apparent centre moved further than its own radius is a
        different object, and two containers closer than that would overlap.
        """
        return max(MATCH_M, 0.5 * self.size_m)


@dataclass
class ObjectMemory:
    """Stable ids across frames, plus the episode's history.

    Ids come from the first frame an object is seen in and never change, so a policy can
    refer to ``banana_0`` twenty steps later and mean the same thing.
    """

    tracked: dict[str, Tracked] = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)
    held: str | None = None
    held_since_step: int | None = None
    just_moved: set = field(default_factory=set)
    """Objects the harness put somewhere this step. They get the wider match distance once,
    because a released object settles a few centimetres from where the fingers opened."""
    step: int = 0
    _counter: dict[str, int] = field(default_factory=dict)
    last_changes: list = field(default_factory=list)
    """What the last update did to identities: new ids, renames, far matches, objects not
    seen. Logged to the transcript, since an id change is invisible from the catalog."""

    def reset(self) -> None:
        self.tracked.clear()
        self.attempts.clear()
        self.held = None
        self.held_since_step = None
        self.step = 0
        self.just_moved.clear()
        self._counter.clear()

    def update(self, catalog: Catalog, tcp_pos=None) -> Catalog:
        """Rewrite the catalog's ids to stable ones and return it.

        Matching is greedy by distance within a label: sufficient at these scales and,
        unlike a Hungarian assignment with ties, deterministic.

        A held object moves with the hand, so its tracked position is first moved to the
        gripper; otherwise a freshly picked object reappears under a new id.
        """
        self.step += 1
        self.last_changes = []
        if self.held is not None and tcp_pos is not None and self.held in self.tracked:
            self.tracked[self.held].centroid = np.asarray(tcp_pos, dtype=np.float64).reshape(3)
        unmatched = list(self.tracked.values())
        for obj in catalog.objects:
            match = self._nearest(obj, unmatched, tcp_pos)
            if match is not None:
                moved = float(np.linalg.norm(match.centroid[:2] - obj.centroid[:2]))
                if match.label != obj.label:
                    self.last_changes.append(
                        {
                            "event": "renamed",
                            "id": match.object_id,
                            "was": match.label,
                            "now": obj.label,
                            "moved_m": moved,
                        }
                    )
                elif moved > MATCH_M:
                    self.last_changes.append(
                        {
                            "event": "matched far",
                            "id": match.object_id,
                            "label": obj.label,
                            "moved_m": moved,
                        }
                    )
                unmatched.remove(match)
                match.centroid = obj.centroid
                match.size_m = max(obj.length, obj.width)
                match.last_seen_t = catalog.t
                match.last_seen_step = self.step
                match.missing_steps = 0
                obj.object_id = match.object_id
            else:
                obj.object_id = self._new_id(obj.label)
                self.last_changes.append(
                    {"event": "new id", "id": obj.object_id, "label": obj.label}
                )
                self.tracked[obj.object_id] = Tracked(
                    object_id=obj.object_id,
                    label=obj.label,
                    centroid=obj.centroid,
                    last_seen_t=catalog.t,
                    last_seen_step=self.step,
                    size_m=max(obj.length, obj.width),
                )
        for stale in unmatched:
            stale.missing_steps += 1
            self.last_changes.append(
                {"event": "not seen", "id": stale.object_id, "steps": stale.missing_steps}
            )
        self.just_moved.clear()
        return catalog

    def _nearest(self, obj: ObjectDescriptor, pool: list[Tracked], tcp_pos=None) -> Tracked | None:
        """The tracked object this detection continues, or None if it is something new.

        The wide tolerance applies whenever the measured centroid is unreliable: the object
        is held, was just placed by the harness, or the hand is near enough to occlude part
        of its cluster (``HAND_SHADOW_M``).
        """
        best, best_d = None, 0.0
        for t in pool:
            if t.label != obj.label:
                # A label change comes from the grounding model, not the world. For an
                # object the harness just placed, its location is known, so the wide
                # tolerance applies under any label; otherwise it reappears as a duplicate.
                d = float(np.linalg.norm(t.centroid[:2] - obj.centroid[:2]))
                limit = HELD_MATCH_M if t.object_id in self.just_moved else CROSS_LABEL_M
                if d < limit and t.object_id != self.held and (best is None or d < best_d):
                    best, best_d = t, d
                continue
            hand_near = (
                tcp_pos is not None
                and float(np.linalg.norm(t.centroid[:2] - np.asarray(tcp_pos)[:2])) < HAND_SHADOW_M
            )
            wide = t.object_id == self.held or t.object_id in self.just_moved or hand_near
            limit = HELD_MATCH_M if wide else t.match_distance
            d = float(np.linalg.norm(t.centroid[:2] - obj.centroid[:2]))
            if d < limit and (best is None or d < best_d):
                best, best_d = t, d
        return best

    def _new_id(self, label: str) -> str:
        from rung.perception.service import _slug

        base = _slug(label)
        n = self._counter.get(base, 0)
        self._counter[base] = n + 1
        return f"{base}_{n}"

    def record(
        self, primitive: str, object_id: str, result: str, reason: str | None = None
    ) -> None:
        self.attempts.append(Attempt(self.step, primitive, object_id, result, reason))

    def relocate(self, object_id: str, centroid) -> None:
        """Tell the memory where an object has been put.

        Otherwise an object the harness moved beyond the match distance returns under a new
        id, and the policy concludes there is a second one.
        """
        tracked = self.tracked.get(object_id)
        if tracked is not None:
            tracked.centroid = np.asarray(centroid, dtype=np.float64).reshape(3)
            tracked.missing_steps = 0
            self.just_moved.add(object_id)

    def set_held(self, object_id: str | None) -> None:
        if object_id != self.held:
            self.held = object_id
            self.held_since_step = self.step if object_id is not None else None

    def missing(self, object_id: str) -> int:
        t = self.tracked.get(object_id)
        return t.missing_steps if t else 0

    def render(self, max_attempts: int = 6) -> str:
        """The memory block that goes into every prompt."""
        lines: list[str] = []
        if self.held:
            since = "" if self.held_since_step is None else f" since step {self.held_since_step}"
            lines.append(f"You are probably holding {self.held}{since}.")
        else:
            lines.append("You are not holding anything.")
        hidden = [t for t in self.tracked.values() if t.missing_steps > 0]
        if hidden:
            lines.append(
                "Not visible in this view: "
                + ", ".join(
                    f"{t.object_id} (last seen {t.missing_steps} steps ago)" for t in hidden
                )
            )
        if self.attempts:
            lines.append("What you have tried:")
            lines.extend(f"  {a.render()}" for a in self.attempts[-max_attempts:])
        return "\n".join(lines)
