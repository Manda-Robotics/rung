"""Reference policies: what the executor can and cannot do, without a model in the loop.

Two of them, and neither is a ceiling.

* ``ScriptedReference`` uses the embodiment's own object poses to run a sensible pick and
  place. Its score says what the primitives and the executor achieve when the perception
  and the object choice are correct. It is a hand-written sequence, so it is not optimal
  and must never be reported as an upper bound.
* ``RandomReference`` picks objects and actions at random under the same executor. Its
  score is what the executor alone achieves, which is the number a model policy has to
  beat before anything can be attributed to the model.

Both read simulator truth. That is allowed because they are diagnostics, not policies
under test, and it is why their results carry the ``scripted`` and ``random`` interface
labels wherever they appear. See docs/protocol.md, the diagnostic boundary rule.
"""

from __future__ import annotations

import numpy as np

from rung.contracts.action import GripperAction, GripperCommand, TCPTarget
from rung.contracts.capabilities import EmbodimentCapabilities
from rung.contracts.observation import Observation
from rung.mock.embodiment import MockEmbodiment
from rung.mock.world import top_down_pose
from rung.policy import Decision, Feedback, Verdict

HOVER_M = 0.12
LIFT_M = 0.20
RELEASE_ABOVE_RIM_M = 0.06


class ScriptedReference:
    """Hover, align, descend, close, lift, carry, release. One decision per phase."""

    interface = "scripted"
    name = "scripted-reference"

    def __init__(self, embodiment: MockEmbodiment):
        self.emb = embodiment
        self._phases: list[str] = []
        self._i = 0

    def reset(self, instruction: str, capabilities: EmbodimentCapabilities) -> None:
        self._phases = ["hover", "descend", "grasp", "lift", "carry", "release", "retreat"]
        self._i = 0

    def act(self, observation: Observation, feedback: Feedback) -> Decision:
        if self.emb.goal is None:
            return Decision(
                "give_up",
                "No goal is defined for this scene.",
                verdict=Verdict.GIVE_UP,
                reason="no goal",
            )
        obj_name, cont_name = self.emb.goal
        obj = self.emb.scene.body(obj_name)
        cont = self.emb.scene.body(cont_name)

        if self._i >= len(self._phases):
            return Decision(
                "done",
                "The object is in the container and the hand is clear.",
                verdict=Verdict.DONE,
            )

        phase = self._phases[self._i]
        self._i += 1
        speed = 0.25
        grip = GripperCommand(GripperAction.HOLD)
        yaw = obj.yaw

        if phase == "hover":
            pos = np.array([obj.pos[0], obj.pos[1], obj.top_z + HOVER_M])
            note = f"Hovering {HOVER_M * 100:.0f} cm above {obj_name}, jaw across its long axis."
        elif phase == "descend":
            pos = np.array([obj.pos[0], obj.pos[1], obj.top_z - 0.01])
            note = f"Descending onto {obj_name} to grasp height."
        elif phase == "grasp":
            pos = self.emb.robot.tcp.pos
            grip = GripperCommand(GripperAction.CLOSE)
            note = f"Closing on {obj_name}."
        elif phase == "lift":
            pos = np.array([obj.pos[0], obj.pos[1], obj.top_z + LIFT_M])
            note = "Lifting straight up."
        elif phase == "carry":
            pos = np.array([cont.pos[0], cont.pos[1], cont.top_z + LIFT_M])
            note = f"Carrying to a point above {cont_name}."
        elif phase == "release":
            pos = np.array([cont.pos[0], cont.pos[1], cont.top_z + RELEASE_ABOVE_RIM_M])
            grip = GripperCommand(GripperAction.OPEN)
            note = f"Opening just above the rim of {cont_name}."
        else:  # retreat
            pos = np.array([cont.pos[0], cont.pos[1], cont.top_z + LIFT_M])
            note = "Clearing the container so the hand is not in the way."
            yaw = 0.0

        target = TCPTarget(
            pose=top_down_pose(pos, yaw),
            gripper=grip,
            max_lin_speed_mps=speed,
            timeout_s=10.0,
            meta={"phase": phase, "policy": self.name},
        )
        return Decision(kind=phase, note=note, targets=[target], payload={"phase": phase})


class RandomReference:
    """Random reachable pose and a random gripper action, under the same executor."""

    interface = "random"
    name = "random-reference"

    def __init__(self, embodiment: MockEmbodiment, seed: int = 0):
        self.emb = embodiment
        self.rng = np.random.default_rng(seed)
        self._caps: EmbodimentCapabilities | None = None

    def reset(self, instruction: str, capabilities: EmbodimentCapabilities) -> None:
        self._caps = capabilities

    def act(self, observation: Observation, feedback: Feedback) -> Decision:
        ws = self._caps.workspace
        pos = np.array(
            [
                self.rng.uniform(ws.min_xyz[0] + 0.05, ws.max_xyz[0] - 0.05),
                self.rng.uniform(ws.min_xyz[1] + 0.05, ws.max_xyz[1] - 0.05),
                self.rng.uniform(ws.table_z + 0.02, ws.table_z + 0.30),
            ]
        )
        action = self.rng.choice([GripperAction.HOLD, GripperAction.CLOSE, GripperAction.OPEN])
        target = TCPTarget(
            pose=top_down_pose(pos, float(self.rng.uniform(-np.pi / 2, np.pi / 2))),
            gripper=GripperCommand(action),
            max_lin_speed_mps=0.25,
            timeout_s=10.0,
            meta={"policy": self.name},
        )
        return Decision(
            kind="random_move",
            note=f"Random pose {np.round(pos, 2).tolist()} with the gripper set to {action.value}.",
            targets=[target],
        )
