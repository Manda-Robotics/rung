"""A CPU stand-in for RoboDojo's simulator side, with the real observation and action dicts.

``FakeRoboDojo`` produces what ``TASK_ENV.get_obs()`` returns for the dual ARX X5 rig
(``vision`` per camera, ``state`` per arm, ``instruction``) and consumes what Rung's codec
sends back (``{side}_ee_pose`` and ``{side}_ee_joint_state`` per arm). It tracks each
commanded end-effector pose with a first-order lag, the way the simulator's IK plays a
0.04 s action, so the bridge's setpoints, receipts and stall detection run for real.
A ``blocked`` arm ignores its commands, which is how a failed simulator-side IK looks.

``run_rpc_episode`` drives a ``RungRoboDojoModel`` through RoboDojo's own call sequence
(``reset``, then ``update_obs`` and ``get_action`` alternately), as
``deploy/robodojo/policy/deploy.py`` does inside the simulator. Nothing is graded: the
fake has no objects, so it tests plumbing and geometry, not manipulation.
"""

from __future__ import annotations

import numpy as np

from rung.adapters.robodojo.rig import ARMS, BASES, CAMERAS

HEAD_EXTRINSIC = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -0.41], [0.0, 0.0, 1.0, 1.308], [0.0, 0.0, 0.0, 1.0]]
)
"""Head camera-to-world in USD axes: 1.308 m up, looking straight down (USD cameras look
along their ``-z``), so its depth reads the 0.765 m table top everywhere."""


class FakeRoboDojo:
    def __init__(
        self,
        instruction: str = "pick up the object on the table",
        gripper_one_is_open: bool = True,
        blocked: set[str] | None = None,
        lag: float = 0.6,
        image_size: tuple[int, int] = (640, 480),
        geometry: bool = False,
    ):
        self.instruction = instruction
        self.gripper_one_is_open = gripper_one_is_open
        self.blocked = set(blocked or ())
        self.lag = lag
        self.width, self.height = image_size
        self.geometry = geometry
        self.steps = 0
        self.ee = {s: np.array([BASES[s][0], -0.2, 0.95, 0.0, 1.0, 0.0, 0.0]) for s in ARMS}
        self.grip = {s: 1.0 if gripper_one_is_open else 0.0 for s in ARMS}

    def _camera(self, extrinsic=None) -> dict:
        data = {"color": np.zeros((self.height, self.width, 3), np.uint8)}
        if self.geometry and extrinsic is not None:
            f = 0.8 * self.width
            data["intrinsic_matrix"] = np.array(
                [[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1.0]]
            )
            data["extrinsic_matrix"] = extrinsic
            data["depth"] = np.full((self.height, self.width), 1.308 - 0.765, np.float32)
        return data

    def get_obs(self) -> dict:
        vision = {CAMERAS["scene"]: self._camera(HEAD_EXTRINSIC)}
        for s in ARMS:
            vision[CAMERAS[f"{s}_wrist"]] = self._camera()
        state = {f"{s}_ee_pose": self.ee[s].copy() for s in ARMS}
        state |= {f"{s}_ee_joint_state": np.array([self.grip[s]]) for s in ARMS}
        return {"vision": vision, "state": state, "instruction": self.instruction}

    def take_action(self, action: dict) -> None:
        self.steps += 1
        for s in ARMS:
            if s in self.blocked:
                continue
            goal = np.asarray(action[f"{s}_ee_pose"], float).reshape(7)
            self.ee[s][:3] += self.lag * (goal[:3] - self.ee[s][:3])
            self.ee[s][3:] = goal[3:]
            self.grip[s] = float(np.asarray(action[f"{s}_ee_joint_state"]).reshape(-1)[0])


def run_rpc_episode(model, sim: FakeRoboDojo, step_lim: int = 2000) -> int:
    """One episode in RoboDojo's call order; returns the number of actions played.

    Stops at ``step_lim`` actions, as RoboDojo does, or when the harness episode ends.
    The final ``reset`` closes the harness episode, as the next episode's reset would.
    """
    model.reset()
    played = 0
    while played < step_lim:
        model.update_obs(sim.get_obs())
        for action in model.get_action():
            sim.take_action(action)
            played += 1
        if model.session is not None and not model.session[2].is_alive():
            break
    model.reset()
    return played
