"""RoboDojo's dual ARX X5 rig (``dual_x5``, env_cfg ``arx_x5``) as a ``RigCodec``.

Facts from the RoboDojo checkout (``env_cfg/robot/dual_x5.yml``, ``env_cfg/arx_x5.yml``,
``env/observation_manager/obs_manager.py``, ``src/eval_client/eval_env.py``):

- Root frame is RoboDojo's end-effector frame: world minus the env origin, z up. The
  arm bases sit at ``(-0.3, -0.45, 0.765)`` (left) and ``(0.3, -0.45, 0.765)`` (right),
  both yawed +90 degrees, so the arms face ``+y`` and the robots' left is ``-x``.
- Per step the simulator sends ``obs["vision"][cam]`` with ``color`` (RGB uint8) and,
  when the env_cfg enables them, ``depth`` (float32 metres), ``intrinsic_matrix`` (3x3)
  and ``extrinsic_matrix`` (4x4 camera-to-world, from the camera prim, so USD axes);
  ``obs["state"]`` with ``{side}_ee_pose`` ``[x y z qw qx qy qz]`` and
  ``{side}_ee_joint_state``, the last *commanded* normalised gripper value.
- The action is a dict per step: ``{side}_ee_pose`` (7, same frame) and
  ``{side}_ee_joint_state`` (1, in [0, 1]); the simulator solves IK (cuRobo) and plays
  one action over ``1 / collect_freq`` = 0.04 s. A failed IK silently holds the arm,
  which Rung sees as a stalled motion.

Read from the downloaded Assets on the first live run (2026-10-27), not from the repo:
``ee_link`` is ``link6`` (``Assets/Robots/x5/robot_config.yml``); the fingers sit 0.087 m
ahead of it along ``+x`` (``X5A.urdf`` joint7/joint8) and their mesh reaches 0.071 m
further, so the fingertip midpoint is about 0.15 m along ``link6 +x``, jaws closing along
``y``. ``1`` on the gripper opens it (wrist images before and after a close).
"""

from __future__ import annotations

import math

import numpy as np

from rung.contracts.capabilities import (
    ArmSpec,
    CameraSpec,
    EmbodimentCapabilities,
    GripperGeometry,
    Workspace,
)
from rung.contracts.frames import Pose, convert_camera_pose, quat_from_axis_angle, rot_to_quat
from rung.contracts.observation import ArmState, CameraFrame, GripperState, Observation

ARMS = ("left", "right")
BASES = {"left": (-0.3, -0.45, 0.765), "right": (0.3, -0.45, 0.765)}
BASE_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)
CONTROL_HZ = 25.0
CAMERAS = {"scene": "cam_head", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
TABLE_Z_M = 0.765
"""Arm mount height; the head and wrist depth read the table top at 0.765 to 0.766 m.
Replaced by a depth measurement when depth is on."""
X5_TCP_FROM_EE = Pose([0.15, 0.0, 0.0], quat_from_axis_angle([0, 1, 0], math.pi / 2))
"""Fingertip midpoint in ``link6``: approach (Rung ``+z``) is ``link6 +x``, closing (``+y``)
is ``link6 y``, the same layout as the DROID flange. From the URDF and finger mesh, not
yet calibrated by touching a surface."""


def capabilities(
    table_z: float = TABLE_Z_M,
    tcp_from_ee: Pose | None = None,
    image_size=(640, 480),
    has_depth: bool = False,
) -> EmbodimentCapabilities:
    tcp_from_ee = tcp_from_ee or X5_TCP_FROM_EE
    gripper = GripperGeometry(
        max_opening_m=0.088, finger_length_m=0.05, pad_width_m=0.02, close_time_s=0.6
    )
    arms = []
    for side in ARMS:
        x, y, _ = BASES[side]
        arms.append(
            ArmSpec(
                side,
                Workspace(
                    min_xyz=np.array([x - 0.45, y + 0.05, table_z]),
                    max_xyz=np.array([x + 0.45, y + 0.65, table_z + 0.5]),
                    table_z=table_z,
                ),
                gripper,
                tcp_from_ee,
                Pose(BASES[side], BASE_QUAT),
                wrist_camera=f"{side}_wrist",
            )
        )
    return EmbodimentCapabilities(
        name="robodojo-dual-x5",
        cameras=(
            CameraSpec("scene", *image_size, has_depth=has_depth),
            *(
                CameraSpec(f"{s}_wrist", *image_size, has_depth=has_depth, mounted_on="tcp")
                for s in ARMS
            ),
        ),
        gripper=gripper,
        workspace=arms[0].workspace,
        control_period_s=1 / CONTROL_HZ,
        tcp_from_ee=tcp_from_ee,
        max_lin_speed_mps=0.30,
        max_ang_speed_rps=1.5,
        steps_per_efold=2.0,
        arms=tuple(arms),
        notes={
            "root_frame": "RoboDojo world frame at the env origin, z up: both arm bases "
            "are at y=-0.45 facing +y (across the table); the left arm is at "
            "x=-0.3 (the robots' left is -x), the right arm at x=+0.3",
            "forward_axis": "y",
            "action": "{side}_ee_pose + {side}_ee_joint_state dicts; simulator-side "
            "cuRobo IK, 0.04 s per action",
            "horizon": "step_lim counts actions, not seconds; run with time_mode=paused",
        },
    )


def _pose_from_matrix(m) -> Pose:
    m = np.asarray(m, dtype=float).reshape(4, 4)
    return Pose(m[:3, 3], rot_to_quat(m[:3, :3]))


class RoboDojoCodec:
    def __init__(
        self,
        table_z: float = TABLE_Z_M,
        tcp_from_ee: Pose | None = None,
        gripper_one_is_open: bool = True,
        camera_convention: str = "opengl",
        env_origin=(0.0, 0.0, 0.0),
    ):
        self.capabilities = capabilities(table_z, tcp_from_ee)
        self.gripper_one_is_open = gripper_one_is_open
        self.camera_convention = camera_convention
        self.env_origin = np.asarray(env_origin, float)
        self._ee_from_tcp = self.capabilities.tcp_from_ee.inverse()

    @staticmethod
    def unwrap(raw: dict) -> dict:
        """``TASK_ENV.get_obs()`` returns one env's dict; the batched form is keyed by index."""
        if "vision" not in raw and 0 in raw:
            return raw[0]
        return raw

    def _travel(self, value) -> float:
        v = float(np.clip(np.asarray(value, float).reshape(-1)[0], 0, 1))
        return 1.0 - v if self.gripper_one_is_open else v

    def _camera(self, role, data, t, state_at_capture=None) -> CameraFrame:
        K = data.get("intrinsic_matrix")
        extrinsic = data.get("extrinsic_matrix")
        pose = None
        if extrinsic is not None:
            world = _pose_from_matrix(extrinsic)
            pose = convert_camera_pose(
                Pose(world.pos - self.env_origin, world.quat), self.camera_convention
            )
        depth = data.get("depth")
        if depth is not None and (K is None or pose is None):
            depth = None  # validate_camera requires calibration alongside depth
        return CameraFrame(
            role,
            np.asarray(data["color"], dtype=np.uint8)[..., :3],
            t,
            depth=None if depth is None else np.asarray(depth, np.float32),
            K=None if K is None else np.asarray(K, float).reshape(3, 3),
            pose=pose,
            state_at_capture=state_at_capture,
        )

    def observation(self, raw: dict, t: float) -> Observation:
        raw = self.unwrap(raw)
        vision, state = raw["vision"], raw["state"]
        arms = {}
        for side in ARMS:
            ee = np.asarray(state[f"{side}_ee_pose"], float).reshape(7)
            tcp = Pose(ee[:3], ee[3:]) * self.capabilities.tcp_from_ee
            travel = self._travel(state.get(f"{side}_ee_joint_state", [1.0]))
            grip = GripperState(
                travel, opening_m=self.capabilities.gripper.opening_for_travel(travel)
            )
            wrist = vision.get(CAMERAS[f"{side}_wrist"])
            arms[side] = ArmState(
                tcp, grip, None if wrist is None else self._camera(f"{side}_wrist", wrist, t, tcp)
            )
        primary = arms[ARMS[0]]
        return Observation(
            t=t,
            scene=self._camera("scene", vision[CAMERAS["scene"]], t),
            tcp=primary.tcp,
            gripper=primary.gripper,
            wrist=primary.wrist,
            arms=arms,
            extra={"instruction": raw.get("instruction")},
        )

    def command(
        self, observation: Observation, setpoints: dict[str, Pose], grips: dict[str, float]
    ) -> dict:
        action = {}
        for side in ARMS:
            ee = setpoints[side] * self._ee_from_tcp
            action[f"{side}_ee_pose"] = np.concatenate([ee.pos, ee.quat]).astype(np.float32)
            g = float(grips[side])
            action[f"{side}_ee_joint_state"] = np.array(
                [1.0 - g if self.gripper_one_is_open else g], dtype=np.float32
            )
        return action
