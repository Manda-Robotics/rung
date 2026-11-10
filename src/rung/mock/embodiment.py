"""A mock embodiment: the tabletop world behind the real ``Embodiment`` contract.

Used by every test, by ``rung.probes``, and as the reference for what an adapter
has to provide. It validates its own observations on every step, so a change to the
contract fails here first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from rung.contracts.action import (
    ExecutionOutcome,
    ExecutionReport,
    GripperAction,
    TCPTarget,
)
from rung.contracts.capabilities import (
    CameraSpec,
    EmbodimentCapabilities,
    GripperGeometry,
    Workspace,
)
from rung.contracts.frames import Pose, quat_angle_between, quat_to_rot, rot_to_quat
from rung.contracts.observation import CameraFrame, GripperState, Observation
from rung.contracts.validate import validate_observation, validate_target
from rung.mock.world import (
    Body,
    Camera,
    GripperModel,
    RobotState,
    Scene,
    _extent_along,
    grasp_check,
    gripper_bodies,
    look_at,
    top_down_pose,
)


@dataclass
class MockConfig:
    scene_size: tuple[int, int] = (320, 240)
    wrist_size: tuple[int, int] = (192, 192)
    control_hz: float = 20.0
    with_wrist: bool = True
    home: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([0.42, 0.0, 0.32]))
    table_z: float = 0.0
    place_tolerance_m: float = 0.02
    wrist_calibration: bool = False
    """Publish the wrist camera's intrinsics and root pose (OpenCV axes) with each frame,
    as RoboLab's --wrist-calibration does. Off by default, so existing mock profiles keep
    an uncalibrated wrist view."""


class MockEmbodiment:
    """One arm, one table, a scene of boxes, and a grader.

    ``execute`` interpolates the TCP toward the target at the requested speed, one control
    period at a time, moving a held body with the hand, clamping at the table, and running
    the grasp check when the gripper closes. Simulated time advances by the control period.
    """

    def __init__(
        self,
        bodies: list[Body],
        instruction: str,
        goal: tuple[str, str] | None = None,
        config: MockConfig | None = None,
    ):
        self.cfg = config or MockConfig()
        self._initial = [
            Body(b.name, b.pos.copy(), b.size.copy(), b.yaw, b.color, b.graspable, b.container)
            for b in bodies
        ]
        self.scene = Scene([], table_z=self.cfg.table_z)
        self.instruction = instruction
        self.goal = goal  # (object_name, container_name)
        self.gripper_model = GripperModel()
        self.t = 0.0
        self.robot = RobotState(tcp=top_down_pose(self.cfg.home))
        self.events: list[dict] = []
        self.scene_cam = Camera(
            "scene",
            look_at(eye=(-0.25, -0.45, 0.55), target=(0.45, 0.0, 0.06)),
            self.cfg.scene_size[0],
            self.cfg.scene_size[1],
            fov_y_deg=55.0,
        )
        self.reset()

    # ---- contract surface -------------------------------------------------

    @property
    def capabilities(self) -> EmbodimentCapabilities:
        cams = [CameraSpec("scene", *self.cfg.scene_size, has_depth=True, mounted_on="world")]
        if self.cfg.with_wrist:
            cams.append(
                CameraSpec("wrist", *self.cfg.wrist_size, has_depth=False, mounted_on="tcp")
            )
        return EmbodimentCapabilities(
            name="mock-tabletop",
            cameras=tuple(cams),
            gripper=GripperGeometry(
                max_opening_m=self.gripper_model.max_opening_m,
                finger_length_m=self.gripper_model.finger_length_m,
                pad_width_m=self.gripper_model.pad_width_m,
                close_time_s=self.gripper_model.close_time_s,
            ),
            workspace=Workspace(
                min_xyz=np.array([0.15, -0.45, self.cfg.table_z]),
                max_xyz=np.array([0.75, 0.45, 0.60]),
                table_z=self.cfg.table_z,
            ),
            control_period_s=1.0 / self.cfg.control_hz,
            tcp_from_ee=Pose(np.array([0.0, 0.0, 0.1034]), np.array([1.0, 0.0, 0.0, 0.0])),
            has_joint_state=False,
            max_lin_speed_mps=0.35,
            max_ang_speed_rps=1.5,
            notes={"physics": "kinematic; no contact dynamics or toppling"},
        )

    def reset(self, episode_seed: int | None = None) -> Observation:
        rng = np.random.default_rng(episode_seed if episode_seed is not None else 0)
        self.scene.bodies = [
            Body(b.name, b.pos.copy(), b.size.copy(), b.yaw, b.color, b.graspable, b.container)
            for b in self._initial
        ]
        if episode_seed is not None:
            for b in self.scene.bodies:
                if b.graspable:
                    b.pos[:2] += rng.uniform(-0.03, 0.03, size=2)
                    b.yaw += float(rng.uniform(-0.4, 0.4))
        self.t = 0.0
        self.robot = RobotState(tcp=top_down_pose(self.cfg.home))
        self.events = []
        return self.observe()

    def observe(self) -> Observation:
        hand = gripper_bodies(self.robot.tcp, self.robot.travel, self.gripper_model)
        rgb, depth, _ = self.scene.render(self.scene_cam, extra=hand)
        scene = CameraFrame(
            name="scene",
            rgb=rgb,
            t=self.t,
            depth=depth,
            K=self.scene_cam.K,
            pose=self.scene_cam.pose,
            convention="opencv",
            state_at_capture=self.robot.tcp,
        )
        wrist = None
        if self.cfg.with_wrist:
            wrist_cam = self._wrist_camera()
            w_rgb, _, _ = self.scene.render(wrist_cam, extra=hand)
            wrist = CameraFrame(
                name="wrist",
                rgb=w_rgb,
                t=self.t,
                convention="opencv",
                state_at_capture=self.robot.tcp,
                **(
                    {"K": wrist_cam.K, "pose": wrist_cam.pose} if self.cfg.wrist_calibration else {}
                ),
            )
        obs = Observation(
            t=self.t,
            scene=scene,
            wrist=wrist,
            tcp=self.robot.tcp,
            gripper=GripperState(
                travel=self.robot.travel,
                opening_m=self.capabilities.gripper.opening_for_travel(self.robot.travel),
            ),
        )
        validate_observation(obs, self.capabilities)
        return obs

    def execute(self, target: TCPTarget) -> ExecutionReport:
        caps = self.capabilities
        validate_target(target, caps)
        dt = caps.control_period_s
        t_start = self.t
        start_pos = self.robot.tcp.pos.copy()
        start_quat = self.robot.tcp.quat.copy()
        goal_pos = np.asarray(target.pose.pos, dtype=np.float64).copy()
        clamped_to = None
        reason = None

        if not caps.workspace.contains(goal_pos):
            clipped = np.clip(goal_pos, caps.workspace.min_xyz, caps.workspace.max_xyz)
            reason = (
                f"target {np.round(goal_pos, 3).tolist()} outside the workspace; "
                f"clamped to {np.round(clipped, 3).tolist()}"
            )
            goal_pos = clipped
            clamped_to = Pose(goal_pos, target.pose.quat)

        floor = caps.workspace.table_z + 0.004
        if goal_pos[2] < floor:
            reason = (reason + "; " if reason else "") + "target below the table; clamped"
            goal_pos = goal_pos.copy()
            goal_pos[2] = floor
            clamped_to = Pose(goal_pos, target.pose.quat)

        lin_dist = float(np.linalg.norm(goal_pos - start_pos))
        rot_dist = quat_angle_between(start_quat, target.pose.quat)
        lin_speed = min(target.max_lin_speed_mps, caps.max_lin_speed_mps)
        ang_speed = min(target.max_ang_speed_rps, caps.max_ang_speed_rps)
        need_s = max(
            lin_dist / lin_speed if lin_speed > 0 else 0.0,
            rot_dist / ang_speed if ang_speed > 0 else 0.0,
        )
        steps_needed = int(np.ceil(need_s / dt)) if need_s > 0 else 0
        steps_allowed = int(np.floor(target.timeout_s / dt + 1e-9))
        steps = min(steps_needed, steps_allowed)

        blocked_by = None
        for i in range(steps):
            frac = (i + 1) / max(steps_needed, 1)
            new_pos = start_pos + (goal_pos - start_pos) * frac
            new_quat = _slerp(start_quat, target.pose.quat, frac)
            hit = self._blocking_body(new_pos)
            if hit is not None:
                blocked_by = hit
                break
            self._set_tcp(Pose(new_pos, new_quat))
            self.t += dt

        # Gripper actuation shares the target deadline and only follows a reached pose.
        grip_s = (
            self.gripper_model.close_time_s
            if target.gripper.action is GripperAction.CLOSE
            else self.gripper_model.close_time_s * 0.5
            if target.gripper.action is GripperAction.OPEN
            else 0.0
        )
        grip_expired = False
        dwell_expired = False
        if blocked_by is None and steps_needed <= steps_allowed:
            arrival_t = self.t
            remaining = target.timeout_s - (self.t - t_start)
            if grip_s > remaining + 1e-9:
                self.t += max(0, int(np.floor(remaining / dt + 1e-9))) * dt
                grip_expired = True
            elif target.gripper.action is GripperAction.CLOSE:
                self._close(target)
            elif target.gripper.action is GripperAction.OPEN:
                self._open(target)
            if not grip_expired:
                dwell_left = max(0, target.min_dwell_s - (self.t - arrival_t))
                remaining = max(0, target.timeout_s - (self.t - t_start))
                needed_ticks = int(np.ceil(dwell_left / dt - 1e-9))
                available_ticks = int(np.floor(remaining / dt + 1e-9))
                self.t += min(needed_ticks, available_ticks) * dt
                dwell_expired = needed_ticks > available_ticks

        actual = self.robot.tcp
        diff = np.abs(actual.pos - np.asarray(target.pose.pos))
        axis_i = int(np.argmax(diff))
        residual = float(diff[axis_i])
        rot_res = quat_angle_between(actual.quat, target.pose.quat)

        if blocked_by is not None:
            outcome = ExecutionOutcome.BLOCKED
            reason = (reason + "; " if reason else "") + f"blocked by {blocked_by}"
        elif clamped_to is not None:
            outcome = ExecutionOutcome.CLAMPED
        elif steps_needed > steps_allowed or grip_expired or dwell_expired:
            outcome = ExecutionOutcome.EXPIRED
            reason = (
                f"needed {steps_needed * dt:.1f} s at the requested speed, "
                f"timeout was {target.timeout_s:.1f} s; split the move"
            )
            if dwell_expired:
                reason = "the requested hold interval had not elapsed before the deadline"
        else:
            outcome = ExecutionOutcome.REACHED

        return ExecutionReport(
            target_id=target.id,
            outcome=outcome,
            t_start=t_start,
            t_end=self.t,
            final_tcp=actual,
            residual_m=residual,
            residual_axis="xyz"[axis_i],
            residual_rot_rad=rot_res,
            executed_lin_m=float(np.linalg.norm(actual.pos - start_pos)),
            executed_rot_rad=quat_angle_between(start_quat, actual.quat),
            clamped_to=clamped_to,
            reason=reason,
            gripper_travel_end=self.robot.travel,
        )

    def grade(self) -> tuple[bool, dict]:
        """Goal is ``(object, container)``: the object is inside the container's footprint,
        below its rim, and not in the hand."""
        if self.goal is None:
            return False, {"reason": "no goal defined"}
        obj_name, cont_name = self.goal
        obj, cont = self.scene.body(obj_name), self.scene.body(cont_name)
        in_xy = cont.contains_xy(obj.pos, margin=-0.005)
        below_rim = obj.bottom_z < cont.top_z - 0.002
        in_hand = self.robot.held == obj_name
        ok = bool(in_xy and below_rim and not in_hand)
        return ok, {
            "object": obj_name,
            "container": cont_name,
            "inside_footprint": bool(in_xy),
            "below_rim": bool(below_rim),
            "still_held": bool(in_hand),
            "object_pos": np.round(obj.pos, 4).tolist(),
        }

    # ---- internals --------------------------------------------------------

    def _wrist_camera(self) -> Camera:
        rot = quat_to_rot(self.robot.tcp.quat)
        # sits behind and above the fingertips, looking along the approach axis
        eye = self.robot.tcp.pos - rot @ np.array([0.0, 0.0, 0.10])
        pose = Pose(eye, rot_to_quat(rot))
        return Camera("wrist", pose, *self.cfg.wrist_size, fov_y_deg=70.0)

    def _set_tcp(self, pose: Pose) -> None:
        prev = self.robot.tcp
        self.robot.tcp = pose
        if self.robot.held is not None:
            b = self.scene.body(self.robot.held)
            rot = quat_to_rot(pose.quat)
            b.pos = pose.pos + rot @ self.robot.held_offset
            b.yaw += _yaw_delta(prev.quat, pose.quat)

    def _blocking_body(self, new_pos: npt.NDArray[np.float64]) -> str | None:
        """Name of the body the hand would collide with at ``new_pos``, or None.

        The TCP lies between the fingers, so it may enter an object's footprint; that is a
        grasp. Blocking occurs when the palm enters a body, or when the fingers straddle a
        body wider than the jaw opening. The palm is a point behind the fingertips along
        the approach axis, so the check holds for top-down and horizontal approaches.
        """
        rot = quat_to_rot(self.robot.tcp.quat)
        g = self.gripper_model
        palm = new_pos + rot @ np.array([0.0, 0.0, -(g.finger_length_m + 0.02)])
        closing = rot[:, 1]
        for b in self.scene.bodies:
            if b.name == self.robot.held or b.container:
                continue
            if b.contains_xy(palm, margin=-0.004) and b.bottom_z <= palm[2] <= b.top_z:
                return b.name
            if not b.contains_xy(new_pos, margin=-0.004):
                continue
            if b.bottom_z <= new_pos[2] <= b.top_z and _extent_along(b, closing) > g.max_opening_m:
                return b.name
        return None

    def _close(self, target: TCPTarget) -> None:
        self.t += self.gripper_model.close_time_s
        candidates = [b for b in self.scene.bodies if b.graspable and b.name != self.robot.held]
        for b in candidates:
            holds, why = grasp_check(self.robot.tcp, 1.0, b, self.gripper_model)
            if holds:
                rot = quat_to_rot(self.robot.tcp.quat)
                self.robot.held = b.name
                self.robot.held_offset = rot.T @ (b.pos - self.robot.tcp.pos)
                width = _closing_width(b, rot)
                self.robot.travel = float(
                    np.clip(1.0 - width / self.gripper_model.max_opening_m, 0.05, 0.95)
                )
                self.events.append({"t": self.t, "event": "grasped", "object": b.name})
                return
        self.robot.travel = 1.0  # closed on nothing
        self.events.append({"t": self.t, "event": "closed_empty"})

    def _open(self, target: TCPTarget) -> None:
        self.t += self.gripper_model.close_time_s * 0.5
        released = self.robot.held
        self.robot.travel = 0.0
        if released is not None:
            b = self.scene.body(released)
            self.robot.held = None
            self.robot.held_offset = None
            self._drop(b)
            self.events.append({"t": self.t, "event": "released", "object": b.name})

    def _drop(self, b: Body) -> None:
        """Fall straight down onto whatever is under it: a container floor or the table."""
        support_z = self.cfg.table_z
        for other in self.scene.bodies:
            if other.name == b.name:
                continue
            if other.contains_xy(b.pos, margin=-0.005):
                if other.container:
                    support_z = max(support_z, other.bottom_z + other.wall_thickness_m)
                else:
                    support_z = max(support_z, other.top_z)
        b.pos[2] = support_z + b.size[2] / 2


def _closing_width(b: Body, rot: npt.NDArray[np.float64]) -> float:
    return _extent_along(b, rot[:, 1])


def _yaw_delta(q_from: npt.ArrayLike, q_to: npt.ArrayLike) -> float:
    from rung.contracts.frames import yaw_of

    d = yaw_of(q_to) - yaw_of(q_from)
    return float((d + np.pi) % (2 * np.pi) - np.pi)


def _slerp(q0: npt.ArrayLike, q1: npt.ArrayLike, frac: float) -> npt.NDArray[np.float64]:
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        out = a + frac * (b - a)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(theta)
    return (np.sin((1 - frac) * theta) / s) * a + (np.sin(frac * theta) / s) * b


# ---- ready-made scenes ----------------------------------------------------


def banana_in_bowl(config: MockConfig | None = None) -> MockEmbodiment:
    """A single elongated object and an open container: the baseline pick and place."""
    bodies = [
        Body(
            "banana",
            np.array([0.42, -0.13, 0.018]),
            np.array([0.17, 0.045, 0.036]),
            yaw=0.35,
            color=(214, 196, 74),
        ),
        Body(
            "bowl",
            np.array([0.48, 0.16, 0.035]),
            np.array([0.20, 0.20, 0.070]),
            color=(150, 158, 172),
            graspable=False,
            container=True,
        ),
    ]
    return MockEmbodiment(
        bodies,
        "Pick up the banana and place it in the bowl",
        goal=("banana", "bowl"),
        config=config,
    )


def tall_bottle_and_crate(config: MockConfig | None = None) -> MockEmbodiment:
    """A body too tall to pinch from above at its centre, and a wide one with a narrow neck.

    A hard geometry case for the analytic grasp planner; failures here are expected until
    the planner handles it.
    """
    bodies = [
        Body(
            "bottle",
            np.array([0.40, -0.10, 0.11]),
            np.array([0.07, 0.07, 0.22]),
            color=(90, 140, 96),
        ),
        Body(
            "box", np.array([0.52, -0.20, 0.03]), np.array([0.14, 0.11, 0.06]), color=(196, 120, 84)
        ),
        Body(
            "crate",
            np.array([0.50, 0.18, 0.05]),
            np.array([0.26, 0.22, 0.10]),
            color=(140, 128, 110),
            graspable=False,
            container=True,
        ),
    ]
    return MockEmbodiment(
        bodies, "Put the bottle in the crate", goal=("bottle", "crate"), config=config
    )


def cluttered_bin(config: MockConfig | None = None) -> MockEmbodiment:
    """Several similar objects near each other: the flood-fill and disambiguation case."""
    bodies = [
        Body(
            "apple",
            np.array([0.38, -0.06, 0.032]),
            np.array([0.065, 0.065, 0.064]),
            color=(198, 74, 66),
        ),
        Body(
            "lime",
            np.array([0.44, -0.12, 0.026]),
            np.array([0.052, 0.052, 0.052]),
            color=(126, 178, 72),
        ),
        Body(
            "mug",
            np.array([0.50, -0.04, 0.045]),
            np.array([0.085, 0.075, 0.090]),
            color=(232, 228, 220),
        ),
        Body(
            "bin",
            np.array([0.52, 0.20, 0.055]),
            np.array([0.28, 0.24, 0.110]),
            color=(120, 124, 132),
            graspable=False,
            container=True,
        ),
    ]
    return MockEmbodiment(
        bodies, "Put the apple in the grey bin", goal=("apple", "bin"), config=config
    )


SCENES = {
    "banana_in_bowl": banana_in_bowl,
    "tall_bottle_and_crate": tall_bottle_and_crate,
    "cluttered_bin": cluttered_bin,
}
