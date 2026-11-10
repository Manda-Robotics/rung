"""The RoboLab adapter, tested without Isaac.

Every failure this catches is one that would otherwise show up as a bad model score: a
camera convention left unconverted, a flange mistaken for the fingertips, motions half as
long as commanded. Fake observations are built in the shapes RoboLab actually produces,
including the torch-like ``.cpu().numpy()`` access the real ones need.
"""

from __future__ import annotations

import numpy as np
import pytest

from rung.adapters.robolab.bridge import RoboLabEmbodiment, RungInferenceClient, run_in_thread
from rung.adapters.robolab.translate import (
    FLANGE_TO_FINGERTIP_M,
    REL_IK_SCALE,
    SCENE_CAMERA,
    action_chunk,
    droid_capabilities,
    flange_to_tcp,
    steps_needed,
    tcp_to_flange,
    to_observation,
)
from rung.contracts.action import ExecutionOutcome, GripperAction, GripperCommand, TCPTarget
from rung.contracts.frames import Pose, deproject, quat_from_axis_angle, quat_to_rot, rot_to_quat
from rung.contracts.validate import validate_observation
from rung.grasp.analytic import top_down_pose


class FakeTensor:
    """Stands in for a torch tensor: batched, with the access pattern RoboLab uses."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __array__(self, dtype=None):
        return self._array if dtype is None else self._array.astype(dtype)

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def shape(self):
        return self._array.shape

    def __getitem__(self, index):
        return FakeTensor(self._array[index])


def opengl_camera_looking_down(height: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """A camera above the table looking straight down, in RoboLab's OpenGL convention.

    OpenGL looks along local -z, so an identity orientation already points at the floor.
    """
    return np.array([0.45, 0.0, height]), np.array([1.0, 0.0, 0.0, 0.0])


def fake_obs(
    envs: int = 2,
    env_id: int = 0,
    flange=(0.45, 0.0, 0.35),
    flange_quat=None,
    travel: float = 0.0,
    depth_value: float = 0.55,
    size: tuple[int, int] = (48, 64),
    with_wrist: bool = True,
    with_depth: bool = True,
) -> dict:
    height, width = size
    cam_pos, cam_quat = opengl_camera_looking_down()
    f = width / 2
    intrinsics = np.array([[f, 0, width / 2 - 0.5], [0, f, height / 2 - 0.5], [0, 0, 1.0]])
    quat = np.asarray(flange_quat if flange_quat is not None else [1.0, 0.0, 0.0, 0.0])

    def batch(value, dtype=np.float64):
        return FakeTensor(np.repeat(np.asarray(value, dtype=dtype)[None], envs, axis=0))

    images = {
        SCENE_CAMERA: batch(np.full((height, width, 3), 120, dtype=np.uint8), np.uint8),
    }
    if with_depth:
        images.update(
            {
                f"{SCENE_CAMERA}_depth": batch(
                    np.full((height, width), depth_value, np.float32), np.float32
                ),
                f"{SCENE_CAMERA}_K": batch(intrinsics),
                f"{SCENE_CAMERA}_pos": batch(cam_pos),
                f"{SCENE_CAMERA}_quat": batch(cam_quat),
            }
        )
    if with_wrist:
        images["wrist_cam"] = batch(np.full((32, 32, 3), 80, dtype=np.uint8), np.uint8)
    proprio = {
        "eef_pos": batch(np.asarray(flange)),
        "eef_quat": batch(rot_to_quat(quat_to_rot(quat) @ quat_to_rot([0.5, -0.5, 0.5, -0.5]))),
        "ee_pos": batch(np.asarray(flange)),
        "ee_quat": batch(quat),
        "gripper_pos": batch(np.asarray([travel])),
    }
    return {"image_obs": images, "proprio_obs": proprio}


# ---- observation ----------------------------------------------------------


def test_observation_passes_the_contract():
    obs = to_observation(fake_obs(), t=1.5)
    validate_observation(obs, droid_capabilities(scene_size=(64, 48), wrist_size=(32, 32)))
    assert obs.scene.name == "scene" and obs.wrist.name == "wrist"
    assert obs.t == 1.5
    assert obs.scene.rgb.dtype == np.uint8
    assert obs.scene.depth.dtype == np.float32
    assert obs.wrist is not None


def test_the_tcp_is_the_fingertips_not_the_flange():
    """RoboLab reports the flange. Everything the harness measures is about the fingers."""
    obs = to_observation(fake_obs(flange=(0.45, 0.0, 0.35)), t=0.0)
    assert obs.tcp.pos[0] == pytest.approx(0.45 + FLANGE_TO_FINGERTIP_M)
    assert obs.tcp.pos[2] == pytest.approx(0.35)


def test_top_down_target_aligns_the_physical_mesh_and_jaw_axes():
    # Independent reference from the actual DROID USD geometry: finger extents are
    # 0.093..0.150 m on base_link X, separated along Y. Base Z is pad thickness.
    # A pose round-trip alone cannot catch an incorrect but self-consistent tool frame.
    target = top_down_pose([0.5, 0, 0.02], yaw=0)
    flange = tcp_to_flange(target)
    physical_approach = quat_to_rot(flange.quat) @ [1, 0, 0]
    physical_closing = quat_to_rot(flange.quat) @ [0, 1, 0]
    assert np.allclose(physical_approach, [0, 0, -1], atol=1e-10)
    assert np.allclose(abs(physical_closing), [0, 1, 0], atol=1e-10)
    assert flange.pos[2] == pytest.approx(target.pos[2] + FLANGE_TO_FINGERTIP_M)


def test_the_fingertip_offset_follows_the_gripper_orientation():
    tilted = quat_from_axis_angle([0, 1, 0], np.pi / 2)  # base_link +x points down
    obs = to_observation(fake_obs(flange=(0.4, 0.0, 0.3), flange_quat=tilted), t=0.0)
    assert obs.tcp.pos[0] == pytest.approx(0.4, abs=1e-9)
    assert obs.tcp.pos[2] == pytest.approx(0.3 - FLANGE_TO_FINGERTIP_M, abs=1e-9)


def test_flange_and_tcp_conversions_are_inverses():
    pose = Pose(np.array([0.4, -0.1, 0.3]), quat_from_axis_angle([0.3, 1.0, 0.2], 0.7))
    flange = tcp_to_flange(pose)
    back = flange_to_tcp(flange.pos, flange.quat)
    assert np.allclose(back.pos, pose.pos, atol=1e-12)


def test_the_camera_convention_is_converted():
    """RoboLab cameras look down local -z with +y up. Left unconverted, every deprojected
    point lands behind the camera, which is the failure the probes exist to catch."""
    obs = to_observation(fake_obs(), t=0.0)
    forward = quat_to_rot(obs.scene.pose.quat) @ np.array([0.0, 0.0, 1.0])
    assert forward[2] < -0.99, forward  # OpenCV +z now points at the table


def test_deprojected_depth_lands_on_the_table():
    """The end-to-end geometry check: a downward camera 90 cm up, depth 55 cm, gives a
    point 35 cm above the floor, and the image centre lands under the camera."""
    obs = to_observation(fake_obs(depth_value=0.55), t=0.0)
    centre = np.array([obs.scene.width / 2 - 0.5, obs.scene.height / 2 - 0.5])
    point = deproject(centre, 0.55, obs.scene.K, obs.scene.pose)
    assert point[2] == pytest.approx(0.35, abs=1e-6)
    assert np.allclose(point[:2], [0.45, 0.0], atol=1e-6)


def test_env_id_selects_the_right_environment():
    raw = fake_obs(envs=3)
    poses = np.stack([[0.3, 0, 0.2], [0.5, 0, 0.4], [0.7, 0, 0.6]])
    raw["proprio_obs"]["ee_pos"] = FakeTensor(poses)
    for env_id, expected in enumerate([0.3, 0.5, 0.7]):
        obs = to_observation(raw, t=0.0, env_id=env_id)
        assert obs.tcp.pos[0] == pytest.approx(expected + FLANGE_TO_FINGERTIP_M)


def test_float_and_rgba_images_are_normalised():
    raw = fake_obs()
    raw["image_obs"][SCENE_CAMERA] = FakeTensor(np.full((2, 48, 64, 4), 0.5, dtype=np.float32))
    obs = to_observation(raw, t=0.0)
    assert obs.scene.rgb.shape == (48, 64, 3)
    assert obs.scene.rgb.dtype == np.uint8
    assert 120 <= int(obs.scene.rgb[0, 0, 0]) <= 135


def test_gripper_travel_is_read_and_clipped():
    assert to_observation(fake_obs(travel=0.4), t=0.0).gripper.travel == pytest.approx(0.4)
    assert to_observation(fake_obs(travel=1.7), t=0.0).gripper.travel == 1.0


def test_depth_without_calibration_is_an_error_not_a_silent_none():
    """Depth without intrinsics or camera pose raises instead of passing ``None`` through."""
    raw = fake_obs()
    del raw["image_obs"][f"{SCENE_CAMERA}_K"]
    with pytest.raises(KeyError, match="depth but not all of"):
        to_observation(raw, t=0.0)


def test_a_missing_camera_names_what_is_available():
    with pytest.raises(KeyError, match="available"):
        to_observation(fake_obs(), t=0.0, camera="nonexistent_camera")


def test_a_non_robolab_observation_is_rejected():
    with pytest.raises(KeyError, match="not from RoboLab"):
        to_observation({"policy": {"scene_rgb": np.zeros((1, 4, 4, 3))}}, t=0.0)


def test_rgb_only_rig_still_works():
    obs = to_observation(fake_obs(with_depth=False), t=0.0)
    assert obs.scene.depth is None and obs.scene.K is None


# ---- actions --------------------------------------------------------------


def test_a_chunk_undoes_the_controller_scale():
    """DroidRelIKActionCfg multiplies by 0.5, so commanded deltas are divided by it.
    Getting this wrong halves every motion and reads as a bad model."""
    here = top_down_pose([0.40, 0.0, 0.30])
    there = top_down_pose([0.44, 0.0, 0.30])
    chunk = action_chunk(here, there, grip=0.0, horizon=4)
    assert chunk.shape == (4, 7)
    assert chunk[:, 0].sum() * REL_IK_SCALE == pytest.approx(0.04, abs=1e-6)


def test_a_chunk_respects_the_per_step_ceiling():
    here = top_down_pose([0.30, 0.0, 0.30])
    there = top_down_pose([0.80, 0.0, 0.30])
    chunk = action_chunk(here, there, grip=0.0, horizon=2)
    per_step = np.linalg.norm(chunk[0, 0:3]) * REL_IK_SCALE
    assert per_step <= 0.02 + 1e-9


def test_a_chunk_carries_the_gripper_command():
    chunk = action_chunk(
        top_down_pose([0.4, 0, 0.3]), top_down_pose([0.4, 0, 0.3]), grip=1.0, horizon=3
    )
    assert np.all(chunk[:, 6] == 1.0)


def test_rotation_is_an_axis_angle_vector_on_root_axes():
    here = top_down_pose([0.4, 0.0, 0.3], yaw=0.0)
    there = top_down_pose([0.4, 0.0, 0.3], yaw=0.3)
    chunk = action_chunk(here, there, grip=0.0, horizon=3)
    total = chunk[:, 3:6].sum(axis=0) * REL_IK_SCALE
    assert total[2] == pytest.approx(0.3, abs=1e-6)
    assert np.allclose(total[:2], 0.0, atol=1e-9)


def test_applying_a_chunk_reaches_the_target():
    """Integrating the chunk the way the controller does must land on the goal."""
    here = top_down_pose([0.40, -0.05, 0.32], yaw=0.0)
    there = top_down_pose([0.46, 0.03, 0.18], yaw=0.4)
    n = steps_needed(here, there)
    chunk = action_chunk(here, there, grip=0.0, horizon=n)
    flange = tcp_to_flange(here)
    pos = np.asarray(flange.pos, dtype=float)
    rot = quat_to_rot(flange.quat)
    for step in chunk:
        pos = pos + step[0:3] * REL_IK_SCALE
        omega = step[3:6] * REL_IK_SCALE
        angle = float(np.linalg.norm(omega))
        if angle > 1e-12:
            rot = quat_to_rot(quat_from_axis_angle(omega / angle, angle)) @ rot
    landed = flange_to_tcp(pos, rot_to_quat(rot))
    assert np.allclose(landed.pos, there.pos, atol=2e-3), landed.pos
    from rung.contracts.frames import quat_angle_between

    assert quat_angle_between(landed.quat, there.quat) < 0.02


def test_steps_needed_scales_with_distance():
    here = top_down_pose([0.30, 0.0, 0.30])
    assert steps_needed(here, top_down_pose([0.32, 0.0, 0.30])) == 1
    assert steps_needed(here, top_down_pose([0.50, 0.0, 0.30])) == 10


# ---- capabilities and the bridge -----------------------------------------


def test_capabilities_describe_the_droid_rig():
    caps = droid_capabilities()
    assert caps.gripper.max_opening_m == 0.085
    assert caps.tcp_from_ee.pos[0] == pytest.approx(FLANGE_TO_FINGERTIP_M)
    assert "OpenGL" in caps.notes["camera_convention"]
    # cameras are named by role, so a tool enum means the same thing on every rig
    assert caps.camera("scene").has_depth
    assert {c.name for c in caps.cameras} == {"scene", "wrist"}


def test_the_bridge_runs_a_motion_through_both_loops():
    """The harness thread asks for a move; the simulator thread plays it out.

    A fake runner stands in for RoboLab: it steps a flange pose by the chunk it is given
    and hands back a fresh observation, which is exactly what the real loop does.
    """
    caps = droid_capabilities(scene_size=(64, 48), wrist_size=(32, 32))
    emb = RoboLabEmbodiment(capabilities=caps, chunk_steps=4)
    client = RungInferenceClient(emb, instruction="test")

    outcome: dict = {}

    def harness():
        try:
            start = emb.observe()
            target = TCPTarget(
                pose=top_down_pose([0.50, 0.05, 0.30]),
                gripper=GripperCommand(GripperAction.CLOSE),
                max_lin_speed_mps=0.25,
                timeout_s=30.0,
            )
            outcome["start"] = start.tcp.pos.copy()
            outcome["report"] = emb.execute(target)
        except Exception as exc:  # surfaced by the assertions below
            outcome["error"] = exc
        finally:
            emb.close()

    thread = run_in_thread(harness)

    # The fake runner integrates position and orientation, because the fingertip offset
    # points wherever the gripper points: a runner that ignored rotation would land the
    # flange 20 cm from the truth and the test would be measuring itself.
    flange = np.array([0.40, 0.0, 0.20])
    rot = np.eye(3)
    for _ in range(200):
        chunk = client.get_action(
            fake_obs(flange=tuple(flange), flange_quat=rot_to_quat(rot), size=(48, 64))
        )
        for step in chunk:
            flange = flange + np.asarray(step[0:3], dtype=float) * REL_IK_SCALE
            omega = np.asarray(step[3:6], dtype=float) * REL_IK_SCALE
            angle = float(np.linalg.norm(omega))
            if angle > 1e-12:
                rot = quat_to_rot(quat_from_axis_angle(omega / angle, angle)) @ rot
        if not thread.is_alive():
            break
    thread.join(timeout=5)

    assert "error" not in outcome, outcome.get("error")
    assert outcome["start"][0] == pytest.approx(0.40 + FLANGE_TO_FINGERTIP_M)
    landed = flange_to_tcp(flange, rot_to_quat(rot))
    assert np.allclose(landed.pos, [0.50, 0.05, 0.30], atol=6e-3), landed.pos
    assert outcome["report"].residual_m < 0.01


def test_the_client_holds_still_once_the_harness_has_finished():
    """The simulator must be able to finish its episode after the policy stops."""
    emb = RoboLabEmbodiment(
        capabilities=droid_capabilities(scene_size=(64, 48), wrist_size=(32, 32))
    )
    client = RungInferenceClient(emb, instruction="test")
    emb.close()
    chunk = client.get_action(fake_obs(travel=0.3, size=(48, 64)))
    assert np.all(chunk[:, 0:6] == 0.0)
    assert np.all(chunk[:, 6] == pytest.approx(0.3))


def drive_targets(make_targets, *, stalled=False, jaw=None):
    """Exercise both bridge threads, with a deadline so queue regressions cannot hang CI."""
    caps = droid_capabilities(scene_size=(64, 48), wrist_size=(32, 32))
    emb = RoboLabEmbodiment(capabilities=caps)
    client = RungInferenceClient(emb, "test", chunk_steps=2)
    reports, chunks, errors = [], [], []

    def harness():
        try:
            start = emb.observe()
            for target in make_targets(start):
                reports.append(emb.execute(target))
        except Exception as exc:
            errors.append(exc)
        finally:
            emb.close()

    def simulator():
        pos = np.array([0.4, 0, 0.3])
        rot = np.eye(3)
        travel = 0.5 if jaw is None else 0.0
        try:
            for _ in range(500):
                chunk = client.get_action(
                    fake_obs(flange=pos, flange_quat=rot_to_quat(rot), travel=travel)
                )
                if client._finished:
                    # Repeated post-finish calls must not block or release the last grip.
                    client.get_action(fake_obs(flange=pos, travel=0.5))
                    break
                chunks.append(chunk)
                if jaw is not None:
                    for step in chunk:
                        travel = jaw(travel, float(step[6]))
                if not stalled:
                    for step in chunk:
                        pos += step[:3] * REL_IK_SCALE
                        omega = step[3:6] * REL_IK_SCALE
                        angle = np.linalg.norm(omega)
                        if angle > 1e-12:
                            rot = quat_to_rot(quat_from_axis_angle(omega / angle, angle)) @ rot
        except Exception as exc:
            errors.append(exc)
            client.report_error(exc)

    worker = run_in_thread(harness)
    runner = run_in_thread(simulator)
    runner.join(4)
    worker.join(1)
    assert not runner.is_alive() and not worker.is_alive(), "bridge failed to terminate"
    assert not errors, errors
    return reports, np.concatenate(chunks) if chunks else np.empty((0, 7))


def test_bridge_obeys_target_speed_and_deadline_when_unreachable():
    def targets(obs):
        return [
            TCPTarget(
                Pose(obs.tcp.pos + [0.2, 0, 0], obs.tcp.quat),
                max_lin_speed_mps=0.015,
                timeout_s=0.2,
            )
        ]

    reports, chunks = drive_targets(targets, stalled=True)
    assert reports[0].outcome.value == "expired"
    assert reports[0].duration_s <= 0.2 + 1e-9
    assert np.max(np.linalg.norm(chunks[:, :3], axis=1)) * REL_IK_SCALE * 15 <= 0.015 + 1e-8


def test_stationary_close_waits_and_hold_preserves_the_close_command():
    def targets(obs):
        return [
            TCPTarget(obs.tcp, GripperCommand(GripperAction.CLOSE), timeout_s=3),
            TCPTarget(Pose(obs.tcp.pos + [0.02, 0, 0], obs.tcp.quat), timeout_s=3),
        ]

    reports, chunks = drive_targets(targets)
    assert reports[0].duration_s >= 1.5
    assert all(r.outcome.value == "reached" for r in reports)
    assert np.all(chunks[:, 6] == 1.0), "HOLD must preserve closure, not measured jaw travel"


def test_close_cannot_run_past_a_short_deadline():
    reports, _ = drive_targets(
        lambda obs: [TCPTarget(obs.tcp, GripperCommand(GripperAction.CLOSE), timeout_s=0.2)]
    )
    assert reports[0].outcome.value == "expired"
    assert reports[0].duration_s <= 0.2 + 1e-9


def test_subperiod_timeout_executes_no_action():
    reports, chunks = drive_targets(
        lambda obs: [TCPTarget(Pose(obs.tcp.pos + [0.2, 0, 0], obs.tcp.quat), timeout_s=0.001)]
    )
    assert reports[0].outcome.value == "expired"
    assert reports[0].duration_s == 0
    assert len(chunks) == 0


def test_rotation_keeps_tcp_on_the_requested_straight_path():
    start = top_down_pose([0.4, 0, 0.3])
    end = Pose([0.5, 0.05, 0.4], quat_from_axis_angle([0, 1, 0], 0.7))
    n = steps_needed(start, end)
    chunks = action_chunk(start, end, 0, n)
    flange = tcp_to_flange(start)
    pos, rot = flange.pos.copy(), quat_to_rot(flange.quat)
    for i, step in enumerate(chunks):
        pos += step[:3] * REL_IK_SCALE
        omega = step[3:6] * REL_IK_SCALE
        angle = np.linalg.norm(omega)
        if angle > 1e-12:
            rot = quat_to_rot(quat_from_axis_angle(omega / angle, angle)) @ rot
        actual = flange_to_tcp(pos, rot_to_quat(rot))
        expected = start.pos + (end.pos - start.pos) * ((i + 1) / n)
        assert np.allclose(actual.pos, expected, atol=1e-6)


def test_actual_ee_eef_rotation_is_not_interchangeable():
    raw = fake_obs()
    obs = to_observation(raw, t=0)
    assert np.allclose(obs.tcp.pos, [0.45 + FLANGE_TO_FINGERTIP_M, 0, 0.35])
    del raw["proprio_obs"]["ee_quat"]
    with pytest.raises(KeyError, match="ee_quat"):
        to_observation(raw, t=0)


def test_verified_client_two_episodes_external_grades():
    from rung.adapters.robolab.client import RungPolicyClient
    from rung.policy import Decision, Verdict

    policies = []

    class DonePolicy:
        name = "done-fixture"
        interface = "L0"

        def reset(self, instruction, capabilities):
            self.instruction = instruction

        def act(self, observation, feedback):
            validate_observation(observation, capabilities_from_observation(fake_obs()))
            return Decision("done", "test", verdict=Verdict.DONE)

    def factory(emb):
        policy = DonePolicy()
        policies.append(policy)
        return policy

    from rung.adapters.robolab.translate import capabilities_from_observation

    client = RungPolicyClient(interface="l0", policy_factory=factory)
    for idx in range(2):
        client.begin_episode(idx)
        action = client.infer_batch(fake_obs(), "test task", env_ids=[0])
        assert action[0]["action"].shape == (7,)
        client.reset()
        assert client.results[idx, 0].success is None
        client.record_grades([dict(env_id=0, success=idx == 0, step=1)], episode_idx=idx)
        assert client.results[idx, 0].success == (idx == 0)
    assert len(policies) == 2
    assert policies[0] is not policies[1]
    assert not client.errors


def test_close_starts_only_after_arrival_and_gets_its_own_settle_time():
    reports, chunks = drive_targets(
        lambda obs: [
            TCPTarget(
                Pose(obs.tcp.pos + [0.1, 0, 0], obs.tcp.quat),
                GripperCommand(GripperAction.CLOSE),
                max_lin_speed_mps=0.05,
                timeout_s=6,
            )
        ]
    )
    closing = np.flatnonzero(chunks[:, 6] == 1)
    assert len(closing)
    first = closing[0]
    assert first > 0
    assert np.all(chunks[:first, 6] == 0.5)
    travelled_before_close = float(chunks[:first, 0].sum() * REL_IK_SCALE)
    assert travelled_before_close >= 0.094
    assert len(chunks[first:]) / 15 >= 1.5
    assert reports[0].outcome.value == "reached"


def test_unreachable_target_never_closes_on_the_way():
    reports, chunks = drive_targets(
        lambda obs: [
            TCPTarget(
                Pose(obs.tcp.pos + [0.1, 0, 0], obs.tcp.quat),
                GripperCommand(GripperAction.CLOSE),
                timeout_s=2,
            )
        ],
        stalled=True,
    )
    # A target the arm cannot reach ends as a stall (blocked) once progress stops, or as
    # an expiry if the deadline comes first. Either way the close never happens.
    assert reports[0].outcome.value in ("expired", "blocked")
    assert np.all(chunks[:, 6] == 0.5)


def test_verified_termination_during_motion_is_not_a_robot_error():
    from rung.adapters.robolab.client import RungPolicyClient
    from rung.loop import EndReason
    from rung.policy import Decision

    class MovingPolicy:
        name = "moving-fixture"
        interface = "L0"

        def reset(self, instruction, capabilities):
            pass

        def act(self, observation, feedback):
            return Decision(
                "move",
                "test",
                targets=[TCPTarget(Pose(observation.tcp.pos + [0.1, 0, 0], observation.tcp.quat))],
            )

    client = RungPolicyClient(policy_factory=lambda emb: MovingPolicy())
    client.begin_episode(0)
    client.infer_batch(fake_obs(), "test", env_ids=[0])
    client.reset()
    result = client.results[0, 0]
    assert result.end_reason is EndReason.EXTERNAL
    assert result.error is None
    assert result.success is None


def test_a_motion_that_ran_out_of_time_is_not_reported_as_unreachable():
    """An expiry mid-convergence does not read as a joint limit.

    The report distinguishes an arm that moved and ran out of time from one that did not
    move, so a reachable target is not described as unreachable.
    """
    from rung.adapters.robolab.bridge import _expiry_reason
    from rung.contracts.action import GripperAction, GripperCommand, TCPTarget
    from rung.contracts.frames import Pose

    target = TCPTarget(
        pose=Pose(np.array([0.60, 0.0, 0.10]), np.array([1.0, 0, 0, 0])),
        gripper=GripperCommand(GripperAction.HOLD),
    )

    def observation_at(x):
        return to_observation(fake_obs(flange=(x, 0.0, 0.10)), t=0.0)

    start = observation_at(0.37)
    moved = _expiry_reason(
        target, start, observation_at(0.57), np.abs(np.array([0.03, 0.0, 0.0])), 0
    )
    assert "converging" in moved and "unreachable" not in moved

    stuck = _expiry_reason(
        target, start, observation_at(0.3701), np.abs(np.array([0.23, 0.0, 0.0])), 0
    )
    assert "did not move" in stuck and "out of reach" in stuck


def test_the_motion_budget_grows_with_the_distance():
    """A flat timeout is wrong at both ends on a rig that converges geometrically."""
    caps = droid_capabilities()
    near = caps.time_to_converge(0.03)
    far = caps.time_to_converge(0.30)
    assert far > near > 0
    # Thirty steps per e-fold at 15 Hz: a 23 cm approach needs well over a flat 12 s.
    assert caps.time_to_converge(0.23) > 12.0
    assert caps.time_to_converge(0.0) > 0


def test_expiry_names_the_part_of_arrival_that_failed():
    """ "0.0 cm short" is never the explanation for a motion that did not arrive.

    Arrival requires position, orientation and a settled gripper; when the position is
    within tolerance, the report names the orientation or gripper that is not.
    """
    from rung.adapters.robolab.bridge import _expiry_reason
    from rung.contracts.action import GripperAction, GripperCommand, TCPTarget
    from rung.contracts.frames import Pose

    target = TCPTarget(
        pose=Pose(np.array([0.50, 0.0, 0.20]), np.array([1.0, 0, 0, 0])),
        gripper=GripperCommand(GripperAction.HOLD),
    )
    start = to_observation(fake_obs(flange=(0.30, 0.0, 0.20)), t=0.0)
    end = to_observation(fake_obs(flange=(0.50, 0.0, 0.20)), t=1.0)
    arrived = np.abs(np.array([0.0001, 0.0, 0.0]))

    turning = _expiry_reason(target, start, end, arrived, 0, rot=0.4, settled=True)
    assert "orientation" in turning and "0.0 cm short" not in turning

    gripping = _expiry_reason(target, start, end, arrived, 0, rot=0.0, settled=False)
    assert "gripper" in gripping


def test_an_arm_that_stops_moving_is_reported_blocked_within_seconds():
    """The controller converges geometrically, so an arm that is still short and not
    closing on its target is pressing on something. After ``STALL_S`` without progress
    the motion is reported blocked."""
    from rung.adapters.robolab.bridge import STALL_S

    caps = droid_capabilities(scene_size=(64, 48), wrist_size=(32, 32))
    emb = RoboLabEmbodiment(capabilities=caps, chunk_steps=4)
    client = RungInferenceClient(emb, instruction="test")
    outcome: dict = {}

    def harness():
        try:
            emb.observe()
            outcome["report"] = emb.execute(
                TCPTarget(
                    pose=top_down_pose([0.50, 0.0, 0.05]),
                    gripper=GripperCommand(GripperAction.HOLD),
                    max_lin_speed_mps=0.25,
                    timeout_s=30.0,
                )
            )
        except Exception as exc:
            outcome["error"] = exc
        finally:
            emb.close()

    thread = run_in_thread(harness)
    flange = np.array([0.40, 0.0, 0.20])
    rot = np.eye(3)
    steps = 0
    for _ in range(600):
        chunk = client.get_action(
            fake_obs(flange=tuple(flange), flange_quat=rot_to_quat(rot), size=(48, 64))
        )
        for step in chunk:
            steps += 1
            # The arm follows the commands until it is 4 cm above the target, then it
            # is stopped by something and no longer moves at all.
            if flange[2] > 0.05 + FLANGE_TO_FINGERTIP_M + 0.04:
                flange = flange + np.asarray(step[0:3], dtype=float) * REL_IK_SCALE
        if not thread.is_alive():
            break
    thread.join(timeout=5)
    assert "error" not in outcome, outcome.get("error")
    report = outcome["report"]
    assert report.outcome is ExecutionOutcome.BLOCKED, report
    assert "made no progress" in report.reason and "cm short on z" in report.reason
    assert report.duration_s < STALL_S + 6.0, report.duration_s


def _run_with_thinking(time_mode: str, think_s: float, excluded_s: float = 0.0):
    """A harness that takes ``think_s`` real seconds before asking for a 6 cm move."""
    import time as _time

    caps = droid_capabilities(scene_size=(64, 48), wrist_size=(32, 32))
    emb = RoboLabEmbodiment(capabilities=caps, chunk_steps=1, time_mode=time_mode)
    client = RungInferenceClient(emb, instruction="test")
    outcome: dict = {}

    def harness():
        try:
            start = emb.observe()
            _time.sleep(think_s)
            if excluded_s:
                emb.exclude_wall(excluded_s)
            outcome["report"] = emb.execute(
                TCPTarget(
                    pose=Pose(start.tcp.pos + np.array([0.06, 0.0, 0.0]), start.tcp.quat),
                    gripper=GripperCommand(GripperAction.HOLD),
                    max_lin_speed_mps=0.25,
                    timeout_s=20.0,
                )
            )
        except Exception as exc:
            outcome["error"] = exc
        finally:
            emb.close()

    thread = run_in_thread(harness)
    flange = np.array([0.40, 0.0, 0.20])
    still, first_move = 0, None
    for i in range(2000):
        chunk = client.get_action(fake_obs(flange=tuple(flange), size=(48, 64)))
        moving = float(np.abs(chunk[:, :6]).sum()) > 1e-9
        if moving and first_move is None:
            first_move = i
        if first_move is None and not moving:
            still += len(chunk)
        for step in chunk:
            flange = flange + np.asarray(step[0:3], dtype=float) * REL_IK_SCALE
        if not thread.is_alive():
            break
    thread.join(timeout=5)
    assert "error" not in outcome, outcome.get("error")
    return outcome["report"], still, client


def test_realtime_mode_holds_the_robot_for_as_long_as_the_harness_took():
    """The simulator waits while the harness thinks; in real time that wait has to pass in
    the simulation too, with the robot holding still, before the motion starts."""
    report, still, client = _run_with_thinking("realtime", think_s=0.4)
    dt = 1.0 / 15.0
    assert report.waited_s == pytest.approx(0.4, abs=0.1)
    assert abs(still - 0.4 / dt) <= 2, still  # six control steps of holding
    assert client.t >= 0.4  # and the episode clock counted them


def test_recording_time_is_not_charged_to_the_robot():
    report, still, _ = _run_with_thinking("realtime", think_s=0.4, excluded_s=0.3)
    assert report.waited_s == pytest.approx(0.1, abs=0.1)
    assert still <= 3


def test_paused_mode_lets_the_world_wait_for_free():
    report, still, _ = _run_with_thinking("paused", think_s=0.4)
    assert report.waited_s == pytest.approx(0.4, abs=0.1)  # still measured and reported
    assert still <= 1


def test_an_unknown_time_mode_is_refused():
    with pytest.raises(ValueError):
        RoboLabEmbodiment(capabilities=droid_capabilities(), time_mode="fast")


def test_a_gain_scales_the_commands_but_not_the_gripper():
    here = top_down_pose([0.40, 0.0, 0.30])
    there = top_down_pose([0.50, 0.05, 0.25], yaw=0.2)
    plain = action_chunk(here, there, grip=1.0, horizon=6)
    compensated = action_chunk(here, there, grip=1.0, horizon=6, gain=0.25)
    assert np.allclose(compensated[:, :6], plain[:, :6] / 0.25, atol=1e-5)
    assert np.all(compensated[:, 6] == 1.0)


@pytest.mark.parametrize("gain", [0.0, -0.3, 1.5])
def test_a_gain_outside_the_unit_interval_is_refused(gain):
    here = top_down_pose([0.40, 0.0, 0.30])
    with pytest.raises(ValueError):
        action_chunk(here, here, grip=0.0, horizon=1, gain=gain)
    with pytest.raises(ValueError):
        RoboLabEmbodiment(tracking_gain=gain)


def _steps_to_arrive(realised: float, gain: float, distance: float = 0.26) -> int:
    """Re-plan every step toward a fixed target on a rig that realises a fixed share of
    each commanded step, the way the bridge drives RoboLab."""
    here = top_down_pose([0.40, -0.10, 0.30])
    target = top_down_pose([0.40, -0.10 + distance, 0.30])
    pos = np.asarray(here.pos, dtype=float)
    for step in range(1, 400):
        now = top_down_pose(pos.tolist())
        n = max(1, steps_needed(now, target, max_dxyz=0.25 / 15))
        row = action_chunk(now, target, grip=0.0, horizon=n, max_dxyz=0.25 / 15, gain=gain)[0]
        pos = pos + row[:3] * REL_IK_SCALE * realised
        if np.linalg.norm(pos - target.pos) <= 0.006:
            return step
    return 400


def test_compensating_the_tracking_gain_moves_at_the_planned_speed():
    """At a realised share of 0.283 an uncompensated 26 cm move takes about three and a
    half times the planned steps; compensated it takes the planned steps."""
    planned = int(np.ceil(0.26 / (0.25 / 15)))
    slow = _steps_to_arrive(realised=0.283, gain=1.0)
    fast = _steps_to_arrive(realised=0.283, gain=0.283)
    assert slow >= 3 * planned
    assert fast <= planned + 1


def _jaw_closing_onto(stop_at: float):
    """A jaw that moves 0.1 of its travel per step toward the command and stops on an
    object at ``stop_at``."""

    def step(travel, command):
        moved = travel + float(np.clip(command - travel, -0.1, 0.1))
        return min(moved, stop_at) if command > travel else moved

    return step


def test_a_close_is_settled_once_the_jaw_stops_on_the_object():
    reports, _ = drive_targets(
        lambda obs: [TCPTarget(obs.tcp, GripperCommand(GripperAction.CLOSE), timeout_s=3)],
        jaw=_jaw_closing_onto(0.3),
    )
    assert reports[0].outcome.value == "reached"
    assert reports[0].duration_s < 1.0


def test_a_jaw_that_only_jitters_still_waits_the_full_close_time():
    """A small twitch of the open jaw is not a close that stopped on an object."""
    samples = []

    def jitter(travel, command):
        samples.append(travel)
        return 0.03 if len(samples) in (2, 3) else 0.0

    reports, _ = drive_targets(
        lambda obs: [TCPTarget(obs.tcp, GripperCommand(GripperAction.CLOSE), timeout_s=3)],
        jaw=jitter,
    )
    assert reports[0].outcome.value == "reached"
    assert reports[0].duration_s >= 1.5


def test_a_jaw_that_never_moves_still_waits_the_full_close_time():
    reports, _ = drive_targets(
        lambda obs: [TCPTarget(obs.tcp, GripperCommand(GripperAction.CLOSE), timeout_s=3)],
        jaw=lambda travel, command: travel,
    )
    assert reports[0].outcome.value == "reached"
    assert reports[0].duration_s >= 1.5


def test_optional_wrist_calibration_is_public_pose_and_intrinsics_without_depth():
    raw = fake_obs()
    images = raw["image_obs"]
    images["wrist_cam_K"] = images[f"{SCENE_CAMERA}_K"]
    images["wrist_cam_pos"] = images[f"{SCENE_CAMERA}_pos"]
    images["wrist_cam_quat"] = images[f"{SCENE_CAMERA}_quat"]
    obs = to_observation(raw, t=0.0, convention="opengl")
    assert obs.wrist.depth is None
    assert obs.wrist.K is not None and obs.wrist.pose is not None
    np.testing.assert_allclose(obs.wrist.pose.rot[:, 2], [0, 0, -1])
    del images["wrist_cam_quat"]
    with pytest.raises(KeyError, match="requires all"):
        to_observation(raw, t=0.0, convention="opengl")


def test_hold_advances_physics_without_changing_pose_or_grip():
    reports, chunks = drive_targets(lambda obs: [TCPTarget(obs.tcp, timeout_s=2, min_dwell_s=0.6)])
    assert reports[0].t_end - reports[0].t_start >= 0.6 - 1e-9
    assert reports[0].outcome is ExecutionOutcome.REACHED
    assert len(chunks) >= 9
    np.testing.assert_allclose(chunks[:, :6], 0, atol=1e-7)
    np.testing.assert_allclose(chunks[:, 6], 0.5)


def test_hold_deadline_expires_instead_of_fabricating_elapsed_time():
    reports, _ = drive_targets(lambda obs: [TCPTarget(obs.tcp, timeout_s=0.2, min_dwell_s=1.0)])
    assert reports[0].outcome is ExecutionOutcome.EXPIRED
    assert reports[0].t_end - reports[0].t_start < 1.0
    assert "hold interval" in reports[0].reason
