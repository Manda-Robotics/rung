"""Astranomy-aligned prompt and observation options: camera geometry text, recent
frames, extra views, wider waypoint chunks. Public rig calibration only."""

import json

import numpy as np
import pytest

from rung.adapters.robolab.bridge import RoboLabEmbodiment, RungInferenceClient
from rung.adapters.robolab.translate import (
    RIGHT_SHOULDER_CAMERA,
    droid_capabilities,
    to_observation,
)
from rung.contracts.frames import Pose
from rung.contracts.observation import CameraFrame
from rung.interfaces.camera_prompt import camera_geometry_text
from rung.interfaces.direct import DirectConfig, DirectPolicy
from rung.mock import SCENES
from rung.policy import Feedback
from tests.test_direct_session import reply, session
from tests.test_robolab_adapter import fake_obs


def down_camera(name="scene", pos=(0.45, 0.0, 0.9), width=64, height=48):
    K = np.array([[32.0, 0, 31.5], [0, 32.0, 23.5], [0, 0, 1]])
    pose = Pose(np.array(pos), np.array([0, 1, 0, 0.0]))  # OpenCV +z looks at -root z
    return CameraFrame(name, np.zeros((height, width, 3), np.uint8), 1.5, K=K, pose=pose)


# ---- camera geometry text ------------------------------------------------------------


def test_camera_geometry_text_states_pose_axes_intrinsics_and_wrist_mount():
    tcp = Pose(np.array([0.5, 0.0, 0.3]), np.array([0, 1, 0, 0.0]))
    wrist = down_camera("wrist", pos=(0.5, 0.0, 0.35))
    extra = CameraFrame("extra", np.zeros((4, 4, 3), np.uint8), 1.5)
    text = camera_geometry_text([down_camera(), wrist, extra], tcp, image_scale=0.5)
    assert "- scene: position [0.45, 0.0, 0.9] m" in text
    assert "optical axis [0.0, 0.0, -1.0]" in text and "tilted 90 deg below horizontal" in text
    assert "fx 16.0 fy 16.0" in text and "at 32x24" in text  # scaled intrinsics and size
    assert "Mounted on the hand" in text and "0.05" in text  # 5 cm above the TCP
    assert "- extra: RGB view only" in text
    assert "K^-1" in text


def test_camera_geometry_prompt_is_opt_in_and_uses_public_calibration_only():
    emb = SCENES["banana_in_bowl"]()
    s, fake = session([reply()])
    p = DirectPolicy(s, DirectConfig(camera_geometry_prompt=True))
    p.reset("t", emb.capabilities)
    p.act(emb.observe(), Feedback())
    texts = [c["text"] for c in fake.requests[0]["input"][0]["content"] if c["type"] == "text"]
    geometry = [t for t in texts if t.startswith("CAMERA GEOMETRY")]
    assert len(geometry) == 1 and "- scene: position" in geometry[0]
    assert "banana" not in geometry[0].lower() and "depth" not in geometry[0].lower()
    s, fake = session([reply()])
    p = DirectPolicy(s)
    p.reset("t", emb.capabilities)
    p.act(emb.observe(), Feedback())
    texts = [c["text"] for c in fake.requests[0]["input"][0]["content"] if c["type"] == "text"]
    assert not any(t.startswith("CAMERA GEOMETRY") for t in texts)


# ---- recent frames and extra views in the prompt --------------------------------------


def test_recent_frames_and_extra_views_are_shown_only_when_enabled():
    emb = SCENES["banana_in_bowl"]()
    obs = emb.observe()
    past = CameraFrame("scene", obs.scene.rgb, obs.t - 0.93, convention="opencv")
    obs.extra["recent_frames"] = [(0.93, past), (0.47, past)]
    obs.extra["views"] = [down_camera("scene_right")]
    s, fake = session([reply()])
    p = DirectPolicy(s, DirectConfig(recent_frames=True, extra_views=True))
    p.reset("t", emb.capabilities)
    p.act(obs, Feedback())
    content = fake.requests[0]["input"][0]["content"]
    images = [c for c in content if c["type"] == "image"]
    texts = [c["text"] for c in content if c["type"] == "text"]
    assert len(images) == 2 + 1 + 2  # two past frames, the extra view, scene and wrist
    assert any(t.startswith("RECENT FRAMES") for t in texts)
    assert any("0.93s before the current frame" in t for t in texts)
    assert any(t.startswith("Camera scene_right") for t in texts)
    # The current frames come after the past ones.
    labels = [t for t in texts if t.startswith("Camera ")]
    assert labels.index(next(t for t in labels if "before the current" in t)) < labels.index(
        next(t for t in labels if t.startswith("Camera scene, captured"))
    )
    s, fake = session([reply()])
    p = DirectPolicy(s)
    p.reset("t", emb.capabilities)
    p.act(obs, Feedback())
    assert len([c for c in fake.requests[0]["input"][0]["content"] if c["type"] == "image"]) == 2


def test_sequence_chunk_may_hold_sixteen_steps():
    assert DirectConfig(max_sequence_steps=16).max_sequence_steps == 16
    with pytest.raises(ValueError):
        DirectConfig(max_sequence_steps=17)


# ---- translation of an extra calibrated view ------------------------------------------


def with_right_camera(raw):
    images = raw["image_obs"]
    for suffix in ("", "_K", "_pos", "_quat", "_depth"):
        images[f"{RIGHT_SHOULDER_CAMERA}{suffix}"] = images[f"over_shoulder_left_camera{suffix}"]
    return raw


def test_extra_camera_becomes_a_calibrated_view_with_a_role_name():
    raw = with_right_camera(fake_obs())
    obs = to_observation(raw, t=0.0, extra_cameras=(RIGHT_SHOULDER_CAMERA,))
    views = obs.extra["views"]
    assert [v.name for v in views] == ["scene_right"]
    assert views[0].has_geometry and views[0].convention == "opencv"
    np.testing.assert_allclose(views[0].pose.pos, obs.scene.pose.pos)
    with pytest.raises(KeyError, match="no extra camera"):
        to_observation(fake_obs(), t=0.0, extra_cameras=("head_cam",))
    del raw["image_obs"][f"{RIGHT_SHOULDER_CAMERA}_K"]
    with pytest.raises(KeyError, match="needs _K"):
        to_observation(raw, t=0.0, extra_cameras=(RIGHT_SHOULDER_CAMERA,))


# ---- bridge frame history ---------------------------------------------------------------


def test_bridge_attaches_downscaled_frames_from_earlier_control_steps():
    caps = droid_capabilities(scene_size=(64, 48), wrist_size=(32, 32))
    emb = RoboLabEmbodiment(
        capabilities=caps, chunk_steps=1, history_steps=(14, 7), history_scale=0.5
    )
    client = RungInferenceClient(emb, instruction="test", chunk_steps=1)
    seen = {}

    def harness():
        try:
            first = emb.observe()
            seen["first"] = first.extra.get("recent_frames")
            from rung.contracts.action import TCPTarget
            from rung.contracts.frames import quat_from_axis_angle

            pose = Pose(first.tcp.pos + [0, 0, 0.03], quat_from_axis_angle([0, 1, 0], np.pi))
            emb.execute(TCPTarget(pose=pose, max_lin_speed_mps=0.25, timeout_s=1.2))
            seen["after"] = emb.observe().extra.get("recent_frames")
        except Exception as exc:
            seen["error"] = exc
        finally:
            emb.close()

    from rung.adapters.robolab.bridge import run_in_thread

    thread = run_in_thread(harness)
    for _ in range(60):
        client.get_action(fake_obs(size=(48, 64)))
        if not thread.is_alive():
            break
    thread.join(timeout=5)
    assert "error" not in seen, seen.get("error")
    assert seen["first"] is None  # nothing earlier than the first frame
    recent = seen["after"]
    assert recent and len(recent) == 4  # two ages x (scene, wrist)
    ages = sorted({round(age, 3) for age, _ in recent}, reverse=True)
    assert ages == [round(14 / 15, 3), round(7 / 15, 3)]
    frame = next(f for _, f in recent if f.name == "scene")
    assert frame.rgb.shape == (24, 32, 3)  # downscaled copy
    with pytest.raises(ValueError):
        RoboLabEmbodiment(capabilities=caps, history_steps=(0,))


# ---- time budget candidates (2026-10-22 subset diagnosis) -------------------------------


def test_terminal_tools_can_be_removed_and_the_prompt_says_how_episodes_end():
    from rung.interfaces.direct import tools as direct_tools

    assert {"done", "give_up"} <= {t.name for t in direct_tools("both")}
    assert not {"done", "give_up"} & {t.name for t in direct_tools("both", terminal_tools=False)}
    emb = SCENES["banana_in_bowl"]()
    s, fake = session([reply("done", {"note": "finished"})])
    p = DirectPolicy(s, DirectConfig(terminal_tools=False))
    p.reset("t", emb.capabilities)
    with pytest.raises(ValueError, match="unknown tool"):
        p.act(emb.observe(), Feedback())
    request = fake.requests[0]
    assert not any(t["name"] in ("done", "give_up") for t in request["tools"])
    assert "There is no done or give_up tool" in request["system_instruction"]
    s, fake = session([reply()])
    p = DirectPolicy(s)
    p.reset("t", emb.capabilities)
    p.act(emb.observe(), Feedback())
    assert "There is no done" not in fake.requests[0]["system_instruction"]


def test_done_can_wait_for_the_grader_instead_of_ending_the_trial():
    from rung.interfaces.direct import tools as direct_tools
    from rung.policy import Verdict

    plain = next(t for t in direct_tools("both") if t.name == "done")
    waiting = next(
        t for t in direct_tools("both", done_awaits_confirmation=True) if t.name == "done"
    )
    assert "ends only if" in waiting.description and "ends only if" not in plain.description
    emb = SCENES["banana_in_bowl"]()
    s, fake = session([reply("done", {"note": "finished"}), reply("observe", {"note": "look"})])
    p = DirectPolicy(s, DirectConfig(done_awaits_confirmation=True, done_confirmation_wait_s=2.5))
    p.reset("t", emb.capabilities)
    first = p.act(emb.observe(), Feedback())
    assert first.verdict is Verdict.CONTINUE and first.kind == "done"
    assert len(first.targets) == 1 and first.targets[0].min_dwell_s == 2.5
    assert first.payload["done_pending_confirmation"]["hold_s"] == 2.5
    assert "done does not end the trial by itself" in fake.requests[0]["system_instruction"]
    # The grader did not end the episode: the next observation says so and the policy goes on.
    later = emb.observe()
    later.t += 2.5
    feedback = Feedback()
    second = p.act(later, feedback)
    assert second.verdict is Verdict.CONTINUE
    assert "grader has not confirmed success" in feedback.structured["done_not_confirmed"]
    receipt = json.dumps(fake.requests[1])
    assert "done_not_confirmed" in receipt
    # Off by default: done ends the trial as before.
    s, _ = session([reply("done", {"note": "finished"})])
    p = DirectPolicy(s)
    p.reset("t", emb.capabilities)
    assert p.act(emb.observe(), Feedback()).verdict is Verdict.DONE


def test_faster_speed_caps_reach_the_target_and_stay_within_the_rig():
    emb = SCENES["banana_in_bowl"]()
    s, _ = session(
        [
            reply(
                "move_relative",
                {"delta_xyz_m": [0, 0, 0.05], "rotation_vector_rad": [0, 0, 0], "note": "lift"},
            )
        ]
    )
    p = DirectPolicy(s, DirectConfig(max_speed_mps=0.3, max_ang_speed_rps=1.5))
    p.reset("t", emb.capabilities)
    target = p.act(emb.observe(), Feedback()).targets[0]
    assert target.max_lin_speed_mps == pytest.approx(min(0.3, emb.capabilities.max_lin_speed_mps))
    assert target.max_ang_speed_rps == pytest.approx(min(1.5, emb.capabilities.max_ang_speed_rps))
