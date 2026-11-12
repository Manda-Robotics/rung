"""Two-arm addressing, the shared bimanual bridge, and the RoboDojo dual ARX X5 adapter.

Everything runs on a CPU against ``FakeRoboDojo``, which has RoboDojo's real observation
and action dicts and tracks end-effector targets the way its simulator-side IK does.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rung.adapters.bimanual import step_toward
from rung.adapters.robodojo.fake import FakeRoboDojo, run_rpc_episode
from rung.adapters.robodojo.policies import ScriptedBimanual, direct_policy
from rung.adapters.robodojo.rig import RoboDojoCodec
from rung.adapters.robodojo.server import RungRoboDojoModel, measure_table_z
from rung.backends.base import ModelResponse, ToolCall
from rung.contracts import ContractError, Pose, TCPTarget, validate_target
from rung.contracts.frames import quat_angle_between
from rung.contracts.observation import CameraFrame
from rung.direct_profile import FAKE_SCRIPT_BIMANUAL, load_profile
from rung.interfaces.direct import DirectConfig, DirectPolicy, tools
from rung.loop import EndReason, build_feedback
from rung.mock import SCENES
from rung.policy import Decision

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "gemini-4-argon-robodojo.json"
ZERO = [0, 0, 0]


class Scripted:
    """A native session that replays tool calls and keeps every request."""

    model = "scripted"

    def __init__(self, calls):
        self.calls, self.requests, self.receipts = list(calls), [], []
        self.reset()

    def reset(self):
        self.turn, self.pending, self.session_id = 0, [], "scripted"

    def resolve(self, receipts):
        self.receipts.append(receipts)
        self.pending = []

    def complete(self, system, blocks, tool_specs):
        from rung.policy import ModelUsage

        self.requests.append((system, blocks, tool_specs))
        self.turn += 1
        name, args = self.calls.pop(0)
        call = ToolCall(name, args, id=f"c{self.turn}")
        self.pending = [call]
        return ModelResponse(tool_calls=[call], usage=ModelUsage(calls=1))


def rel(arm, delta):
    return (
        "move_relative",
        {"arm": arm, "delta_xyz_m": delta, "rotation_vector_rad": ZERO, "note": "n"},
    )


def run(calls, sim=None, bimanual=True):
    sim = sim or FakeRoboDojo()
    session = Scripted(calls)
    config = DirectConfig(bimanual=bimanual)
    model = RungRoboDojoModel(lambda emb: DirectPolicy(session, config))
    codec = RoboDojoCodec()
    start = codec.observation(sim.get_obs(), 0.0)
    played = run_rpc_episode(model, sim)
    end = codec.observation(sim.get_obs(), 0.0)
    assert not model.errors
    (result,) = model.results.values()
    return result, session, start, end, played


# ---- contracts --------------------------------------------------------------------


def test_targets_name_arms_only_on_multi_arm_rigs():
    caps = RoboDojoCodec().capabilities
    pose = Pose([-0.3, -0.1, 0.9], [0, 1, 0, 0])
    validate_target(TCPTarget(pose, arm="left"), caps)
    validate_target(TCPTarget(pose, arm="left", paired=(TCPTarget(pose, arm="right"),)), caps)
    with pytest.raises(ContractError, match="no arm"):
        validate_target(TCPTarget(pose, arm="middle"), caps)
    with pytest.raises(ContractError, match="distinct"):
        validate_target(TCPTarget(pose, arm="left", paired=(TCPTarget(pose, arm="left"),)), caps)
    single = SCENES["banana_in_bowl"]().capabilities
    assert single.arms == ()
    with pytest.raises(ContractError, match="one arm"):
        validate_target(TCPTarget(Pose([0.5, 0, 0.2], [0, 1, 0, 0]), arm="left"), single)


def test_step_toward_bounds_translation_and_rotation():
    a = Pose([0, 0, 0], [1, 0, 0, 0])
    b = Pose([0.1, 0, 0], [0, 1, 0, 0])
    s = step_toward(a, b, 0.01, 0.1)
    assert np.linalg.norm(s.pos) == pytest.approx(0.01)
    assert quat_angle_between(a.quat, s.quat) == pytest.approx(0.1, abs=1e-6)
    assert np.allclose(step_toward(a, b, 1, 4).pos, b.pos)


def test_bimanual_tools_add_arm_and_move_both_and_leave_single_arm_alone():
    single = {t.name: t for t in tools("relative", action_sequence=True)}
    assert "arm" not in single["move_relative"].parameters["properties"]
    assert "move_both" not in single
    both = {t.name: t for t in tools("relative", action_sequence=True, arms=["left", "right"])}
    assert both["move_relative"].parameters["properties"]["arm"]["enum"] == ["left", "right"]
    assert both["set_gripper"].parameters["required"] == ["arm", "action", "note"]
    assert set(both["move_both"].parameters["required"]) == {"left", "right", "note"}
    assert (
        "arm" in both["execute_sequence"].parameters["properties"]["steps"]["items"]["properties"]
    )


def test_bimanual_profile_requires_two_arms():
    policy = DirectPolicy(Scripted([]), DirectConfig(bimanual=True))
    with pytest.raises(ValueError, match="two arms"):
        policy.reset("t", SCENES["banana_in_bowl"]().capabilities)


# ---- codec ------------------------------------------------------------------------


def test_robodojo_codec_layout():
    codec = RoboDojoCodec()
    obs = codec.observation(FakeRoboDojo().get_obs(), 0.0)
    assert set(obs.arms) == {"left", "right"} and obs.arms["left"].gripper.travel == 0.0
    assert obs.tcp == obs.arms["left"].tcp
    action = codec.command(obs, {a: s.tcp for a, s in obs.arms.items()}, {"left": 1, "right": 0})
    assert set(action) == {
        "left_ee_pose",
        "right_ee_pose",
        "left_ee_joint_state",
        "right_ee_joint_state",
    }
    assert action["left_ee_pose"].shape == (7,)
    assert action["left_ee_joint_state"][0] == 0.0 and action["right_ee_joint_state"][0] == 1.0
    # The commanded ee pose is the measured one when the setpoint is the measured TCP.
    sim = FakeRoboDojo()
    assert action["right_ee_pose"] == pytest.approx(sim.ee["right"], abs=1e-6)
    flipped = RoboDojoCodec(gripper_one_is_open=False)
    closed = flipped.observation(FakeRoboDojo(gripper_one_is_open=False).get_obs(), 0.0)
    assert closed.arms["left"].gripper.travel == 0.0
    batched = codec.observation({0: FakeRoboDojo().get_obs()}, 0.0)
    assert batched.arms["right"].tcp.pos == pytest.approx(obs.arms["right"].tcp.pos)


def test_robodojo_camera_geometry_is_converted_from_usd_axes():
    scene = RoboDojoCodec().observation(FakeRoboDojo(geometry=True).get_obs(), 0.0).scene
    assert scene.has_geometry
    # USD cameras look down their -z; in OpenCV axes the optical axis (+z) is world -z here.
    assert scene.pose.rot[:, 2] == pytest.approx([0, 0, -1])
    assert measure_table_z(scene, 0.765) == pytest.approx(0.765, abs=2e-3)


def test_robodojo_table_height_is_measured_near_the_declared_height_or_kept():
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    down = Pose([0.0, -0.1, 1.5], [0, 1, 0, 0])  # optical axis straight down
    flat = CameraFrame(
        "scene",
        np.zeros((240, 320, 3), np.uint8),
        0.0,
        depth=np.full((240, 320), 1.5 - 0.772, np.float32),
        K=K,
        pose=down,
    )
    assert measure_table_z(flat, 0.765) == pytest.approx(0.772, abs=1e-3)
    floor = CameraFrame(
        "scene",
        np.zeros((240, 320, 3), np.uint8),
        0.0,
        depth=np.full((240, 320), 1.5, np.float32),
        K=K,
        pose=down,
    )
    assert measure_table_z(floor, 0.765) is None
    bare = CameraFrame("scene", np.zeros((4, 4, 3), np.uint8), 0.0)
    assert measure_table_z(bare, 0.765) is None


# ---- episodes through RoboDojo's call sequence ------------------------------------


def test_each_arm_alone_then_both_together_with_per_arm_receipts():
    result, session, start, end, played = run(
        [
            rel("left", [0, 0, 0.05]),
            ("set_gripper", {"arm": "right", "action": "close", "note": "n"}),
            (
                "move_both",
                {
                    "left": {"delta_xyz_m": [0, 0.03, 0], "rotation_vector_rad": ZERO},
                    "right": {"delta_xyz_m": [0, 0.03, 0], "rotation_vector_rad": ZERO},
                    "note": "n",
                },
            ),
            rel("middle", [0, 0, 0.01]),
            ("done", {"note": "n"}),
        ]
    )
    assert result.end_reason == EndReason.DONE and result.decisions == 4
    left = end.arms["left"].tcp.pos - start.arms["left"].tcp.pos
    right = end.arms["right"].tcp.pos - start.arms["right"].tcp.pos
    assert left == pytest.approx([0, 0.03, 0.05], abs=0.008)
    assert right == pytest.approx([0, 0.03, 0], abs=0.008)
    assert end.arms["right"].gripper.travel > 0.9 and end.arms["left"].gripper.travel < 0.1
    receipts = [r for batch in session.receipts for r in batch.values()]
    assert any("unknown arm" in str(r.get("error")) for r in receipts)
    both = receipts[2]["execution_feedback"]["execution"][0]
    assert set(both["arms"]) == {"left", "right"}
    assert all(a["outcome"] == "reached" for a in both["arms"].values())
    system, blocks, _ = session.requests[0]
    assert "arm argument" in system
    assert '"arms"' in blocks[0].text and "root_frame" in blocks[0].text
    labels = [b.label for b in blocks if hasattr(b, "label")]
    assert labels == ["scene", "left_wrist", "right_wrist"]
    assert played > 0


def test_blocked_arm_reports_blocked_for_that_arm():
    _, session, start, end, _ = run(
        [rel("right", [0, 0, 0.05]), ("give_up", {"note": "n"})],
        sim=FakeRoboDojo(blocked={"right"}),
    )
    receipt = session.receipts[0]["c1"]["execution_feedback"]["execution"][0]
    assert receipt["outcome"] == "blocked"
    assert receipt["arms"]["right"]["outcome"] == "blocked"
    assert "right arm stopped" in receipt["reason"]
    assert end.arms["right"].tcp.pos == pytest.approx(start.arms["right"].tcp.pos)


def test_single_arm_profile_on_the_two_arm_rig_drives_the_primary_arm():
    calls = [
        ("move_relative", {"delta_xyz_m": [0, 0, 0.04], "rotation_vector_rad": ZERO, "note": "n"}),
        ("done", {"note": "n"}),
    ]
    _, _, start, end, _ = run(calls, bimanual=False)
    assert end.arms["left"].tcp.pos - start.arms["left"].tcp.pos == pytest.approx(
        [0, 0, 0.04], abs=0.008
    )
    assert end.arms["right"].tcp.pos == pytest.approx(start.arms["right"].tcp.pos, abs=1e-3)


def test_feedback_lists_every_arm_gripper():
    obs = RoboDojoCodec().observation(FakeRoboDojo().get_obs(), 0.0)
    feedback = build_feedback(Decision("observe", "n"), [], obs)
    assert feedback.structured["arm_gripper_travel"].keys() == {"left", "right"}
    assert feedback.lines[-1].startswith("Grippers: left open; right open")
    single = build_feedback(Decision("observe", "n"), [], SCENES["banana_in_bowl"]().observe())
    assert "arm_gripper_travel" not in single.structured


def test_scripted_motion_check_moves_each_arm_and_closes_both_grippers():
    sim = FakeRoboDojo(geometry=True)
    lines = []
    model = RungRoboDojoModel(lambda emb: ScriptedBimanual(log=lambda *a: lines.append(a)))
    codec = RoboDojoCodec()
    start = codec.observation(sim.get_obs(), 0.0)
    run_rpc_episode(model, sim)
    end = codec.observation(sim.get_obs(), 0.0)
    (result,) = model.results.values()
    assert result.end_reason == EndReason.DONE and len(lines) == 6
    for arm in ("left", "right"):  # lift 5 cm, then 4 cm forward (+y on this rig)
        moved = end.arms[arm].tcp.pos - start.arms[arm].tcp.pos
        assert moved == pytest.approx([0, 0.04, 0.05], abs=0.01)
        assert end.arms[arm].gripper.travel == pytest.approx(1.0)


# ---- the profile and the fake model ------------------------------------------------


def test_bundled_profile_is_bimanual_and_loads():
    profile = load_profile(PROFILE)
    assert profile.direct_config.bimanual and profile.model == "gemini-4-argon"


def test_direct_policy_with_the_fake_model_runs_the_bimanual_script(tmp_path):
    data = json.loads(PROFILE.read_text()) | {"model": "fake"}
    fake = tmp_path / "fake.json"
    fake.write_text(json.dumps(data))
    model = RungRoboDojoModel(
        lambda emb: direct_policy(fake, horizon_s=8, max_calls=12, trace_root=tmp_path / "t")
    )
    run_rpc_episode(model, FakeRoboDojo())
    (result,) = model.results.values()
    assert not model.errors
    assert result.end_reason == EndReason.DONE
    assert result.usage.calls == len(FAKE_SCRIPT_BIMANUAL)
    assert any((tmp_path / "t").iterdir())


def test_direct_policy_refuses_a_single_arm_profile(tmp_path):
    single = tmp_path / "single.json"
    single.write_text(json.dumps({"model": "fake", "direct_config": {}}))
    with pytest.raises(ValueError, match="bimanual"):
        direct_policy(single, horizon_s=8, max_calls=4)


# ---- the XPolicyLab shim ----------------------------------------------------------


def _load_shim(monkeypatch):
    template = types.ModuleType("XPolicyLab.model_template")
    template.ModelTemplate = object
    monkeypatch.setitem(sys.modules, "XPolicyLab", types.ModuleType("XPolicyLab"))
    monkeypatch.setitem(sys.modules, "XPolicyLab.model_template", template)
    spec = importlib.util.spec_from_file_location(
        "rung_robodojo_shim", ROOT / "deploy" / "robodojo" / "policy" / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("policy", ["scripted", "direct"])
def test_xpolicylab_shim_runs_an_episode_from_its_environment(monkeypatch, tmp_path, policy):
    data = json.loads(PROFILE.read_text()) | {"model": "fake"}
    (tmp_path / "fake.json").write_text(json.dumps(data))
    monkeypatch.setenv("RUNG_POLICY", policy)
    monkeypatch.setenv("RUNG_PROFILE", str(tmp_path / "fake.json"))
    monkeypatch.setenv("RUNG_OUT", str(tmp_path / "out"))
    shim = _load_shim(monkeypatch).Model({})
    assert json.loads((tmp_path / "out" / "status.json").read_text())["state"] == "ready"
    sim = FakeRoboDojo()
    shim.reset()
    for _ in range(2000):  # RoboDojo's loop, through the shim's own methods
        shim.update_obs(sim.get_obs())
        for action in shim.get_action():
            sim.take_action(action)
        if not shim.rung.session[2].is_alive():
            break
    done = json.loads((tmp_path / "out" / "status.json").read_text())
    assert done["state"] == "episode_done" and len(done["results"]) == 1
    shim.reset()
    status = json.loads((tmp_path / "out" / "status.json").read_text())
    (episode,) = status["results"].values()
    assert episode["end_reason"] == "policy_done" and not status["errors"]
    with pytest.raises(NotImplementedError):
        shim.get_action_batch()


def test_xpolicylab_shim_rejects_an_unknown_policy(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNG_POLICY", "relay")
    monkeypatch.setenv("RUNG_OUT", str(tmp_path))
    with pytest.raises(ValueError, match="scripted or direct"):
        _load_shim(monkeypatch).Model({})
