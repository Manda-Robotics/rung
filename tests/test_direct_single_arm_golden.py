"""Every single-arm direct profile sends byte-identical requests after the bimanual work.

The golden file was written from the direct interface before per-arm addressing existed
(``RUNG_REGEN_DIRECT_GOLDEN=1 pytest tests/test_direct_single_arm_golden.py``). Each
profile fingerprints the system prompt, every text block, every image's bytes and the
tool schemas over two turns (a first look and a turn after a motion receipt), so any
change a single-arm model could see fails here.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from rung.backends.base import ImageBlock, ModelResponse, TextBlock, ToolCall
from rung.interfaces.direct import DirectConfig, DirectPolicy
from rung.mock import SCENES
from rung.policy import Feedback, ModelUsage

GOLDEN = Path(__file__).parent / "fixtures" / "direct_single_arm_golden.json"

# tcp_marker is left out: the mock's wrist camera is uncalibrated and the marker refuses it.
FLAGS = (
    "public_overlay",
    "table_projection",
    "stereo_triangulation",
    "spatial_frame_cues",
    "tcp_frame_motions",
    "geometry_feedback",
    "depth_probe",
    "action_sequence",
    "sequence_grasp_guard",
    "table_extent",
    "done_awaits_confirmation",
    "camera_geometry_prompt",
    "recent_frames",
    "extra_views",
)


def profiles():
    out = {}
    for motion in ("relative", "absolute", "both"):
        out[f"{motion}/default"] = {"motion_tools": motion}
        for flag in FLAGS:
            out[f"{motion}/{flag}"] = {"motion_tools": motion, flag: True}
    out["relative/no_terminal"] = {"terminal_tools": False}
    out["relative/everything"] = {f: True for f in FLAGS}
    return out


class Recorder:
    model = "golden"

    def __init__(self, calls):
        self.calls, self.requests = list(calls), []
        self.reset()

    def reset(self):
        self.turn, self.pending, self.session_id = 0, [], "golden"

    def resolve(self, receipts):
        self.requests.append({"receipts": sorted(receipts)})
        self.pending = []

    def complete(self, system, blocks, tools):
        self.requests.append(
            {
                "system": system,
                "blocks": [
                    b.text
                    if isinstance(b, TextBlock)
                    else ["image", b.label, b.mime, hashlib.sha256(b.data).hexdigest()]
                    for b in blocks
                    if isinstance(b, (TextBlock, ImageBlock))
                ],
                "tools": [[t.name, t.description, t.parameters] for t in tools],
            }
        )
        self.turn += 1
        name, args = self.calls.pop(0)
        call = ToolCall(name, args, id=f"call{self.turn}")
        self.pending = [call]
        return ModelResponse(tool_calls=[call], usage=ModelUsage(calls=1))


def fingerprint(config):
    emb = SCENES["banana_in_bowl"]()
    motion = config.get("motion_tools", "relative")
    move = (
        (
            "move_to",
            {
                "xyz_m": emb.observe().tcp.pos.tolist(),
                "quat_wxyz": emb.observe().tcp.quat.tolist(),
                "note": "hold",
            },
        )
        if motion == "absolute"
        else (
            "move_relative",
            {"delta_xyz_m": [0.01, 0, 0], "rotation_vector_rad": [0, 0, 0], "note": "nudge"},
        )
    )
    session = Recorder([move, ("observe", {"note": "look"})])
    policy = DirectPolicy(
        session,
        DirectConfig(**config),
        budgets={"horizon_s": 60, "max_model_calls": 10, "time_mode": "paused"},
    )
    policy.reset("put the banana in the bowl", emb.capabilities)
    obs = emb.observe()
    decision = policy.act(obs, Feedback())
    feedback = Feedback(structured={"outcomes": []})
    for target in decision.targets:
        report = emb.execute(target)
        feedback = Feedback(
            structured={
                "outcomes": [report.outcome.value],
                "final": report.final_tcp.pos.round(6).tolist(),
            }
        )
    policy.act(emb.observe(), feedback)
    text = json.dumps(session.requests, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def test_single_arm_profiles_are_byte_identical():
    current = {name: fingerprint(cfg) for name, cfg in profiles().items()}
    if os.environ.get("RUNG_REGEN_DIRECT_GOLDEN"):
        GOLDEN.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        pytest.skip("golden regenerated")
    golden = json.loads(GOLDEN.read_text())
    assert set(golden) == set(current)
    changed = sorted(k for k in golden if golden[k] != current[k])
    assert not changed, f"single-arm request bytes changed for {changed}"
