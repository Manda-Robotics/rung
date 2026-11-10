"""Scripted direct-interface episodes on the mock, canonicalised for byte-identity checks.

Opt-in mechanisms (the ``look`` tool first) must leave every existing profile's model
requests unchanged. ``request_digest`` runs a fixed reply script through the real loop
with a fake Gemini session and hashes what the model would have been sent: system text,
tool declarations, state JSON, receipts and image placement. Image bytes and random
target ids are replaced by placeholders so the digest does not depend on the renderer.

The digests in ``fixtures/direct_requests_before.json`` were produced at f27e8f8, before
the look tool existed. Regenerate only from that revision:
``python -c "from tests.direct_regression_scenarios import write_fixture; write_fixture()"``.
"""

import hashlib
import json
import re
from pathlib import Path

from rung.interfaces.direct import DirectConfig, DirectPolicy
from rung.loop import Limits, run_episode
from rung.mock import SCENES

FIXTURE = Path(__file__).parent / "fixtures" / "direct_requests_before.json"

# Run fields as deploy/robolab/run.py receives them:
# the default profile and the aligned direct profile the RoboLab leaderboard run used.
PROFILES = {
    "default": {},
    "aligned_proof": dict(
        motion_tools="both",
        max_translation_m=0.35,
        max_rotation_rad=1.57,
        table_projection=True,
        stereo_triangulation=True,
        camera_geometry_prompt=True,
        extra_views=True,
        image_scale=0.3,
        max_speed_mps=0.3,
        max_ang_speed_rps=1.5,
        done_awaits_confirmation=True,
    ),
    "sequence_table": dict(
        motion_tools="both",
        action_sequence=True,
        sequence_grasp_guard=True,
        table_extent=True,
        depth_probe=True,
        tcp_frame_motions=True,
        geometry_feedback=True,
    ),
    "no_terminal": dict(terminal_tools=False, spatial_frame_cues=True, recent_frames=True),
}


def _reply(name, args, index):
    return {
        "status": "completed",
        "steps": [
            {"type": "thought", "signature": f"sig-{index}", "summary": []},
            {"type": "function_call", "id": f"call{index}", "name": name, "arguments": args},
        ],
        "usage": {
            "total_input_tokens": 100,
            "total_output_tokens": 10,
            "total_thought_tokens": 20,
            "total_cached_tokens": 40,
        },
    }


SCRIPT = [
    ("move_relative", {"delta_xyz_m": [0, 0, -0.05], "rotation_vector_rad": [0, 0, 0.1]}),
    ("observe", {}),
    ("set_gripper", {"action": "close"}),
    ("wait", {"seconds": 0.2}),
    ("set_gripper", {"action": "open"}),
    (
        "execute_sequence",
        {
            "steps": [
                {
                    "tool": "move_relative",
                    "delta_xyz_m": [0, 0, 0.02],
                    "rotation_vector_rad": [0, 0, 0],
                },
                {"tool": "wait", "seconds": 0.1},
            ]
        },
    ),
    ("move_relative", {"delta_xyz_m": [0.03, 0, 0.05], "rotation_vector_rad": [0, 0, 0]}),
    ("done", {}),
    ("give_up", {}),
]


def _canonical(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "data" and value.get("type") == "image":
                out[key] = "<image>"
            elif key == "target_id":
                out[key] = "<id>"
            else:
                out[key] = _canonical(item)
        return out
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"\b[0-9a-f]{64}\b", "<sha>", value)
        return re.sub(r'"target_id": "[0-9a-f]{12}"', '"target_id": "<id>"', value)
    return value


def request_digest(profile_fields, script=SCRIPT):
    from types import SimpleNamespace

    from rung.backends.gemini_session import GeminiSession, TokenPrices
    from tests.test_direct_session import Replies

    replies = [
        _reply(name, {**args, "note": f"step {i}"}, i) for i, (name, args) in enumerate(script)
    ]
    fake = Replies(replies)
    session = GeminiSession(
        "test", TokenPrices(1, 2, 0.1, "test"), client=SimpleNamespace(interactions=fake)
    )
    emb = SCENES["banana_in_bowl"]()
    policy = DirectPolicy(
        session,
        DirectConfig(**profile_fields),
        budgets={"horizon_s": 60, "max_model_calls": 40, "time_mode": "paused"},
    )
    result = run_episode(
        emb,
        policy,
        task="t",
        limits=Limits(horizon_s=60, max_decisions=len(script), max_dollars=1),
        episode_seed=3,
    )
    blob = json.dumps(
        {
            "requests": _canonical(fake.requests),
            "end_reason": str(result.end_reason),
            "decisions": result.decisions,
            "servo_steps": result.servo_steps,
            "sim_seconds": round(result.sim_seconds, 6),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def write_fixture():
    FIXTURE.write_text(
        json.dumps(
            {
                "revision": "f27e8f8",
                "digests": {name: request_digest(fields) for name, fields in PROFILES.items()},
            },
            indent=1,
        )
        + "\n"
    )
