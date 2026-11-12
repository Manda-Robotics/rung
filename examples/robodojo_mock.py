"""Drive a two-arm rig through RoboDojo's call sequence on a CPU, with no simulator or key.

``FakeRoboDojo`` has the real observation and action dicts of RoboDojo's dual ARX X5 rig.
``RungRoboDojoModel`` is the same object the XPolicyLab policy server hosts on a GPU host.
The policy is either the scripted two-arm motion check or the bimanual Direct profile with
the fake model. Nothing is graded: the fake has no objects.
"""

import argparse
import json
import tempfile
from pathlib import Path

from rung.adapters.robodojo.fake import FakeRoboDojo, run_rpc_episode
from rung.adapters.robodojo.policies import ScriptedBimanual, direct_policy
from rung.adapters.robodojo.rig import RoboDojoCodec
from rung.adapters.robodojo.server import RungRoboDojoModel
from rung.loop import EndReason, Limits

PROFILE = Path(__file__).resolve().parents[1] / "configs" / "gemini-4-argon-robodojo.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", choices=("scripted", "direct"), default="direct")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "fake-profile.json"
        fake.write_text(json.dumps(json.loads(PROFILE.read_text()) | {"model": "fake"}))

        def factory(emb):
            if args.policy == "scripted":
                return ScriptedBimanual(log=lambda *a: None)
            return direct_policy(fake, horizon_s=8, max_calls=12)

        sim = FakeRoboDojo(geometry=True)
        codec = RoboDojoCodec()
        start = codec.observation(sim.get_obs(), 0.0)
        model = RungRoboDojoModel(factory, limits=Limits(max_decisions=12, horizon_s=8))
        actions = run_rpc_episode(model, sim)
        end = codec.observation(sim.get_obs(), 0.0)
    (result,) = model.results.values()
    print(result.summary())
    print(f"RoboDojo actions played: {actions} ({actions * 0.04:.2f} s at 25 Hz)")
    for arm in ("left", "right"):
        moved = (end.arms[arm].tcp.pos - start.arms[arm].tcp.pos).round(3).tolist()
        travel = end.arms[arm].gripper.travel
        print(f"{arm} arm moved {moved} m; gripper travel {travel:.2f} (1 = closed)")
    print("Fake model usage is metered for testing; no API request or charge occurs.")
    return 0 if result.end_reason == EndReason.DONE and not model.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
