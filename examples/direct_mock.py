"""Exercise direct-control tools and receipts with a fake model and the mock tabletop."""

from rung.contracts.trace import TraceWriter
from rung.direct_profile import DirectProfile, build_direct_policy
from rung.loop import EndReason, Limits, run_episode
from rung.mock import SCENES


def main() -> int:
    trace = TraceWriter(None)
    policy = build_direct_policy(DirectProfile(model="fake"))
    result = run_episode(
        SCENES["banana_in_bowl"](),
        policy,
        task="banana_in_bowl",
        limits=Limits(horizon_s=120, max_decisions=10, max_dollars=0.10),
        trace=trace,
        episode_seed=0,
    )
    print(result.summary())
    print(f"Model calls: {result.usage.calls}; trace events: {len(trace.events)}")
    print("Fake token usage is metered for testing; no API request or actual charge occurs.")
    print("The fake script demonstrates the control loop and does not solve the task.")
    return 0 if result.end_reason == EndReason.DONE and result.usage.calls == 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
