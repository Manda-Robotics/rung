"""Trace-guided failure inspection, inspired by ASPIRE; never grades or drives a robot.

Reads existing immutable evidence and produces a small index back to raw events and
before/after RGB. Findings are hypotheses for investigation, not inferred contacts
or model-visible oracle feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def error_family(message: str) -> str:
    """Classify only explicit error evidence; never infer quota from generic failures."""
    text = message.lower()
    if any(x in text for x in ("at capacity", "model capacity")):
        return "provider_capacity"
    status_503 = re.search(r"\b(?:http(?:\s+error|\s+status)?|code|status)[\s\"'_:=(]*503\b", text)
    status_429 = re.search(r"\b(?:http(?:\s+error|\s+status)?|code|status)[\s\"'_:=(]*429\b", text)
    if status_503 or (
        "unavailable" in text and any(x in text for x in ("eligibility", "service", "server"))
    ):
        return "provider_unavailable"
    if status_429 or (
        "rate limit" in text and any(x in text for x in ("request", "http", "provider", "quota"))
    ):
        return "provider_rate_limit_not_proven_quota"
    if any(x in text for x in ("authentication", "unauthorized", "forbidden")):
        return "authentication"
    if "timeout" in text or "expired" in text:
        return "deadline"
    if any(x in text for x in ("stream disconnected", "error sending request")):
        return "transport"
    if "vulkan" in text or "cuda error" in text:
        return "simulator_platform"
    return "unclassified"


def _lines(path: Path):
    with path.open() as stream:
        yield from stream


def inspect_episode(directory: Path) -> dict:
    directory = directory.resolve()
    trace = directory / "trace.jsonl"
    turns: list[dict] = []
    current: dict[tuple, dict] = {}
    errors: Counter = Counter()
    kinds: Counter = Counter()
    malformed = []
    unattached_findings = []
    for line_number, line in enumerate(_lines(trace), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed.append(line_number)
            continue
        if not isinstance(event, dict) or not isinstance(event.get("payload", {}), dict):
            malformed.append(line_number)
            continue
        kind = event.get("kind", "")
        if not isinstance(kind, str) or not isinstance(event.get("env_id", 0), (int, str)):
            malformed.append(line_number)
            continue
        if event.get("episode") is not None and not isinstance(event["episode"], str):
            malformed.append(line_number)
            continue
        kinds[kind] += 1
        payload = event.get("payload", {})
        identity = (event.get("episode"), event.get("env_id", 0))
        if kind == "decision":
            index = payload.get("index")
            turn = {
                "episode": identity[0],
                "env_id": identity[1],
                "index": index,
                "trace_line": line_number,
                "seq": event.get("seq"),
                "sim_time_s": event.get("t_sim"),
                "tool": payload.get("decision_kind"),
                "note": payload.get("note"),
                "execution_outcomes": [],
                "findings": [],
                "evidence_lines": [line_number],
            }
            for label, frame_index in (
                [("before", index - 1), ("after", index)] if isinstance(index, int) else []
            ):
                turn[label + "_rgb"] = [
                    p.name for p in sorted(directory.glob(f"obs_{frame_index:04d}_*.png"))
                ]
            turns.append(turn)
            current[identity] = turn
        turn = current.get(identity)
        if turn is None:
            if kind in ("execution_error", "intervention"):
                reason = str(payload.get("error") or payload.get("reason") or "")
                unattached_findings.append(
                    {
                        "trace_line": line_number,
                        "kind": kind,
                        "reason": reason,
                        "error_family": error_family(reason),
                    }
                )
            continue
        if kind == "execution" and not payload.get("servo", False):
            outcome = payload.get("outcome")
            turn["execution_outcomes"].append(outcome)
            turn["evidence_lines"].append(line_number)
            if outcome and outcome != "reached":
                turn["findings"].append(
                    {
                        "kind": "motion_" + str(outcome),
                        "reason": payload.get("reason"),
                        "residual_m": payload.get("residual_m"),
                        "residual_rot_rad": payload.get("residual_rot_rad"),
                    }
                )
        if kind in ("execution_error", "intervention"):
            reason = str(
                payload.get("error") or payload.get("reason") or payload.get("message") or ""
            )
            signature = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", reason)
            errors[signature] += 1
            turn["evidence_lines"].append(line_number)
            turn["findings"].append(
                {"kind": kind, "reason": reason, "error_family": error_family(reason)}
            )
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    classification = "unresolved"
    if result:
        interrupted = result.get("grade_detail", {}).get("infrastructure_interrupted", False)
        interrupted |= bool(result.get("usage", {}).get("unknown_cost_calls", 0))
        interrupted |= result.get("end_reason") in ("wall_clock_limit", "unaccounted_model_call")
        if interrupted:
            classification = "infrastructure_partial"
        elif result.get("error"):
            classification = "unresolved_terminal_error"
        elif isinstance(result.get("success"), bool):
            classification = "native_success" if result["success"] else "native_policy_failure"
    if classification == "native_policy_failure":
        for turn in turns:
            if turn["tool"] == "done":
                turn["findings"].append(
                    {
                        "kind": "completion_claim_disagrees_with_native_outcome",
                        "reason": (
                            "Inspect final RGB and native grade; model done is not task success."
                        ),
                    }
                )
    salient = sorted(
        (t for t in turns if t["findings"]), key=lambda t: (-len(t["findings"]), t["trace_line"])
    )
    return {
        "schema": "rung.failure-index.v1",
        "purpose": "researcher_only_not_policy_input",
        "episode_directory": str(directory),
        "trace_sha256": file_sha(trace),
        "result_sha256": file_sha(result_path) if result_path.exists() else None,
        "raw_result": result,
        "outcome_classification": classification,
        "unattached_findings": unattached_findings,
        "terminal_error_family": error_family(str(result["error"]))
        if result and result.get("error")
        else None,
        "event_counts": dict(kinds),
        "decision_count": len(turns),
        "malformed_trace_lines": malformed,
        "incomplete_trace": bool(malformed)
        or not any(kinds[k] for k in ("policy_end", "episode_end")),
        "repeated_failure_signatures": [{"reason": r, "count": n} for r, n in errors.most_common()],
        "salient_turns": salient,
        "turns": turns,
        "limitations": [
            "Finding labels are inspection hypotheses, not verified root causes.",
            "Missing after RGB stays missing; no image or outcome is fabricated.",
            "Do not expose native grades or researcher diagnostics to benchmark actors.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("refusing to overwrite an existing diagnostic report")
    report = inspect_episode(a.episode)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(a.out),
                "decisions": report["decision_count"],
                "salient_turns": len(report["salient_turns"]),
            }
        )
    )


if __name__ == "__main__":
    main()
