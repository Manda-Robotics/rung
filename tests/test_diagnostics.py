"""Regression coverage for trace inspection and evidence classification."""

import json

from rung.diagnostics import error_family, inspect_episode


def test_trace_does_not_join_environments_or_fabricate_after_rgb(tmp_path):
    events = [
        {"kind": "decision", "env_id": 0, "payload": {"index": 1, "decision_kind": "move"}},
        {"kind": "decision", "env_id": 1, "payload": {"index": 2, "decision_kind": "done"}},
        {"kind": "execution", "env_id": 0, "payload": {"outcome": "blocked", "residual_m": 0.1}},
        {"kind": "execution", "env_id": 0, "payload": {"outcome": "blocked", "servo": True}},
        {"kind": "policy_end", "payload": {}},
    ]
    (tmp_path / "trace.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n{bad\nnull\n")
    (tmp_path / "result.json").write_text(json.dumps({"success": False}))
    (tmp_path / "obs_0000_scene.png").touch()
    report = inspect_episode(tmp_path)
    assert report["terminal_error_family"] is None
    assert report["decision_count"] == 2
    assert report["turns"][0]["execution_outcomes"] == ["blocked"]
    assert report["turns"][1]["execution_outcomes"] == []
    assert report["turns"][0]["before_rgb"] == ["obs_0000_scene.png"]
    assert report["turns"][0]["after_rgb"] == []
    assert (
        report["turns"][1]["findings"][0]["kind"]
        == "completion_claim_disagrees_with_native_outcome"
    )
    assert report["malformed_trace_lines"] == [6, 7]
    assert report["incomplete_trace"]
    assert report["purpose"] == "researcher_only_not_policy_input"


def test_error_classification_does_not_invent_quota_or_root_cause():
    assert error_family("Selected model is at capacity") == "provider_capacity"
    assert error_family("UNAVAILABLE code 503") == "provider_unavailable"
    assert error_family("HTTP429") == "provider_rate_limit_not_proven_quota"
    assert error_family("empty success") == "unclassified"


def test_local_tool_unavailability_is_not_claimed_provider_failure():
    assert error_family("geometry tool unavailable") == "unclassified"


def test_partial_infrastructure_grade_does_not_become_policy_failure(tmp_path):
    (tmp_path / "trace.jsonl").write_text(
        json.dumps({"kind": "decision", "payload": {"index": 1, "decision_kind": "done"}})
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "success": False,
                "grade_detail": {"infrastructure_interrupted": True},
                "usage": {"unknown_cost_calls": 1},
            }
        )
    )
    report = inspect_episode(tmp_path)
    assert report["outcome_classification"] == "infrastructure_partial"
    assert not report["turns"][0]["findings"]


def test_failure_before_first_decision_is_inspectable_and_malformed_types_skipped(tmp_path):
    events = [
        {"kind": "execution_error", "payload": {"error": "eligibility UNAVAILABLE 503"}},
        {"kind": {}, "payload": {}},
        {"kind": "decision", "episode": []},
        {"kind": "decision", "env_id": [], "payload": {}},
    ]
    (tmp_path / "trace.jsonl").write_text("\n".join(map(json.dumps, events)))
    report = inspect_episode(tmp_path)
    assert report["decision_count"] == 0
    assert report["unattached_findings"][0]["error_family"] == "provider_unavailable"
    assert report["malformed_trace_lines"] == [2, 3, 4]


def test_coordinates_are_not_mistaken_for_provider_http_status_codes():
    assert error_family("target x0.503 y0.429 outside workspace") == "unclassified"
    assert error_family("angular rate limit") == "unclassified"


def test_unflagged_terminal_error_is_not_declared_a_policy_failure(tmp_path):
    (tmp_path / "trace.jsonl").write_text("")
    (tmp_path / "result.json").write_text(
        json.dumps({"success": False, "error": "BackendError: HTTP503"})
    )
    report = inspect_episode(tmp_path)
    assert report["outcome_classification"] == "unresolved_terminal_error"
    assert report["terminal_error_family"] == "provider_unavailable"
