"""Invariants on the task split.

If the split drifts, held-out numbers silently become development numbers. These tests
check structure only, not judgement (whether a given task belongs in test).
"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROLES_PATH = Path(__file__).resolve().parents[1] / "tasks" / "roles.yaml"
ROLES = ("development", "validation", "test", "full_only")

# Tasks the previous controller (vlm-pinpoint) was tuned on; they stay in development.
PINPOINT_TUNED = {
    "BananaInBowlTask",
    "BananasInCrateTask",
    "OneBottleInSquarePailTask",
    "FruitsOnionTask",
}
PINPOINT_FAILED = {
    "FruitsOrangesOnPlateTask",
    "RubiksCubeLeftOfBowlTask",
    "BowlStackingRightOnLeftTask",
    "FoodPacking1CansTask",
    "WhiteMugsInBinTask",
}


@pytest.fixture(scope="module")
def roles():
    return yaml.safe_load(ROLES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tasks(roles):
    return roles["tasks"]


def test_all_120_tasks_present_exactly_once(tasks):
    names = [t["name"] for t in tasks]
    assert len(names) == 120
    duplicates = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not duplicates


def test_every_task_has_the_required_fields(tasks):
    required = {"name", "template", "role", "difficulty", "instruction", "in_scope"}
    for t in tasks:
        missing = required - set(t)
        assert not missing, f"{t.get('name')} is missing {missing}"
        assert t["role"] in ROLES, f"{t['name']} has role {t['role']!r}"


def test_role_counts_match_the_header(roles, tasks):
    counts = collections.Counter(t["role"] for t in tasks)
    assert counts["development"] >= 12
    assert 12 <= counts["validation"] <= 18
    assert 25 <= counts["test"] <= 35
    assert sum(counts.values()) == 120


def test_every_pinpoint_tuned_task_is_development(tasks):
    """A task the old controller was tuned on cannot appear in validation or test."""
    by_name = {t["name"]: t for t in tasks}
    for name in PINPOINT_TUNED | PINPOINT_FAILED:
        assert name in by_name, f"{name} is not in the split"
        assert by_name[name]["role"] == "development", (
            f"{name} was used to tune vlm-pinpoint and must stay in development, "
            f"found {by_name[name]['role']}"
        )


def test_development_tasks_say_why_they_are_there(tasks):
    for t in tasks:
        if t["role"] == "development":
            assert t.get("why_dev"), f"{t['name']} is in development with no why_dev"


def test_templates_are_declared(roles, tasks):
    declared = {t["id"] for t in roles["templates"]}
    used = {t["template"] for t in tasks}
    assert used <= declared, f"undeclared templates: {sorted(used - declared)}"


def test_only_the_two_large_templates_are_split_across_evaluation_roles(roles, tasks):
    """A template may span roles only if the split is deliberate and declared.

    Sharing a template between validation and test means an object-name swap is being
    counted as independent evidence, which is exactly what the grouping rule forbids.
    """
    allowed = set(roles["meta"].get("templates_split_across_roles", []))
    spans = collections.defaultdict(set)
    for t in tasks:
        if t["role"] in ("validation", "test"):
            spans[t["template"]].add(t["role"])
    offenders = {name for name, r in spans.items() if len(r) > 1 and name not in allowed}
    assert not offenders, (
        f"templates spanning validation and test without a declaration: {offenders}"
    )


def test_out_of_scope_tasks_carry_a_reason(tasks):
    for t in tasks:
        if t["in_scope"] is not True:
            assert t.get("scope_note"), f"{t['name']} is not in scope but gives no reason"


def test_test_set_spans_difficulties(tasks):
    d = collections.Counter(t["difficulty"] for t in tasks if t["role"] == "test")
    assert d["simple"] > 0 and d["moderate"] > 0, d


def test_test_set_contains_out_of_scope_probes(tasks):
    """The frozen set has to include things the harness is expected to fail, or a high
    score means only that the split was flattering."""
    probes = [t for t in tasks if t["role"] == "test" and t["in_scope"] is not True]
    assert len(probes) >= 2, [t["name"] for t in probes]


def test_the_split_covers_exactly_the_benchmark(tasks):
    """Every assigned task exists, and every task in the benchmark has a role.

    Checked against the task class names in NVlabs/RoboLab v0.3.2 (b6cfcf6). A task
    the benchmark does not have fails at run time; a missing task shrinks the aggregate.
    """
    listed = {
        line.strip()
        for line in (ROLES_PATH.parent / "robolab120_tasks.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    ours = {t["name"] for t in tasks}
    assert len(listed) == 120
    assert ours - listed == set(), f"in the split but not in the benchmark: {sorted(ours - listed)}"
    assert listed - ours == set(), f"in the benchmark but unassigned: {sorted(listed - ours)}"
