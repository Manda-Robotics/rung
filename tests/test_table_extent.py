"""Declared table footprint: state field, edge receipt and the RoboLab rig's values."""

import json

import numpy as np

from rung.adapters.robolab.translate import TABLE_EXTENT_ROOT_M
from rung.contracts.capabilities import Workspace
from rung.interfaces.direct import DirectConfig, DirectPolicy, table_edge_report
from rung.mock import SCENES
from rung.policy import Feedback
from tests.test_direct_session import reply, session


def test_robolab_table_extent_matches_the_fixture_and_the_workspace():
    x_min, x_max, y_min, y_max = TABLE_EXTENT_ROOT_M
    assert abs((x_max - x_min) - 0.900) < 1e-3 and abs((y_max - y_min) - 0.759) < 2e-3
    ws = Workspace(
        np.array([0.15, -0.55, 0.0]), np.array([0.90, 0.55, 0.75]), 0.0, TABLE_EXTENT_ROOT_M
    )
    assert ws.table_margin([0.4, 0.0, 0.1]) > 0.3  # centre of the table
    # the mug release that fell, 2026-10-23
    assert abs(ws.table_margin([0.4, 0.486, 0.2]) + 0.107) < 1e-3
    assert ws.table_margin([0.449, -0.382, 0.23]) < 0  # the spatula release that fell
    assert Workspace(np.zeros(3), np.ones(3), 0.0).table_margin([0, 0, 0]) is None


def test_edge_report_warns_near_and_beyond_the_edge_only():
    ws = Workspace(
        np.array([0.15, -0.55, 0.0]), np.array([0.90, 0.55, 0.75]), 0.0, TABLE_EXTENT_ROOT_M
    )
    assert "warning" not in table_edge_report([0.4, 0.0, 0.1], ws)
    near = table_edge_report([0.4, 0.33, 0.1], ws)
    assert "from the table edge" in near["warning"] and near["tcp_margin_m"] == 0.049
    beyond = table_edge_report([0.4, 0.45, 0.1], ws)
    assert "beyond the table edge" in beyond["warning"]


def policy(replies, extent=None, **config):
    from dataclasses import replace

    emb = SCENES["banana_in_bowl"]()
    s, fake = session(replies)
    p = DirectPolicy(s, DirectConfig(**config))
    caps = emb.capabilities
    if extent is not None:
        caps = replace(caps, workspace=replace(caps.workspace, table_extent=extent))
    p.reset("test task", caps)
    return emb, p, fake


def test_table_extent_reaches_state_prompt_and_receipt_when_the_rig_declares_one():
    emb, p, fake = policy([reply(), reply()], extent=TABLE_EXTENT_ROOT_M, table_extent=True)
    p.act(emb.observe(), Feedback())
    first = json.dumps(fake.requests[0])
    assert "table_extent_m" in first and "falls to the floor" in first
    p.act(emb.observe(), Feedback())
    second = json.dumps(fake.requests[1])
    assert "table_edge" in second and "tcp_margin_m" in second


def test_table_extent_option_is_silent_when_the_rig_declares_none():
    emb, p, fake = policy([reply(), reply()], table_extent=True)
    assert emb.capabilities.workspace.table_extent is None
    p.act(emb.observe(), Feedback())
    p.act(emb.observe(), Feedback())
    assert "table_extent_m" not in json.dumps(fake.requests[0])
    assert "table_edge" not in json.dumps(fake.requests[1])


def test_table_extent_off_by_default():
    emb, p, fake = policy([reply(), reply()])
    p.act(emb.observe(), Feedback())
    p.act(emb.observe(), Feedback())
    assert "table_extent_m" not in json.dumps(fake.requests[0])
    assert "table_edge" not in json.dumps(fake.requests[1])
