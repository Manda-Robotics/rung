"""The probes are part of the gate: they must pass on the mock, and they must fail loudly
on a broken adapter rather than passing quietly."""

from __future__ import annotations

import numpy as np

from rung.contracts.action import ExecutionReport, TCPTarget
from rung.mock import SCENES
from rung.probes import (
    probe_camera_geometry,
    probe_commanded_equals_travelled,
    probe_depth_reconstructs_the_table,
    run_probes,
)


def test_all_probes_pass_on_every_mock_scene():
    for name, make in SCENES.items():
        results = run_probes(make())
        failed = [r.name for r in results if not r.passed]
        assert not failed, f"{name}: {failed}"


class _HalfScaleArm:
    """An adapter with the classic bug: it moves half as far as it was told."""

    def __init__(self):
        self._inner = SCENES["banana_in_bowl"]()

    @property
    def capabilities(self):
        return self._inner.capabilities

    def reset(self, episode_seed=None):
        return self._inner.reset(episode_seed)

    def observe(self):
        return self._inner.observe()

    def execute(self, target: TCPTarget) -> ExecutionReport:
        here = self._inner.observe().tcp.pos
        halfway = here + (np.asarray(target.pose.pos) - here) * 0.5
        from rung.contracts.frames import Pose

        return self._inner.execute(
            TCPTarget(
                pose=Pose(halfway, target.pose.quat),
                gripper=target.gripper,
                max_lin_speed_mps=target.max_lin_speed_mps,
                max_ang_speed_rps=target.max_ang_speed_rps,
                timeout_s=target.timeout_s,
                id=target.id,
            )
        )


def test_scale_bug_is_caught():
    r = probe_commanded_equals_travelled(_HalfScaleArm())
    assert not r.passed
    assert r.measured["ratios"][0] < 0.6


class _OpenGLCamera:
    """An adapter that forgot to convert its camera convention."""

    def __init__(self):
        self._inner = SCENES["banana_in_bowl"]()

    @property
    def capabilities(self):
        return self._inner.capabilities

    def reset(self, episode_seed=None):
        return self._flip(self._inner.reset(episode_seed))

    def observe(self):
        return self._flip(self._inner.observe())

    def execute(self, target):
        return self._inner.execute(target)

    @staticmethod
    def _flip(obs):
        from rung.contracts.frames import Pose, rot_to_quat

        cam = obs.scene
        r = cam.pose.rot @ np.diag([1.0, -1.0, -1.0])
        cam.pose = Pose(cam.pose.pos, rot_to_quat(r))
        return obs


def test_unconverted_camera_convention_is_caught():
    emb = _OpenGLCamera()
    assert not probe_camera_geometry(emb).passed
    assert not probe_depth_reconstructs_the_table(emb).passed
