"""Rung as an XPolicyLab policy model, served to RoboDojo's simulator over its websocket.

RoboDojo runs ``deploy.eval_one_episode`` inside the simulator; that loop calls the
policy server's ``reset``, then ``update_obs(obs)`` and ``get_action()`` alternately
until the task's ``is_episode_end``. This model is that server-side object. It starts a
harness episode on the first observation after ``reset`` and answers each ``get_action``
with ONE action dict, so every simulator step goes through the bridge's closed loop.

RoboDojo never tells the server that an episode ended (success, or ``step_lim``); the
next ``reset`` closes the previous harness episode. The grade lives only in RoboDojo's
``_result.json``. No Isaac or XPolicyLab import here; the XPolicyLab shim in
``deploy/robodojo/policy/Rung/model.py`` subclasses this.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from rung.adapters.bimanual import BimanualDriver, BimanualEmbodiment, run_in_thread
from rung.adapters.robodojo.rig import RoboDojoCodec
from rung.contracts.trace import TraceWriter
from rung.loop import Limits, run_episode
from rung.record import Recorder


def measure_table_z(
    scene, declared: float, window_m: float = 0.08, min_share: float = 0.3
) -> float | None:
    """The table height from the head camera's depth, searched near the declared height.

    Rung's ``estimate_table_z`` looks where a DROID table is (x near 0.5 m, z between -0.1
    and 0.4 m) and returns its fallback when it finds nothing; on RoboDojo, whose table is
    at 0.765 m, that fallback (0.0) collapsed the workspace below the arms on the first
    live run. Here the histogram is taken within ``window_m`` of the declared height, and
    None (keep the declared value) is returned unless at least ``min_share`` of those
    points lie within 1 cm of the peak.
    """
    from rung.contracts.frames import depth_image_to_points

    if scene.depth is None or scene.K is None or scene.pose is None:
        return None
    pts, _ = depth_image_to_points(scene.depth, scene.K, scene.pose, stride=4)
    pts = pts[np.isfinite(pts).all(axis=1)]
    near = pts[np.abs(pts[:, 2] - declared) < window_m]
    if len(near) < 200:
        return None
    hist, edges = np.histogram(
        near[:, 2], bins=80, range=(declared - window_m, declared + window_m)
    )
    peak = float(edges[int(np.argmax(hist))] + (edges[1] - edges[0]) / 2)
    band = near[np.abs(near[:, 2] - peak) < 0.01]
    if len(band) < min_share * len(near):
        return None
    return float(np.median(band[:, 2]))


class RungRoboDojoModel:
    def __init__(
        self,
        policy_factory: Callable,
        *,
        codec_factory: Callable | None = None,
        limits: Limits | None = None,
        record_dir: str | Path | None = None,
        time_mode: str = "paused",
        measure_table: bool = True,
    ):
        self.policy_factory = policy_factory
        self.codec_factory = codec_factory or RoboDojoCodec
        self.limits = limits or Limits(max_decisions=40, horizon_s=60.0, max_model_calls=40)
        self.record_dir = Path(record_dir) if record_dir else None
        self.time_mode = time_mode
        self.measure_table = measure_table
        self.episode = -1
        self.session = None
        self.results, self.errors = {}, {}
        self._obs = None

    def reset(self, *_):
        self._close()
        self.episode += 1
        self._obs = None

    def update_obs(self, obs):
        self._obs = obs
        if self.session is None:
            self._start(obs)

    def get_action(self, *_):
        if self._obs is None:
            raise RuntimeError("update_obs must precede get_action")
        _, driver, _, name = self.session
        action = driver.get_action(self._obs)
        if name in self.errors:
            raise RuntimeError(f"harness failed for {name}") from self.errors[name]
        return [action]

    def _codec(self, obs):
        codec = self.codec_factory()
        if not self.measure_table:
            return codec
        try:
            scene = codec.observation(obs, 0.0).scene
        except Exception:
            return codec
        measured = measure_table_z(scene, codec.capabilities.workspace.table_z)
        return codec if measured is None else self.codec_factory(table_z=measured)

    def _start(self, obs):
        codec = self._codec(obs)
        emb = BimanualEmbodiment(codec.capabilities, time_mode=self.time_mode)
        instruction = RoboDojoCodec.unwrap(obs).get("instruction") or ""
        emb.instruction = instruction
        driver = BimanualDriver(emb, codec)
        policy = self.policy_factory(emb)
        name = f"episode-{self.episode:04d}"
        folder = self.record_dir / name if self.record_dir else None
        trace = TraceWriter(folder / "trace.jsonl" if folder else None)
        trace.emit(
            "setup",
            0.0,
            name,
            embodiment=codec.capabilities.name,
            arms=[a.name for a in codec.capabilities.arms],
            table_z=codec.capabilities.workspace.table_z,
            time_mode=self.time_mode,
            instruction=instruction,
        )
        recorder = Recorder(self.record_dir) if folder else None

        def drive():
            try:
                self.results[name] = run_episode(
                    emb,
                    policy,
                    task=instruction,
                    episode_id=name,
                    limits=self.limits,
                    trace=trace,
                    recorder=recorder,
                )
            except BaseException as exc:
                self.errors[name] = exc
            finally:
                emb.close()

        self.session = (emb, driver, run_in_thread(drive), name)

    def _close(self):
        if self.session is None:
            return
        emb, _, thread, _ = self.session
        emb.close()
        thread.join(timeout=10)
        self.session = None
