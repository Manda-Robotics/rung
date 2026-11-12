"""XPolicyLab ``Model`` for RoboDojo that hands every call to Rung's harness.

``setup.sh`` installs this as ``XPolicyLab/policy/Rung/model.py`` beside ``deploy.py`` and
``deploy.yml`` (the launch scripts are copied from ``demo_policy``). It is configured from
the environment so RoboDojo's stock launch scripts stay unchanged:

- ``RUNG_POLICY``: ``scripted`` (a two-arm motion check with no model) or ``direct`` (the
  bimanual Direct profile in ``RUNG_PROFILE``; ``"model": "fake"`` needs no key, any other
  model uses its paid API with the key from the environment, e.g. ``GEMINI_API_KEY``)
- ``RUNG_PROFILE``: profile JSON, default ``configs/gemini-4-argon-robodojo.json``
- ``RUNG_OUT``: output folder for traces, recordings and ``status.json``
- ``RUNG_MAX_CALLS``: model calls (and decisions) per episode
- ``RUNG_HORIZON_S``: seconds of simulated motion the model is told it has; RoboDojo's
  ``step_lim`` (200 actions = 8 s on ``general_pickup``) is what actually ends the episode
- ``RUNG_GRIPPER_ONE_IS_OPEN``: ``1`` (default, measured on the ARX X5) or ``0``
"""

import json
import os
import time
from pathlib import Path

from XPolicyLab.model_template import ModelTemplate

from rung.adapters.robodojo.policies import ScriptedBimanual, direct_policy
from rung.adapters.robodojo.rig import RoboDojoCodec
from rung.adapters.robodojo.server import RungRoboDojoModel
from rung.loop import Limits

RUNG_ROOT = Path(os.environ.get("RUNG_ROOT", "/workspace/rung"))


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.out = Path(os.environ.get("RUNG_OUT", "/workspace/out/robodojo"))
        self.out.mkdir(parents=True, exist_ok=True)
        kind = os.environ.get("RUNG_POLICY", "scripted")
        if kind not in ("scripted", "direct"):
            raise ValueError(f"RUNG_POLICY must be scripted or direct, not {kind!r}")
        calls = int(os.environ.get("RUNG_MAX_CALLS", "12"))
        horizon = float(os.environ.get("RUNG_HORIZON_S", "8"))
        one_is_open = os.environ.get("RUNG_GRIPPER_ONE_IS_OPEN", "1") == "1"
        profile = os.environ.get(
            "RUNG_PROFILE", str(RUNG_ROOT / "configs" / "gemini-4-argon-robodojo.json")
        )

        def factory(emb):
            if kind == "scripted":
                return ScriptedBimanual(log=lambda *a: print(*a, flush=True))
            return direct_policy(
                profile, horizon_s=horizon, max_calls=calls, trace_root=self.out / "model"
            )

        def codec(**kw):
            return RoboDojoCodec(gripper_one_is_open=one_is_open, **kw)

        self.rung = RungRoboDojoModel(
            factory,
            codec_factory=codec,
            record_dir=self.out / "rung",
            limits=Limits(max_decisions=calls, horizon_s=horizon, max_model_calls=calls),
        )
        self._reported = False
        self._status("ready", policy=kind, profile=profile if kind == "direct" else None)

    def _status(self, state, **extra):
        status = {"state": state, "t": time.time(), **extra}
        (self.out / "status.json").write_text(json.dumps(status, indent=2))

    def update_obs(self, obs):
        self.rung.update_obs(obs)

    def get_action(self):
        actions = self.rung.get_action()
        session = self.rung.session
        if session is not None and not session[2].is_alive() and not self._reported:
            self._reported = True  # the harness episode ended; RoboDojo plays on to step_lim
            self._report("episode_done")
        return actions

    def reset(self):
        self.rung.reset()
        self._reported = False
        self._report("running")

    def _report(self, state):
        self._status(
            state,
            episode=self.rung.episode,
            results={
                k: {
                    "end_reason": r.end_reason.value,
                    "decisions": r.decisions,
                    "model_calls": r.usage.calls,
                    "usd": r.usage.dollars,
                }
                for k, r in self.rung.results.items()
            },
            errors={k: repr(e) for k, e in self.rung.errors.items()},
        )

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("Rung runs one environment; set eval_batch: false")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("Rung runs one environment; set eval_batch: false")
