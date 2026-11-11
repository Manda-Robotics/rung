"""Which RoboLab tree is installed, and the Isaac Lab helpers Rung needs from it.

Rung runs against two evaluator trees:

* RoboLab-Verified (the fork) ships ``robolab.core.utils.isaaclab_compat``, which
  converts between RoboLab's stable WXYZ quaternions and Isaac Lab 3's XYZW. When that
  module is importable it is used unchanged, so the Verified path behaves exactly as
  before.
* Upstream NVlabs/RoboLab v0.3.2 (b6cfcf6) has no such module and supports only Isaac
  Sim 5.0/5.1 with Isaac Lab 2.x, whose quaternions are already WXYZ. There the
  conversions are identities, provided here with the same names and signatures.
* Upstream main 9cce00c (v0.3.2 plus the merged PR #47, Isaac Sim 6.0.1 / Isaac Lab 3; the
  pre-merge PR head d839ddf likewise) ships the same ``isaaclab_compat`` module as the fork,
  so the module alone does not identify the tree.
  The fork is recognised by a fork-only module (``robolab.core.task.confirm``, which
  confirms success); a tree with the compat module but without it is upstream, and its
  own compat module is used.

Nothing here imports Isaac Lab, robolab or torch at import time, so the module is safe
to import before ``AppLauncher`` starts Kit and in tests without Isaac.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from types import SimpleNamespace
from typing import Any

FORK_MODULE = "robolab.core.utils.isaaclab_compat"
# Fork-only module (success confirmation). Present in every RoboLab-Verified tree Rung
# runs (frozen 12d60f3 and later), absent from upstream v0.3.2, PR #47 and main 9cce00c.
FORK_MARKER = "robolab.core.task.confirm"
VERIFIED = "verified"
UPSTREAM = "upstream"
# grade_detail["source"] per tree. The Verified string is the one Rung always wrote.
GRADER_SOURCE = {VERIFIED: "RoboLab-Verified", UPSTREAM: "RoboLab-upstream"}


def _import_optional(module: str):
    """The module, or None when the installed tree does not have it.

    Only the absence of the module itself (or of robolab altogether) counts as absent;
    any other import failure inside an existing module is a real error and raises.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        parts = module.split(".")
        missing = {".".join(parts[: i + 1]) for i in range(len(parts))}
        if exc.name in missing:
            return None
        raise


def _import_fork():
    """The installed tree's ``isaaclab_compat`` module (fork, or upstream with PR #47), or None."""
    return _import_optional(FORK_MODULE)


def robolab_tree() -> str:
    """``"verified"`` for the fork (compat module and fork-only marker), else ``"upstream"``."""
    if _import_fork() is None:
        return UPSTREAM
    return VERIFIED if _import_optional(FORK_MARKER) is not None else UPSTREAM


def _isaaclab_major() -> int | None:
    try:
        return int(importlib.metadata.version("isaaclab").split(".", 1)[0])
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return None


def _require_isaaclab2():
    major = _isaaclab_major()
    if major is not None and major >= 3:
        raise RuntimeError(
            "Isaac Lab 3 uses XYZW quaternions; the Rung shim covers Isaac Lab 2.x only. "
            "Install a RoboLab tree that provides robolab.core.utils.isaaclab_compat."
        )


def as_torch(value: Any):
    """A torch tensor view of a tensor, NumPy array, sequence, ProxyArray or Warp array."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(value)
    tensor = getattr(value, "torch", None)
    if isinstance(tensor, torch.Tensor):
        return tensor
    try:
        import warp as wp

        return wp.to_torch(value)
    except (ImportError, RuntimeError, TypeError) as exc:
        raise TypeError(f"Cannot convert {type(value).__name__} to torch.Tensor") from exc


def quat_wxyz_to_isaaclab(value: Any, *, uses_xyzw: bool | None = None):
    """Isaac Lab 2.x already uses WXYZ: identity."""
    if uses_xyzw:
        raise ValueError("the Isaac Lab 2.x shim has no XYZW convention")
    _require_isaaclab2()
    return value


def quat_isaaclab_to_wxyz(value: Any, *, uses_xyzw: bool | None = None):
    """Isaac Lab 2.x already uses WXYZ: identity."""
    if uses_xyzw:
        raise ValueError("the Isaac Lab 2.x shim has no XYZW convention")
    _require_isaaclab2()
    return value


def _copy_scene_state(state: dict) -> dict:
    """The fork's tree copy with an identity pose conversion: new dicts, same leaves."""
    _require_isaaclab2()
    copied = {}
    for family, assets in state.items():
        if not isinstance(assets, dict):
            copied[family] = assets
            continue
        copied[family] = {
            name: dict(asset) if isinstance(asset, dict) else asset
            for name, asset in assets.items()
        }
    return copied


def scene_state_to_isaaclab(state: dict) -> dict:
    return _copy_scene_state(state)


def scene_state_from_isaaclab(state: dict) -> dict:
    return _copy_scene_state(state)


SHIM = SimpleNamespace(
    ISAACLAB_USES_XYZW=False,
    as_torch=as_torch,
    quat_wxyz_to_isaaclab=quat_wxyz_to_isaaclab,
    quat_isaaclab_to_wxyz=quat_isaaclab_to_wxyz,
    scene_state_to_isaaclab=scene_state_to_isaaclab,
    scene_state_from_isaaclab=scene_state_from_isaaclab,
    __name__="rung.adapters.robolab.compat.SHIM",
)


def isaaclab_compat():
    """The tree's ``isaaclab_compat`` module when installed (fork, or upstream with PR #47),
    else the Isaac Lab 2.x shim."""
    fork = _import_fork()
    return fork if fork is not None else SHIM


def instruction_text(instruction) -> str:
    """The primary instruction string from a string or a (tracked) field mapping.

    Upstream v0.3.2 hands clients a ``TrackedInstructionFields`` dict. Its own
    ``instruction_text`` is used when importable, so field use is recorded exactly as
    upstream records it; the fallback mirrors it for trees and tests without it.
    """
    if isinstance(instruction, str):
        return instruction
    try:
        from robolab.core.task.instruction import instruction_text as upstream
    except ImportError:
        upstream = None
    if upstream is not None:
        return upstream(instruction)
    if not instruction:
        raise ValueError("instruction mapping must be nonempty")
    name = "default" if "default" in instruction else next(iter(instruction))
    value = instruction[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"instruction field '{name}' must be a nonempty string")
    return value


class ReResetLoop(RuntimeError):
    """Upstream re-reset an episode more often than the guard allows."""


class ReResetGuard:
    """Catch upstream's uncapped early re-reset (``env.py:95-101`` at v0.3.2).

    Upstream resets any env that terminates within two steps and carries on, without a
    cap or a record. A start whose success (or failure) predicate holds at once loops
    there until the native step limit, with the policy driving a scene that keeps
    snapping back. ``episode_length_buf`` grows by one per step and drops to zero on a
    reset, so a value that fails to grow between two inference calls is a re-reset.
    One re-reset is upstream's intended artifact handling; more than ``max_resets`` in
    one episode ends it as infrastructure.
    """

    def __init__(self, max_resets: int = 2):
        if max_resets < 0:
            raise ValueError("max_resets must be non-negative")
        self.max_resets = max_resets
        self.start()

    def start(self):
        self.previous: dict[int, int] = {}
        self.resets: dict[int, int] = {}

    def observe(self, env, env_ids) -> None:
        buffer = env.episode_length_buf
        for env_id in env_ids:
            value = buffer[env_id]
            length = int(value.item() if hasattr(value, "item") else value)
            previous = self.previous.get(env_id)
            self.previous[env_id] = length
            if previous is not None and length <= previous:
                self.resets[env_id] = self.resets.get(env_id, 0) + 1
                if self.resets[env_id] > self.max_resets:
                    raise ReResetLoop(
                        f"env {env_id} re-reset {self.resets[env_id]} times within its first "
                        f"steps (upstream early re-reset loop; cap {self.max_resets})"
                    )
