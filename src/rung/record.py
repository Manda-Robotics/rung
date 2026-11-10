"""Recording: everything the policy saw, kept at full fidelity.

The rule is that a saved episode can be audited later for grounding, occlusion, prompt
rendering and segmentation failures. Thumbnails cannot answer those questions, so images
are stored at the resolution the policy received, losslessly, for every evaluated episode.

Layout::

    <root>/<episode_id>/
        obs_0000_scene.png
        obs_0000_wrist.png
        obs_0000.json          state, capture times, camera calibration
        prompt_0000.txt        exactly what was sent, when a model policy is driving
        result.json
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rung.contracts.observation import Observation
from rung.contracts.trace import _Encoder

if TYPE_CHECKING:
    from rung.loop import EpisodeResult


class Recorder:
    """Writes per-decision inputs. ``images=False`` keeps the JSON but skips the pixels,
    for development loops where the disk cost is not worth it."""

    def __init__(self, root: str | Path, images: bool = True):
        self.root = Path(root)
        self.images = images
        self.root.mkdir(parents=True, exist_ok=True)

    def episode_dir(self, episode_id: str) -> Path:
        d = self.root / episode_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def record_observation(self, episode_id: str, index: int, obs: Observation) -> None:
        d = self.episode_dir(episode_id)
        meta: dict[str, Any] = {
            "index": index,
            "t": obs.t,
            "tcp_pos": obs.tcp.pos,
            "tcp_quat": obs.tcp.quat,
            "gripper": dataclasses.asdict(obs.gripper),
            "cameras": {},
        }
        for name, cam in obs.cameras.items():
            meta["cameras"][name] = {
                "t": cam.t,
                "width": cam.width,
                "height": cam.height,
                "K": cam.K,
                "pose_pos": None if cam.pose is None else cam.pose.pos,
                "pose_quat": None if cam.pose is None else cam.pose.quat,
                "convention": cam.convention,
                "has_depth": cam.depth is not None,
            }
            if self.images:
                self._write_png(d / f"obs_{index:04d}_{name}.png", cam.rgb)
                if cam.depth is not None:
                    np.save(d / f"obs_{index:04d}_{name}_depth.npy", cam.depth.astype(np.float16))
        (d / f"obs_{index:04d}.json").write_text(
            json.dumps(meta, cls=_Encoder, indent=2), encoding="utf-8"
        )

    def record_prompt(self, episode_id: str, index: int, prompt: str) -> None:
        (self.episode_dir(episode_id) / f"prompt_{index:04d}.txt").write_text(
            prompt, encoding="utf-8"
        )

    def record_json(self, episode_id: str, name: str, value) -> None:
        (self.episode_dir(episode_id) / name).write_text(
            json.dumps(value, cls=_Encoder, indent=1), encoding="utf-8"
        )

    def record_result(self, result: EpisodeResult) -> None:
        (self.episode_dir(result.episode_id) / "result.json").write_text(
            json.dumps(dataclasses.asdict(result), cls=_Encoder, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _write_png(path: Path, rgb: np.ndarray) -> None:
        from PIL import Image

        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path, format="PNG", optimize=False)
