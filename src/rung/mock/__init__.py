"""A dependency-free tabletop embodiment used by every test and by the probes."""

from rung.mock.embodiment import SCENES, MockConfig, MockEmbodiment
from rung.mock.world import Body, Camera, GripperModel, Scene, look_at, top_down_pose

__all__ = [
    "SCENES",
    "Body",
    "Camera",
    "GripperModel",
    "MockConfig",
    "MockEmbodiment",
    "Scene",
    "look_at",
    "top_down_pose",
]
