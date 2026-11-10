"""Wire codec of the hosted-API client: arrays, contracts and refusals."""

import numpy as np
import pytest

pytest.importorskip("msgpack")

from rung.api.codec import CodecError, from_wire, packb, to_wire, unpackb  # noqa: E402
from rung.mock import SCENES  # noqa: E402


def test_arrays_use_the_openpi_msgpack_numpy_layout():
    frame = packb({"actions": np.arange(14, dtype=np.float32).reshape(2, 7)})
    import msgpack

    raw = msgpack.unpackb(frame, raw=False, strict_map_key=False)
    assert raw["actions"][b"__ndarray__"] is True
    assert raw["actions"][b"dtype"] == "<f4" and list(raw["actions"][b"shape"]) == [2, 7]
    back = unpackb(frame)["actions"]
    assert back.dtype == np.float32 and back.shape == (2, 7) and back[1, 6] == 13


def test_object_arrays_and_unknown_tags_are_refused():
    with pytest.raises(CodecError):
        packb({"x": np.array([object()], dtype=object)})
    with pytest.raises(CodecError, match="unknown contract"):
        from_wire({"__rung__": "Popen", "args": ["rm"]})
    with pytest.raises(CodecError, match="malformed"):
        unpackb(b"\xc1not msgpack")


def test_observation_and_capabilities_survive_the_wire():
    emb = SCENES["banana_in_bowl"]()
    obs = emb.reset()
    back = from_wire(unpackb(packb(to_wire(obs))))
    assert np.array_equal(back.scene.rgb, obs.scene.rgb)
    assert np.allclose(back.scene.depth, obs.scene.depth, equal_nan=True)
    assert np.allclose(back.tcp.pos, obs.tcp.pos) and back.gripper.travel == obs.gripper.travel
    caps = from_wire(unpackb(packb(to_wire(emb.capabilities))))
    assert caps.cameras == emb.capabilities.cameras
    assert caps.workspace.table_z == emb.capabilities.workspace.table_z
