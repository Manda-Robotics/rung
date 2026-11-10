"""Wire encoding: msgpack with numpy arrays, as openpi and RoboLab policy servers use.

Arrays use openpi's ``msgpack_numpy`` layout (``__ndarray__``, ``data``, ``dtype``,
``shape``), so a frame packed here is read by ``openpi_client`` and the reverse. Object,
void and complex dtypes are refused: decoding never constructs arbitrary objects.

Harness contracts (observations, targets, reports, capabilities) travel as plain maps
tagged with their class name. Only the classes in ``CONTRACTS`` can be decoded.
"""

from __future__ import annotations

import dataclasses
import functools
from enum import Enum

import msgpack
import numpy as np

from rung.contracts.action import (
    ExecutionOutcome,
    ExecutionReport,
    GripperAction,
    GripperCommand,
    TCPTarget,
)
from rung.contracts.capabilities import (
    CameraSpec,
    EmbodimentCapabilities,
    GripperGeometry,
    Workspace,
)
from rung.contracts.frames import Pose
from rung.contracts.observation import CameraFrame, GripperState, JointState, Observation

MAX_MESSAGE_BYTES = 32 * 1024 * 1024
"""Three 1280 x 720 RGB views plus float32 depth is about 14 MB uncompressed."""

CONTRACTS = {
    cls.__name__: cls
    for cls in (
        Pose,
        CameraFrame,
        GripperState,
        JointState,
        Observation,
        TCPTarget,
        GripperCommand,
        ExecutionReport,
        EmbodimentCapabilities,
        CameraSpec,
        GripperGeometry,
        Workspace,
    )
}
ENUMS = {cls.__name__: cls for cls in (GripperAction, ExecutionOutcome)}


class CodecError(ValueError):
    """The peer sent something this protocol does not accept."""


def _pack_default(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise CodecError(f"unsupported dtype {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": np.ascontiguousarray(obj).tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise TypeError(f"cannot encode {type(obj).__name__}")


def _unpack_hook(obj):
    if b"__ndarray__" in obj:
        dtype = np.dtype(obj[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise CodecError(f"unsupported dtype {dtype}")
        shape = tuple(int(n) for n in obj[b"shape"])
        array = np.frombuffer(obj[b"data"], dtype=dtype)
        if array.size != int(np.prod(shape, dtype=np.int64)):
            raise CodecError("array data does not match its shape")
        return array.reshape(shape).copy()
    if b"__npgeneric__" in obj:
        dtype = np.dtype(obj[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise CodecError(f"unsupported dtype {dtype}")
        return dtype.type(obj[b"data"])
    return obj


def packb(message) -> bytes:
    return msgpack.packb(message, default=_pack_default, use_bin_type=True)


unpackb_raw = functools.partial(
    msgpack.unpackb, object_hook=_unpack_hook, raw=False, strict_map_key=False
)


def unpackb(data: bytes):
    if len(data) > MAX_MESSAGE_BYTES:
        raise CodecError("message too large")
    try:
        return unpackb_raw(data)
    except CodecError:
        raise
    except Exception as exc:  # malformed frames are the peer's error, not a crash
        raise CodecError(f"malformed message: {exc}") from None


# ---- harness contracts ---------------------------------------------------------


def to_wire(value):
    """A contract object (or a structure of them) as plain msgpack-able values."""
    if isinstance(value, Enum):
        return {"__enum__": type(value).__name__, "value": value.value}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in CONTRACTS:
            raise TypeError(f"{name} is not a wire contract")
        out = {"__rung__": name}
        for f in dataclasses.fields(value):
            if f.init:
                out[f.name] = to_wire(getattr(value, f.name))
        return out
    if isinstance(value, dict):
        return {k: to_wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_wire(v) for v in value]
    return value


def from_wire(value):
    """The inverse of ``to_wire``. Unknown tags are an error, never a lookup."""
    if isinstance(value, list):
        return [from_wire(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "__enum__" in value:
        cls = ENUMS.get(value["__enum__"])
        if cls is None:
            raise CodecError(f"unknown enum {value['__enum__']!r}")
        return cls(value["value"])
    if "__rung__" in value:
        cls = CONTRACTS.get(value["__rung__"])
        if cls is None:
            raise CodecError(f"unknown contract {value['__rung__']!r}")
        fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
        kwargs = {}
        for key, raw in value.items():
            if key == "__rung__":
                continue
            if key not in fields:
                raise CodecError(f"{cls.__name__} has no field {key!r}")
            decoded = from_wire(raw)
            if isinstance(decoded, list) and str(fields[key].type).startswith("tuple"):
                decoded = tuple(decoded)
            kwargs[key] = decoded
        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise CodecError(f"invalid {cls.__name__}: {exc}") from None
    return {k: from_wire(v) for k, v in value.items()}
