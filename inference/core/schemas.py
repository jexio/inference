from enum import Enum
from dataclasses import dataclass, field
from functools import reduce
from typing import Tuple

import numpy as np


class TensorType(Enum):
    """Tensor datatype."""

    TYPE_BOOL = np.bool_
    TYPE_UINT8 = np.uint8
    TYPE_UINT16 = np.uint16
    TYPE_UINT32 = np.uint32
    TYPE_UINT64 = np.uint64
    TYPE_INT8 = np.int8
    TYPE_INT16 = np.int16
    TYPE_INT32 = np.int32
    TYPE_INT64 = np.int64
    TYPE_FP16 = np.float16
    TYPE_FP32 = np.float32
    TYPE_FP64 = np.float64
    TYPE_STRING = np.object_


@dataclass(frozen=True)
class TritonExecutorInput:
    """DTO class Triton input configuration.

    Attributes:
        name: The name of input whose data will be described by this object.
        shape: The shape of the associated input.
        datatype: The datatype of the associated input.
        bytes: Total bytes.
    """

    name: str = field(default="INPUT0")
    shape: Tuple[int, ...] = field(default_factory=tuple)
    datatype: Enum = field(default=TensorType.TYPE_UINT8)
    size: int = field(default=0)

    def __post_init__(self) -> None:
        """Compute size of array of `shape` in bytes."""
        b = np.dtype(self.datatype.value).itemsize
        b = int(reduce(lambda x, y: x * y, [b, *self.shape]))
        object.__setattr__(self, "size", b)


__all__ = ("TensorType", "TritonExecutorInput")
