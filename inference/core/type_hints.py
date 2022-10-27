"""Mypy stubs."""
import pathlib
from typing import TYPE_CHECKING, Any, Dict, Union

import numpy as np


TPath = Union[str, pathlib.Path]


if TYPE_CHECKING:
    import numpy.typing as npt

    FloatScalar = npt.NDArray[np.float_]
    NPType = npt.Any
    NDArray = npt.NDArray[Any]
    ImageNDArray = npt.NDArray[npt.int_]
    BatchedNDArray = npt.NDArray[Any]
else:
    import nptyping as npt

    FloatScalar = npt.Float
    NDArray = npt.NDArray[npt.Shape["Size"], Any]
    ImageNDArray = npt.NDArray[npt.Shape["Height, Width, 3"], npt.UInt8]
    BatchedNDArray = npt.NDArray[npt.Shape["Batch, ..."], Any]

ActorResponse = Dict[str, BatchedNDArray]

__all__ = (
    "NPType",
    "NDArray",
    "ImageNDArray",
    "BatchedNDArray",
    "FloatScalar",
    "TPath",
    "ActorResponse",
)
