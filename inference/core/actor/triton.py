import abc
import warnings
from typing import ClassVar, Dict, Optional, Sequence, Tuple, TypeVar, Union

from inference.core.actor import ACTOR_REGISTRY
from inference.core.actor.exceptions import InvalidBatchSize, InvalidShapeException
from inference.core.actor.base import TritonActor
from inference.core.actor.schemas import BoundingBox, Embedding, Image, ActorImageParameters
from inference.core.executor.interface import Executor
from inference.core.schemas import TensorType, TritonExecutorInput
from inference.core.type_hints import ActorResponse


TImage = TypeVar("TImage", bound=Image)
VImage = TypeVar("VImage", bound=Union[BoundingBox, Embedding])


class TritonImageActor(TritonActor[TImage, VImage], metaclass=abc.ABCMeta):
    """An interface that defines behaviour of all neural network models related to image data."""

    _max_batch_size: ClassVar[int] = -1
    _axes: ClassVar[int] = 3
    __slots__ = (
        "_parameters",
        "_executor",
        "_model",
        "_version",
        "_datatype",
        "_input_names",
        "_output_names",
        "_model_metadata",
        "_model_config",
    )

    def __init__(self, executor: Executor, parameters: ActorImageParameters) -> None:
        """Create a new instance of TritonImageActor."""
        self._parameters = parameters
        self._executor: Executor = executor
        self._model: Optional[str] = None
        self._version: Optional[str] = None
        self._datatype: Optional[TensorType] = None
        self._input_names: Optional[Tuple[str, ...]] = None
        self._output_names: Optional[Tuple[str, ...]] = None
        self._model_metadata: Optional[Dict[str, str]] = None
        self._model_config: Optional[Dict[str, str]] = None
        self._initialize_state()

    @property
    def batch_size(self) -> int:
        """Get the batch size.

        Returns:
              Batch size of the Actor.

        Raises:
            InvalidBatchSize: if a batch size does not belong to the ranges [-1; 0), (0; +inf].
        """
        bs = self._model_metadata.get("max_batch_size", self._max_batch_size)
        if bs == 0 or bs < -1:
            raise InvalidBatchSize(f"Expected - ranges [-1; 0), (0; +inf] , but got - {bs}.")
        return bs

    async def _initialize_model_configuration(self) -> None:
        """Initialize inference engine here."""
        self._model_metadata = await self._executor.get_model_metadata(self._model, self._version)
        self._model_config = await self._executor.get_model_config(self._model, self._version)

    def _check_state(self) -> None:
        """Check configuration here.

        Raises:
            InvalidShapeException: if actor's batch size larger than model's batch size.
        """
        # If the model is less, raise error. Cant send more to Triton than the max batch size
        model_batch_size = self._model_metadata.get("max_batch_size", -1)
        if model_batch_size != -1 and self._max_batch_size > model_batch_size:
            raise InvalidShapeException(
                f"Model max batch size ({model_batch_size}) "
                f"is less than configured max batch size ({self._max_batch_size}). "
                "Reduce max batch size to be less than or equal to model max batch size."
            )
        # If the model is more, thats fine. Gen warning
        if model_batch_size != -1 and self._max_batch_size < model_batch_size:
            warnings.warn(
                f"Model max batch size ({model_batch_size}) "
                f"is more than configured max batch size ({self._max_batch_size}). "
                "May result in sub optimal performance"
            )

    def _check_input(self, data: Sequence[TImage]) -> None:
        """Check input data.

        Args:
            data: Raw images.

        Raises:
            InvalidShapeException: If the number of axes does not match `_axes` or length of `data` less that `_max_batch_size`.
        """
        batch_size: int = self._model_metadata.get("max_batch_size", -1)
        if len(data) > batch_size != -1:
            raise InvalidShapeException(
                f"Number of elements - {len(data)}" f" greater than `max_batch_size` - {batch_size}."
            )

        available_shape = self._model_metadata["inputs"][0]["shape"]
        for item in data:
            shape = item.shape
            if len(shape) != self._axes:
                raise InvalidShapeException(f"Excepted - {available_shape}, but got {shape}")

    async def _run(self, data: Sequence[TImage]) -> ActorResponse:
        """Performs neural networks output.

        Args:
            data: Input data.

        Returns:
            Executor's response.
        """
        inputs = self._prepare_inputs(data)
        outputs = await self._executor.run(
            self._model,
            self._version,
            ([item.to_tensor() for item in data],),
            inputs,
            self._output_names,
        )
        return outputs

    def _prepare_inputs(self, data: Sequence[TImage]) -> Tuple[TritonExecutorInput]:
        """Prepare user input to :class:`inference.core.executor.triton.TritonExecutor` executor.

        Args:
            data: Input data.

        Returns:
            Prepared inputs to the triton executor.
        """
        batch_size = len(data)
        inputs: Tuple[TritonExecutorInput] = tuple(
            [
                TritonExecutorInput(
                    name=name,
                    shape=(batch_size, self._parameters.channels, self._parameters.height, self._parameters.width),
                    datatype=self._datatype,
                )
                for name in self._input_names
            ]
        )

        return inputs

    @classmethod
    async def from_metadata(cls, executor: Executor, parameters: ActorImageParameters) -> "TritonActor":
        actor = cls(executor, parameters)
        await actor._initialize_model_configuration()
        actor._check_state()
        return actor


class TritonActorFactory:

    @staticmethod
    async def create(name: str, executor: Executor, parameters: ActorImageParameters) -> TritonActor:
        instance = await ACTOR_REGISTRY[name].from_metadata(executor, parameters)
        return instance


__all__ = ("TritonImageActor", "TritonActorFactory")
