import abc
from typing import Generic, Sequence, TypeVar, final

from meiga import Error, Result

from inference.core.exceptions import UnrecoverableError
from inference.core.type_hints import ActorResponse


T = TypeVar("T")
V = TypeVar("V")


class BaseActor(abc.ABC, Generic[T, V]):
    """An interface that defines behaviour of all actors."""

    @property
    def batch_size(self) -> int:
        """Get batch size. Default -1.

        -1 means that a data sequence of any length can be processed.

        Returns:
            The batch size.
        """
        return -1

    @abc.abstractmethod
    def _check_input(self, data: Sequence[T]) -> None:
        """Check input data.

        Args:
            data: Input data.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    def _run(self, data: Sequence[T]) -> Sequence[V]:
        """Performs some job.

        Args:
            data: Input data.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    def run(self, data: Sequence[T]) -> Result[Sequence[V], Error]:
        """Performs some job.

        Args:
            data: Input data.

        Returns:
            Successful result or failure.
        """
        try:
            self._check_input(data)
            outputs = self._run(data)
        except UnrecoverableError as e:
            return Result(failure=e)
        return Result(success=outputs)

    @classmethod
    def from_metadata(cls, *args, **kwargs) -> "BaseActor":
        raise NotImplementedError("Method must be implemented in child class.")


class TritonActor(BaseActor, Generic[T, V]):
    """An interface that defines behaviour of all neural network models."""

    @abc.abstractmethod
    def _initialize_state(self) -> None:
        """Prepare initial state.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    def _preprocess(self, data: Sequence[T]) -> Sequence[T]:
        """Preprocess input data.

        Args:
            data: Input data.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    def _postprocess(self, data: Sequence[T], transformed_data: Sequence[T], outputs: ActorResponse) -> Sequence[V]:
        """Postprocess outputs.

        Args:
            data: Input data.
            transformed_data: Transformed data.
            outputs: Actor response.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    async def _run(self, data: Sequence[T]) -> ActorResponse:
        """Performs some job.

        Args:
            data: Input data.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @final
    async def run(self, data: Sequence[T]) -> Result[Sequence[V], Error]:
        """Performs some job.

        Args:
            data: Input data.

        Returns:
            Successful result or failure.
        """
        try:
            self._check_input(data)
            transformed_data = self._preprocess(data)
            outputs = await self._run(transformed_data)
            outputs = self._postprocess(data, transformed_data, outputs)
        except UnrecoverableError as e:
            return Result(failure=e)
        return Result(success=outputs)

    @classmethod
    async def from_metadata(cls, *args, **kwargs) -> "TritonActor":
        raise NotImplementedError("Method must be implemented in child class.")


__all__ = (
    "BaseActor",
    "TritonActor",
)
