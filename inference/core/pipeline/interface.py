import abc
import inspect
from collections import OrderedDict
from typing import Generic, Sequence, TypeVar, final, AsyncIterable, Optional

from meiga import Error, Result
from reretry import retry

from inference.core.actor.base import BaseActor
from inference.core.exceptions import UnrecoverableError
from inference.core.executor.interface import Executor
from inference.settings import runner_policy
from inference.core.pipeline import PIPELINE_REGISTRY


T = TypeVar("T")
V = TypeVar("V")


class BasePipeline(abc.ABC, Generic[T, V]):
    """An interface that defines behaviour of all pipeline."""

    def __init__(self, actor: BaseActor) -> None:
        """Create a new instance of BasePipeline.

        Args:
            actor: An instance of actor class.
        """
        self._actor = actor

    @property
    def max_batch_size(self) -> int:
        """Get max batch size.

        Returns:
            The batch size or -1 if `max_batch_size` was not initialized by the actor.
        """
        return self._actor.batch_size

    @final
    async def run(self, data: Sequence[T]) -> Result[Sequence[V], Error]:
        """Performs some job.

        Args:
            data: Input data.

        Returns:
            Get a result of inference output from ``_actor``.
        """
        result = self._actor.run(data)
        if inspect.isawaitable(result):
            result = await result
        return result

    @staticmethod
    async def from_factory(*args, **kwargs) -> "BasePipeline":
        """Create a new instance of FaceDetectionPipeline."""
        raise NotImplementedError("Method must be implemented in child class.")


class BaseRunner(abc.ABC, Generic[T, V]):
    """An interface that defines behaviour of all runner."""

    def __init__(self, executor: Executor) -> None:
        """Create a new instance of BaseRunner."""
        self._executor = executor
        self._pipeline: OrderedDict[str, BasePipeline] = OrderedDict()

    @final
    def _add_pipeline(self, name: str, pipeline: BasePipeline) -> None:
        """Add a pipeline in the bottom of the pipeline list.

        Args:
            name: A name of ``pipeline``. To debug purposes.
            pipeline: A class:`inference.pipelines.base` instance.
        """
        self._pipeline[name] = pipeline

    async def with_pipeline(self, pipeline_name: str) -> None:
        """Initialize your pipelines here.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        pipeline_type: Optional[BasePipeline] = PIPELINE_REGISTRY.get(pipeline_name, None)
        if pipeline_type is None:
            raise UnrecoverableError(f"Pipeline {pipeline_name} has not been register yet or doesn't exist.")
        pipeline = await pipeline_type.from_factory(self._executor)
        self._add_pipeline(pipeline_name, pipeline)

    @abc.abstractmethod
    async def _run(self, data: Sequence[T]) -> AsyncIterable[Result[Sequence[V], Error]]:
        """Run your pipelines here.

        Args:
            data: Input data.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @final
    @retry((UnrecoverableError,),
           tries=runner_policy.tries,
           delay=runner_policy.delay,
           max_delay=runner_policy.max_delay,
           backoff=runner_policy.backoff,
           jitter=(runner_policy.jitter_min,
                   runner_policy.jitter_max))
    async def run(self, data: Sequence[T]) -> AsyncIterable[V]:
        """Run your pipelines here.

        Use the @RunnerCircuitBreaker decorator to prevent frequent access to a broken inference engine.

        Args:
            data: Input data.

        Returns:
            Output of your pipelines.
        """
        async for collection in self._run(data):
            for item in collection:
                value = item.unwrap_or_throw()
                yield value


__all__ = (
    "BasePipeline",
    "BaseRunner",
)
