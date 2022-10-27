from typing import ClassVar, Sequence, List, AsyncIterable

import asyncer
from asyncer import SoonValue
from meiga import Error, Result

from inference.core.actor.triton import TritonActorFactory
from inference.core.actor.schemas import (Image, BoundingBox, ActorImageParameters)
from inference.core.executor import Executor
from inference.core.pipeline import register_pipeline
from inference.core.pipeline.interface import BasePipeline, BaseRunner
from inference.core.utils.iter import chunkify

RunnerOutput = Sequence[BoundingBox]


@register_pipeline("face_detection_scrface")
class FaceDetectionPipeline(BasePipeline[Image, Sequence[BoundingBox]]):
    """Pipeline for detecting faces on images.

    Examples:
            .. code::python
                >>> data = [Image.from_base64(...), Image.from_base64(...)]
                >>> executor = ...
                >>> pipeline = await FaceDetectionPipeline.from_factory(executor)
                >>> r = pipeline.run(data)
    """
    _name: ClassVar[str] = "scrface"

    @staticmethod
    async def from_factory(executor: Executor) -> BasePipeline:
        """Create a new instance of FaceDetectionPipeline."""
        actor_parameters = ActorImageParameters(height=768, width=768, channels=3)
        actor = await TritonActorFactory.create(FaceDetectionPipeline._name, executor, actor_parameters)
        return BasePipeline(actor)


class FaceDetectionPipelineRunner(BaseRunner[Image, Sequence[BoundingBox]]):
    """Wrapper around the face detection task.

    Run an image data through the pipelines.
    Here is the sequence of using the pipelines: image -> detector

    Examples:
            .. code::python
                >>> data = [Image.from_base64(...), Image.from_base64(...)]
                >>> executor = ...
                >>> runner = FaceDetectionPipelineRunner(executor)
                >>> collection = await runner.run(data)
                >>> for item in collection:
                >>>     ...

    """

    _detection_pipeline_name: ClassVar[str] = "face_detection_scrface"

    async def _run(self, data: Sequence[Image]) -> AsyncIterable[SoonValue[Result[RunnerOutput, Error]]]:
        """Return values if success or throw exception."""
        n_batches_per_request = 2
        batch_size = self._pipeline[self._detection_pipeline_name].max_batch_size
        chunks = []
        tasks: List[SoonValue[Result[RunnerOutput, Error]]] = []
        for chunk in chunkify(data, batch_size):
            chunks.append(chunk)

            if len(chunks) < n_batches_per_request:
                continue

            async with asyncer.create_task_group() as task_group:
                for data in chunks:
                    bounding_boxes = task_group.soonify(self._run_detector)(data=data)
                    tasks.append(bounding_boxes)

            yield [task.value for task in tasks]
            chunks = []
            tasks = []

        if len(chunks) > 0:
            async with asyncer.create_task_group() as task_group:
                for data in chunks:
                    bounding_boxes = task_group.soonify(self._run_detector)(data=data)
                    tasks.append(bounding_boxes)

            yield [task.value for task in tasks]

    async def _run_detector(self, data: Sequence[Image]) -> Result[RunnerOutput, Error]:
        """Run the face detection pipeline."""
        pipeline_output = await self._pipeline[self._detection_pipeline_name].run(data)
        return pipeline_output


__all__ = ("FaceDetectionPipelineRunner", "FaceDetectionPipeline")
