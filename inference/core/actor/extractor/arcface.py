from typing import ClassVar, Sequence

from inference.core.actor import register_actor
from inference.core.actor.schemas import Image, Embedding, ActorImageParameters
from inference.core.actor.triton import TritonImageActor
from inference.core.executor import Executor
from inference.core.schemas import TensorType
from inference.core.type_hints import ActorResponse, ImageNDArray
from inference.core.utils.geometric import isotropically_resize_image


@register_actor("arcface")
class ArcFaceActor(TritonImageActor[Image, Embedding]):
    """ArcFace feature extractor.

    Attributes:
        executor: An neural network executor.
        actor_parameters: A configuration options for a triton-actor parameters.

    References: https://arxiv.org/abs/1801.07698
    """

    _mean: ClassVar[float] = 0.5
    _std: ClassVar[float] = 0.5
    _max_pix_value: ClassVar[float] = 255.0

    def __init__(self, executor: Executor, actor_parameters: ActorImageParameters) -> None:
        """Create a new instance of ArcFaceActor."""
        super().__init__(executor, actor_parameters)

    def _initialize_state(self) -> None:
        """Prepare initial state."""
        self._model = "arcface"
        self._version = "1"
        self._datatype = TensorType.TYPE_FP32
        self._output_names = ("output__0",)
        self._input_names = ("input__0",)

    def _preprocess(self, data: Sequence[Image]) -> Sequence[Image]:
        """Preprocess input data."""
        images = [item.data.copy() for item in data]
        images = [(image / self._max_pix_value - self._mean) / self._std for image in images]

        transformed_images = [self._resize(item) for item in images]
        data = [Image(transformed_image) for transformed_image in transformed_images]

        return data

    def _postprocess(
        self, data: Sequence[Image], transformed_data: Sequence[Image], outputs: ActorResponse
    ) -> Sequence[Embedding]:
        """Postprocess neural networks output."""
        embeddings = [Embedding.of(outputs[self._output_names[0]][index]) for index, _ in enumerate(data)]
        return embeddings

    def _resize(self, data: ImageNDArray) -> ImageNDArray:
        """Rescale an image keeping the aspect ratio.

        Args:
            data: The original image.

        Returns:
            The rescaled image keeping the aspect ratio.
        """
        return isotropically_resize_image(data, max(self._parameters.width, self._parameters.height))


__all__ = ("ArcFaceActor",)
