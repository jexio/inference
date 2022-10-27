from typing import Dict, List, Sequence, Union

import numpy as np

from inference.core.actor import register_actor
from inference.core.actor.triton import TritonImageActor
from inference.core.schemas import TensorType
from inference.core.actor.schemas import Image, BoundingBox, FaceLandmarks, Region
from inference.core.type_hints import ActorResponse, NDArray
from inference.core.utils.geometric import pad_to_size


@register_actor("scrface")
class ScrFaceActor(TritonImageActor[Image, Sequence[BoundingBox]]):
    """Scr face detector.

    Params(M): 3.86
    Flops(G): 34.13

    References: https://arxiv.org/abs/2105.04714
    """

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return 1

    def _initialize_state(self) -> None:
        """Prepare initial state."""
        self._model = "scrface"
        self._version = "1"
        self._datatype = TensorType.TYPE_UINT8
        self._output_names = ("scores", "bboxes", "kpss")
        self._input_names = ("input",)

    def _preprocess(self, data: Sequence[Image]) -> Sequence[Image]:
        """Preprocess input data."""
        max_width = max([image.width for image in data])
        max_height = max([image.height for image in data])
        transformed_data = list()

        for item in data:
            padded = pad_to_size(target_size=(max_height, max_width), image=item.data)
            pads = padded["pads"]
            transformed_data.append(Image(padded["image"], metadata={"pads": pads}))

        return transformed_data

    def _postprocess(
        self, data: Sequence[Image], transformed_data: Sequence[Image], outputs: ActorResponse
    ) -> Sequence[Sequence[BoundingBox]]:
        """Postprocess neural networks output."""
        if self.batch_size == 1:
            for key in self._output_names:
                outputs[key] = outputs[key][np.newaxis, ...]

        bounding_boxes = []
        for index in range(len(data)):

            bounding_boxes.append(list())
            confidences = outputs[self._output_names[0]][index]
            boxes = outputs[self._output_names[1]][index]
            landmarks = outputs[self._output_names[2]][index]
            annotations = self._extract_annotations(boxes, confidences, landmarks)

            for annotation in annotations:
                score = annotation["score"]

                if score != -1:
                    bbox = annotation["bbox"]
                    landmarks = FaceLandmarks.of(annotation["landmarks"])
                    region = Region(bbox[0], bbox[1], bbox[2], bbox[3])
                    face = BoundingBox(region, score, landmarks)
                    bounding_boxes[index].append(face)

        return bounding_boxes

    def _extract_annotations(
        self,
        boxes: NDArray,
        confidences: NDArray,
        landmarks: NDArray,
    ) -> List[Dict[str, Union[List, float]]]:
        """Extract annotations from raw data."""
        annotations: List[Dict[str, Union[List, float]]] = []

        if boxes.shape[0] == 0:
            return [{"bbox": [], "score": -1, "landmarks": []}]

        for crop_id, bbox in enumerate(boxes):
            annotations += [
                {
                    "bbox": bbox.tolist(),
                    "score": float(confidences[crop_id]),
                    "landmarks": landmarks[crop_id].reshape(-1, 2).tolist(),
                }
            ]

        return annotations


__all__ = ("ScrFaceActor",)
