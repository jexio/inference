import os
from typing import Any, Callable, Dict
from inference.core.utils.registry import registry


PIPELINE_REGISTRY: Dict[str, "BasePipeline"] = {}


def register_pipeline(name: str) -> Callable:
    """New pipeline types can be added to Inference with the :func:`register_pipeline` function decorator.

    For example::
        @register_pipeline('face_detection')
        class FaceDetectionPipeline:
            (...)
    .. note:: All pipelines must implement the :class:`cls.__name__` interface.

    Args:
        name: the name of the pipeline.

    Returns:
        A decorator function to register pipelines.
    """

    def register_pipeline_cls(cls: Any) -> Any:
        """Add a pipeline to a registry."""
        if name in PIPELINE_REGISTRY:
            raise ValueError(f"Cannot register duplicate pipeline ({name})")
        PIPELINE_REGISTRY[name] = cls
        return cls

    return register_pipeline_cls


registry(os.path.dirname(__file__), "inference.core.pipeline")


__all__ = (
    "PIPELINE_REGISTRY",
    "register_pipeline",
)
