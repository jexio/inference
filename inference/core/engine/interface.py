import abc
from typing import Dict, Generic, Mapping, Optional, Sequence, TypeVar, Union

from inference.core.interface import Closeable, HealthProvider, Runnable
from inference.core.engine.schemas import TModelStatistic


T = TypeVar("T")
NestedItem = Mapping[str, Union[int, float, Sequence]]
ModelMetadata = Dict[str, Union[NestedItem, str, int, float]]
ModelConfig = Dict[str, Union[NestedItem, str, int, float]]


class Engine(Generic[T], Runnable[T], Closeable, HealthProvider, metaclass=abc.ABCMeta):
    """This interface is designed to hide low level dependencies associated with connection to inference server."""

    @abc.abstractmethod
    def get_inference_statistics(self, name: Optional[str] = "", version: Optional[str] = "") -> TModelStatistic:
        """Get the inference statistics for the specified model name and version.

        Args:
            name: The name of the model to get inference statistics.
            version: The version of the model.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    def get_model_metadata(self, name: str, version: str) -> ModelMetadata:
        """Get the metadata for specified model.

        Args:
            name: The name of the model to get model metadata.
            version: The version of the model.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    def get_model_config(self, name: str, version: str) -> ModelConfig:
        """Get the configuration for specified model.

        Args:
            name: The name of the model to get model config.
            version: The version of the model.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Contact the inference server and get liveness.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Contact the inference server and get readiness.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    def is_model_ready(self, name: str, version: str) -> bool:
        """Contact the inference server and get the readiness of specified model.

        Args:
            name: Name of the model specifies which model can handle the inference requests that are sent to inference server.
            version: Version of the model.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")


__all__ = (
    "Engine",
    "ModelMetadata",
    "ModelConfig",
)
