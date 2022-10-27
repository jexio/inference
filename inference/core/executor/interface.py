import abc
from typing import Dict

from inference.core.interface import Runnable
from inference.core.type_hints import ActorResponse


class Executor(Runnable[ActorResponse], metaclass=abc.ABCMeta):
    """Encapsulate here the logic for working with executors such as `onnx/triton-inference-service'."""

    @abc.abstractmethod
    def get_model_metadata(self, model_name: str, model_version: str) -> Dict[str, str]:
        """Get the metadata for specified model.

        Args:
            model_name: Model name.
            model_version: Model version.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    def get_model_config(self, name: str, version: str) -> Dict[str, str]:
        """Get the configuration for specified model.

        Args:
            name: Model name.
            version: Model version.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")


__all__ = ("Executor",)
