from functools import wraps
from typing import Callable, Optional, TypeVar, ParamSpec

from loguru import logger
from tritonclient.utils import InferenceServerException

from inference.core.exceptions import TritonInferenceServerException


T = TypeVar("T")
P = ParamSpec("P")


def inference_executor_exception_handler(func: Callable[[P], T]) -> Callable[[P], Optional[T]]:
    """Decorator to catch exceptions."""

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN201
        """Inner decorator."""
        try:
            response = func(*args, **kwargs)
        except InferenceServerException as e:
            logger.error(e.message())
            raise TritonInferenceServerException(f"non-OK-status RPC termination: {str(e)}.")
        except Exception as e:  # noqa: B902
            logger.critical("Uncaught error: {error}.", error=e)
            raise

        return response

    return wrapper


__all__ = ("inference_executor_exception_handler",)
