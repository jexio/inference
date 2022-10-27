import os
from distutils.util import strtobool
from typing import Callable, Optional, Type, TypeVar, Union, ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def get_value(
    key: str,
    to_type: Union[Type, Callable[[P], T]] = str,
    default_value: Optional[T] = None,
    prefix: str = "APP"
) -> Optional[T]:
    """Get the value of a given key from the .env file.

    Args:
        key: Key name.
        to_type: Type to convert value.
        default_value: Default value.
        prefix: Application environment variable prefix.

    Returns:
        An environment variable value or `default_value`.
    """
    key = f"{prefix}_{key}"
    value = os.getenv(key, default=default_value)
    if value is not None:
        if isinstance(value, str) and value.lower() in ("false", "true"):
            value = strtobool(value)
        value = to_type(value)
    return value


__all__ = ("get_value",)
