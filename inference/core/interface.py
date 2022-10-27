import abc
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generic, TypeVar, Union


TModelStatistic = Union[Dict[str, str], Dict[str, Dict[str, str]]]
T = TypeVar("T")


class Status(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class HealthDetails:
    pass


@dataclass(frozen=True)
class HealthStatus:
    status: Status
    details: HealthDetails


class HealthProvider(metaclass=abc.ABCMeta):
    """This interface is designed to provide the health status of services."""

    @abc.abstractmethod
    def is_supported(self) -> bool:
        """Check if a service is available."""
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    def get_name(self) -> str:
        """Get a name of a service."""
        raise NotImplementedError("Method must be implemented in child class.")

    @abc.abstractmethod
    def get_health(self) -> HealthStatus:
        """Get health status and health details of a service.

        Available status:
            UP, DOWN, UNKNOWN.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")


class Closeable(metaclass=abc.ABCMeta):
    """This interface is designed to close object resources."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Close resources, relinquishing any underlying resources."""
        raise NotImplementedError("Method must be implemented in child class.")


class Runnable(Generic[T], metaclass=abc.ABCMeta):
    """This interface is designed to mark an object runnable."""

    @abc.abstractmethod
    async def run(self, *args, **kwargs) -> T:
        """Do some jobs here.

        Args:
            *args: Some specific parameters.
            **kwargs: Some specific parameters.

        Raises:
            NotImplementedError: This is an abstract class method.
        """
        raise NotImplementedError("Method must be implemented in child class.")


__all__ = (
    "HealthProvider",
    "Closeable",
    "Runnable",
    "HealthStatus",
    "HealthDetails",
    "Status",
)
