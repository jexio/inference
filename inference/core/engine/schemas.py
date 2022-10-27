from dataclasses import dataclass
from typing import Dict, Optional, Union

from inference.core.interface import HealthDetails, HealthStatus, Status


TModelStatistic = Union[Dict[str, str], Dict[str, Dict[str, str]]]


@dataclass(frozen=True)
class EngineHealthDetails(HealthDetails):
    """Health details of inference serving software."""

    version: Optional[str] = None
    failure: Optional[str] = None


@dataclass(frozen=True)
class EngineHealthStatus(HealthStatus):
    """Health status of inference serving software."""

    status: Status
    details: EngineHealthDetails


__all__ = (
    "EngineHealthDetails",
    "EngineHealthStatus",
    "TModelStatistic",
)
