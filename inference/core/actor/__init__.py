import os
from typing import Any, Callable, Dict

from inference.core.actor.base import BaseActor, TritonActor
from inference.core.utils.registry import registry


ACTOR_REGISTRY: Dict[str, BaseActor | TritonActor] = {}


def register_actor(name: str) -> Callable:
    """New actor types can be added to Inference with the :func:`register_actor` function decorator.

    For example::
        @register_actor('retinaface')
        class RetinaFaceActor:
            (...)
    .. note:: All actors must implement the :class:`cls.__name__` interface.

    Args:
        name: the name of the actor.

    Returns:
        A decorator function to register actors.
    """

    def register_actor_cls(cls: Any) -> Any:
        """Add an actor to a registry."""
        if name in ACTOR_REGISTRY:
            raise ValueError(f"Cannot register duplicate actor ({name})")
        ACTOR_REGISTRY[name] = cls
        return cls

    return register_actor_cls


registry(os.path.dirname(__file__), "inference.core.actor")


__all__ = (
    "ACTOR_REGISTRY",
    "register_actor",
)
