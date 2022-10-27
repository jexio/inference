from inference.core.exceptions import UnrecoverableError


class ActorException(UnrecoverableError):
    """The base exception."""

    pass


class InvalidShapeException(ActorException):
    """An exception occurs if the data size is not equal to the expected size."""

    pass


class InvalidBatchSize(ActorException):
    """An exceptions occurs if the batch size is not valid."""

    pass


__all__ = (
    "ActorException",
    "InvalidShapeException",
    "InvalidBatchSize",
)
