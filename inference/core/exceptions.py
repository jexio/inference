from meiga import Error


class BaseError(Error):

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


class RecoverableError(BaseError):
    """Base class for errors that can be fixed later."""

    pass


class UnrecoverableError(BaseError):
    """Base class for persistent errors."""

    pass


class TritonInferenceException(UnrecoverableError):
    """The base exception."""

    pass


class TritonInferenceServerException(TritonInferenceException):
    """An exception is due to errors on the server side."""

    pass


class TritonInferenceClientException(TritonInferenceException):
    """An exception is due to errors on the client side."""

    pass


class TritonUnableToGetModelException(TritonInferenceException):
    """An exception occurs due to the following cases: The model is not loaded or the model does not exist."""

    pass


class IncompatibleSharedMemory(TritonInferenceException):
    """Error thrown if operation cannot be completed with given shared memory type."""

    pass


class StorageError(RecoverableError):
    """An exception related to 5xx http error."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args)
        self.http_code = kwargs.get("http_code")
        self.http_response = kwargs.get("http_response")


class DownloadError(UnrecoverableError):
    """An exception related to any http error."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args)
        self.http_code = kwargs.get("http_code")
        self.http_response = kwargs.get("http_response")


__all__ = (
    "TritonInferenceServerException",
    "TritonUnableToGetModelException",
    "TritonInferenceClientException",
    "TritonInferenceException",
    "IncompatibleSharedMemory",
    "RecoverableError",
    "UnrecoverableError",
    "StorageError",
    "DownloadError",
)
