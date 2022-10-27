import io
import os
from urllib import parse

import requests
from loguru import logger
from starlette.status import HTTP_404_NOT_FOUND

from inference.core.exceptions import DownloadError, StorageError
from inference.core.utils.image import image_to_bytes


def download(uri: str) -> io.BytesIO:
    """Downloads media file from given URI and returns results as io.BytesIO.

    Args:
        uri: URI to media file as string.

    Returns:
        Raw file bytes.

    Raises:
        StorageError: if files storage returns non-successful response code.
        DownloadError: If the file download fails.
    """
    logger.debug("Downloading media file from <{uri}>...", uri=uri)

    parsed_uri = parse.urlparse(uri)
    if parsed_uri.scheme in ("file",):
        if not os.path.isfile(parsed_uri.path) or not os.path.exists(parsed_uri.path):
            logger.error("Unable to download file from <{path}>.", path=parsed_uri.path)
            raise DownloadError(
                f"File downloading failed with response code {HTTP_404_NOT_FOUND}.", http_code=HTTP_404_NOT_FOUND
            )
        return image_to_bytes(parsed_uri.path)

    response = requests.get(uri)
    sc = response.status_code
    if sc != 200:
        logger.error(
            "Unable to download file from <{uri}>.",
            uri=uri,
            extra={"http_code": response.status_code, "http_response": str(response.content)},
        )

        if sc // 100 == 5:
            raise StorageError(
                f"Files storage returns non-successful response code {sc}: {str(response.content)}.",
                http_code=response.status_code,
                http_response=str(response.content),
            )

        raise DownloadError(
            f"File downloading failed with response code {sc}: {str(response.content)}.",
            http_code=response.status_code,
            http_response=str(response.content),
        )

    file_bytes = io.BytesIO(response.content)
    logger.debug("Downloaded <{bytes}> bytes.", bytes=f"{sc}: {str(response.content)}.")

    return file_bytes


__all__ = ("download",)
