import base64
import io
import math
from typing import Optional

import numpy as np
from PIL import Image


def np_image_to_base64(image: np.ndarray, image_format: str = "png") -> Optional[str]:
    """Helper function that converts image represents as Numpy array into an image that encoded as base64 string.

    Args:
        image: numpy representation of an image.
        image_format: format of result image (png or jpeg).

    Returns:
        Base64 encoded image or None in case of conversion error.
    """
    if image.ndim == 1:
        dim = math.isqrt(int(len(image) / 3))
        try:
            image = image.reshape((dim, dim, 3))
        except ValueError:
            return None

    rgb_image = image[..., ::-1].copy()
    pil_image = Image.fromarray(rgb_image.astype("uint8"))
    buffer = io.BytesIO()
    pil_image.save(buffer, image_format)

    buffer.seek(0)
    base64_bytes = base64.b64encode(buffer.read())

    return base64_bytes.decode("utf-8")


def base64_to_np_image(image: str) -> np.ndarray:
    """Helper function that converts image represented as Base64 string into a Numpy array.

    Args:
        image: Base64 encoded image.

    Returns:
        numpy representation of an image.
    """
    image_bytes = base64.b64decode(image)
    buffer = io.BytesIO(image_bytes)

    pil_image = Image.open(buffer)
    rgb_image = np.asarray(pil_image, dtype=np.uint8)

    return rgb_image[..., ::-1].copy()


def bytes_to_np_image(image: io.BytesIO) -> np.ndarray:
    """Helper function that converts image represented as bytes into a Numpy array.

    Args:
        image: Raw file bytes.

    Returns:
        numpy representation of an image.
    """
    pil_image = Image.open(image)
    rgb_image = np.asarray(pil_image, dtype=np.uint8)

    return rgb_image[..., ::-1].copy()


def image_to_bytes(image: str) -> io.BytesIO:
    """Helper function that converts image represented as file into a raw bytes.

    Args:
        image: Path to image on disk.

    Returns:
        Raw file bytes.
    """
    pil_im = Image.open(image)
    b = io.BytesIO()
    pil_im.save(b, pil_im.format)
    return b


__all__ = (
    "base64_to_np_image",
    "np_image_to_base64",
    "bytes_to_np_image",
    "image_to_bytes",
)
