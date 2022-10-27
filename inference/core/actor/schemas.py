import io
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from inference.core.type_hints import ImageNDArray, NDArray
from inference.core.validator import Validator
from inference.core.utils.image import base64_to_np_image, bytes_to_np_image, np_image_to_base64


TColor = Tuple[int, int, int]
TLandmark = Tuple[int, int]
TRegion = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Region(Validator):
    """Class that represents a region of an image.

    Attributes:
        x1: x coordinate of top-left corner.
        y1: y coordinate of top-left corner.
        x2: x coordinate of bottom-right corner.
        y2: y coordinate of bottom-right corner.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def __str__(self) -> str:
        """Returns string representation of Region."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Returns repr of Region."""
        return f"Region[x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}]"

    @property
    def width(self) -> int:
        """Get the width of the region."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Get the height of the region."""
        return self.y2 - self.y1

    def to_coords(self) -> TRegion:
        """Convert to plain coordinates.

        Returns:
            coordinates of a region (x1, y1, x2, y2)
        """
        return self.x1, self.y1, self.x2, self.y2

    def validate(self) -> None:
        """Validate values.

        Raises:
            ValueError: if coordinates were negative.
        """
        if self.x1 < 0 or self.y1 < 0 or self.x2 < 0 or self.y2 < 0:
            raise ValueError(f"Expected positive coordinates but got negative.")

    @classmethod
    def of(cls, coords: TRegion) -> "Region":
        """Static constructor for Region class.

        Args:
            coords: coordinates of a region (x1, y1, x2, y2).

        Returns:
            `Region` from bounding box.

        Examples:
            .. code::python
                >>> coordinates = (0, 0, 100, 100)
                >>> r = Region.of(coordinates)
        """
        return Region(*coords)

    @classmethod
    def parse(cls, coords: str) -> "Region":
        """Static constructor for Region class that creates region from a string.

        Args:
            coords: serialized coordinates of a region.

        Returns:
            `Region` from bounding box.

        Raises:
            ValueError: if length of ``coords`` is less than zero or greater than 4.

        Examples:
            .. code::python
                >>> coordinates1 = "[0, 0, 100, 100]"
                >>> coordinates2 = "0, 0, 100, 100"
                >>> r1 = Region.parse(coordinates1)
                >>> r2 = Region.parse(coordinates2)
        """
        coords = coords.replace("[", "").replace("]", "")
        int_coords = tuple([int(c.strip()) for c in coords.split(",")])
        if len(int_coords) < 0 or len(int_coords) > 4:
            raise ValueError(f"Expected - four coordinates, but got - {len(int_coords)}")

        return cls.of(int_coords)


@dataclass(frozen=True)
class Landmark(Validator):
    """Class that represents a landmark in an image.

    Attributes:
        x1: x coordinate.
        y1: y coordinate.
    """

    x1: int
    y1: int

    def __str__(self) -> str:
        """Returns string representation of Landmark."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Returns repr of Landmark."""
        return f"Landmark[x1={self.x1}, y1={self.y1}]"

    def __add__(self, other: "Landmark") -> "Landmark":
        return Landmark(self.x1 + other.x1, self.y1 + other.y1)

    def __sub__(self, other: "Landmark") -> "Landmark":
        return Landmark(self.x1 - other.x1, self.y1 - other.y1)

    def to_coords(self) -> TLandmark:
        """Convert to plain coordinates.

        Returns:
            coordinates of a region (x1, y1)
        """
        return self.x1, self.y1

    @classmethod
    def of(cls, coords: Tuple[int, int]) -> "Landmark":
        """Static constructor for Landmark class.

        Args:
            coords: coordinates of a region (x1, y1).

        Returns:
            A new `Landmark` instance.

        Examples:
            .. code::python
                >>> coordinates = (0, 0)
                >>> r = Landmark.of(coordinates)
        """
        return cls(*coords)

    @classmethod
    def parse(cls, coords: str) -> "Landmark":
        """Static constructor for Landmark class that creates landmark from a string.

        Args:
            coords: serialized coordinates of a landmark.

        Returns:
            A new `Landmark` instance.

        Raises:
            ValueError: if length of ``coords`` is less than zero or greater than 2.

        Examples:
            .. code::python
                >>> coordinates1 = "[0, 0]"
                >>> coordinates2 = "0, 0"
                >>> r1 = Landmark.parse(coordinates1)
                >>> r2 = Landmark.parse(coordinates2)
        """
        coords = coords.replace("[", "").replace("]", "")
        int_coords = tuple([int(c.strip()) for c in coords.split(",")])
        if len(int_coords) < 0 or len(int_coords) > 2:
            raise ValueError(f"Expected - two coordinates, but got - {len(int_coords)}")

        return cls.of(int_coords)

    def validate(self) -> None:
        """Validate values.

        Raises:
            ValueError: if coordinates were negative.
        """
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError(f"Expected positive coordinates but got negative.")


class Landmarks(Validator):
    """Class that represents a list of landmarks in an image.

    Attributes:
        landmarks: A list of important image points.
    """

    landmarks: List[Landmark] = field(default_factory=lambda: list())

    def __str__(self) -> str:
        """Returns string representation of Landmarks."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Returns repr of Landmarks."""
        return f"There are {len(self.landmarks)} landmarks."

    def to_coords(self) -> List[TLandmark]:
        """Convert to plain coordinates.

        Returns:
            List of (x,y) coordinates.
        """
        return [(landmark.x1, landmark.y1) for landmark in self.landmarks]

    @classmethod
    def of(cls, coords: List[Tuple[int, int]]) -> "Landmarks":
        """Static constructor for Landmark class.

        Args:
            coords: A list of (x, y) coordinates.

        Returns:
            A new `Landmarks` instance.

        Examples:
            .. code::python
                >>> coordinates = [(0, 0)]
                >>> r = Landmarks.of(coordinates)
        """
        return cls(landmarks=[Landmark(*item) for item in coords])


@dataclass(frozen=True)
class FaceLandmarks(Landmarks):
    """Class that represents a list of face landmarks.

    The indices are directed clockwise. Initial point - the left eye.
    """

    landmarks: List[Landmark] = field(default_factory=lambda: list())

    @property
    def left_eye(self) -> Landmark:
        """Get the key point, which is the center of the left eye."""
        return self.landmarks[0]

    @property
    def right_eye(self) -> Landmark:
        """Get the key point, which is the center of the right eye."""
        return self.landmarks[1]

    @property
    def nose(self) -> Landmark:
        """Get the key point, which is the center of the nose."""
        return self.landmarks[2]

    @property
    def mouth_right(self) -> Landmark:
        """Get the key point, which is the right edge of the mouth."""
        return self.landmarks[3]

    @property
    def mouth_left(self) -> Landmark:
        """Get the key point, which is the left edge of the mouth."""
        return self.landmarks[4]

    def validate_landmarks(self, value: List[Landmark]) -> None:
        """Ensure the number of landmarks is 5.

        Args:
            value:  A list of important image points.

        Raises:
            ValueError: if size of ``value`` is not equal to 5.
        """
        length = len(value)
        if length != 5:
            raise ValueError(f"Expected - 5 landmarks, but got - {length}.")


@dataclass(frozen=True)
class Image:
    """Class that represents an image as numpy array.

    Attributes:
        data: raw image data as numpy array.
        metadata: some specific parameters as dict.
    """

    data: ImageNDArray = field(compare=False)
    metadata: Dict[str, Any] = field(default_factory=lambda: dict(), compare=False)

    def __str__(self) -> str:
        """Returns string representation of Image."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Returns repr of Image."""
        return f"Image[shape={self.shape}]"

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns a shape of data."""
        return self.data.shape

    @property
    def channel(self) -> int:
        """Returns a number of channels of this image.

        Returns:
            a number of channels
        """
        return self.shape[2]

    @property
    def width(self) -> int:
        """Returns a width of this image.

        Returns:
            width of this image in pixels.
        """
        return self.shape[1]

    @property
    def height(self) -> int:
        """Returns a height of this image in pixels.

        Returns:
            height of this image in pixels.
        """
        return self.shape[0]

    def _fitted_coordinates(self, region: Region) -> Tuple[int, int, int, int]:
        return (
            max(0, region.x1),
            max(0, region.y1),
            min(self.width, region.x2),
            min(self.height, region.y2),
        )

    def copy(self) -> "Image":
        """Creates a copy of this image.

        Returns:
            copy of the image.
        """
        return Image(self.data.copy(), deepcopy(self.metadata))

    def crop(self, region: Region) -> "Image":
        """Crops image.

        Args:
            region: region to crop.

        Returns:
            crop result as newly created image.
        """
        assert self.width > region.x1 and self.height > region.y1, "Region top-left position out of image boundaries."

        x1, y1, x2, y2 = self._fitted_coordinates(region)
        cropped_data = self.data[y1:y2, x1:x2]
        return Image(cropped_data)

    def fill(self, region: Region, color: TColor) -> "Image":
        """Fill an area of the image with a specific color."""
        x1, y1, x2, y2 = self._fitted_coordinates(region)
        new_data = self.data.copy()
        new_data[y1:y2, x1:x2] = color
        return Image(new_data)

    def to_tensor(self) -> NDArray:
        """Convert this image to chw format.

        Returns:
            chw representation of data.
        """
        return self.data.transpose((2, 0, 1))

    def to_base64(self, image_type: str = "png") -> str:
        """Convert this image to base64 encoded string.

        Args:
            image_type: "png", "jpeg" e.t.c

        Returns:
            base64 encoded image.
        """
        return np_image_to_base64(self.data, image_type)

    @classmethod
    def from_base64(cls, image: str) -> "Image":
        """Create a new instance of Image from Base64.

        Args:
            image: Base64 encoded image

        Returns:
            A new instance of Image.
        """
        return cls(base64_to_np_image(image))

    @classmethod
    def from_bytes(cls, image: io.BytesIO) -> "Image":
        """Create a new instance of Image from io.BytesIO.

        Args:
            image: Raw file bytes.

        Returns:
            A new instance of Image.
        """
        return cls(bytes_to_np_image(image))

    @classmethod
    def of(cls, image: Union[str, io.BytesIO]) -> "Image":
        """Static constructor for Image class.

        Create a new instance of Image from base64 string or raw bytes.

        Args:
            image: Base64 string or raw bytes

        Returns:
            A new instance of Image.

        Examples:
            .. code::python
                >>> value = "base64"
                >>> image = Image.of(value)
        """
        if isinstance(image, str):
            return cls.from_base64(image)
        return cls.from_bytes(image)


@dataclass(frozen=True)
class Embedding:
    """A class that represents an object features.

    Attributes:
        features: A dense vector.
    """

    features: List[float] = field(default_factory=lambda: list())

    def __str__(self) -> str:
        """Returns string representation of Embedding."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Returns repr of Landmarks."""
        return f"There are {len(self.features)} features."

    @classmethod
    def of(cls, array: NDArray) -> "Embedding":
        """Static constructor for Embedding class.

        Args:
            array: A face feature as a dense vector

        Returns:
            A new `FaceEmbedding` instance.

        Examples:
            .. code::python
                >>> embedding = NDArray([0.1, 0.2, 0.3])
                >>> embedding = Embedding.of(embedding)
        """
        return cls(array.tolist())


@dataclass(frozen=True)
class BoundingBox(Validator):
    """A class that represents an object that detected on a frame.

    Attributes:
        region: An object boundaries on a frame.
        probability: An object detection probability.
        landmarks: A key points of an object.
        embedding: A feature vector describing this object.
    """

    region: Region
    probability: float
    landmarks: Optional[Landmarks] = field(default=None)
    embedding: Optional[Embedding] = field(default=None)

    def __str__(self) -> str:
        """Returns string representation of BoundingBox."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Returns repr of BoundingBox."""
        return "BoundingBox[region=%r; probability=%0.2f; landmarks=%r; embedding=%r]" % (
            self.region,
            self.probability,
            self.landmarks,
            self.embedding,
        )

    def with_embedding(self, embedding: Embedding) -> "BoundingBox":
        """Creates a new BoundingBox with given embeddings.

        Args:
            embedding: A feature vector.

        Returns:
            A new BoundingBox instance.

        Raises:
            ValueError: if ``self.embedding`` is already set.
        """
        if self.embedding is not None:
            raise ValueError("`embedding` field is already set.")
        return BoundingBox(self.region, self.probability, self.landmarks, embedding)

    def validate_probability(self, value: float) -> None:
        """Validate the ``probability`` field.

        Args:
            value:  A probability.

        Raises:
            ValueError: if the ``value'' is not between 0 and 1.
        """
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Unexpected value of probability - {value}.")


@dataclass(frozen=True)
class ActorImageParameters(Validator):
    """Base image actor configuration.

    Attributes:
        height: a height of the output image in pixels.
        width: a width of the output image in pixels.
        channels: number of channels in the image.
    """

    height: int
    width: int
    channels: int

    def validate(self) -> None:
        if self.height > 1280 or self.height < 64:
            raise ValueError(f"Invalid height size. Got - {self.height}.")
        if self.width > 1280 or self.width < 64:
            raise ValueError(f"Invalid width size. Got - {self.width}.")
        if self.channels > 3 or self.channels < 0:
            raise ValueError(f"Invalid channels size. Got - {self.channels}.")


__all__ = (
    "Region",
    "Image",
    "BoundingBox",
    "Embedding",
    "Landmark",
    "Landmarks",
    "FaceLandmarks",
    "ActorImageParameters"
)
