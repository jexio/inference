from typing import Iterator, Sequence, TypeVar

T = TypeVar("T")


def chunkify(sequence: Sequence[T], chunk_size: int) -> Iterator[Sequence[T]]:
    """Yield successive n-sized chunks from lst.

    if `chunk_size` is larger than `sequence` length then yield `sequence` as is.

    Args:
        sequence: Array for splitting into parts.
        chunk_size: Size of a part.

    Yields:
        Part of `sequence` of size `chunk_size`.
    """
    if chunk_size == -1 or chunk_size > len(sequence):
        chunk_size = len(sequence)
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i : i + chunk_size]


__all__ = ("chunkify",)
