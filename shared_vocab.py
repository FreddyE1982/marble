from typing import Iterable, Any

from bit_tensor_dataset import object_to_bytes, bytes_to_tensors, flatten_tensor_to_bitstream, build_vocab


def build_shared_vocab(
    datasets: Iterable[Iterable[tuple[Any, Any]]],
    *,
    min_len: int = 3,
    max_len: int = 8,
    max_size: int | None = None,
    start_id: int = 256,
    min_occurrence: int = 4,
) -> dict[tuple[int, ...], int]:
    """Construct a shared vocabulary from multiple datasets.

    Parameters
    ----------
    datasets:
        Iterable of datasets where each dataset yields ``(input, target)`` pairs.
    min_len:
        Minimum pattern length considered when building the vocabulary.
    max_len:
        Maximum pattern length to evaluate.
    max_size:
        Optional limit on the vocabulary size.
    start_id:
        First token ID when assigning words.
    min_occurrence:
        Minimum number of appearances for a pattern to be included.
    """

    bitstream: list[int] = []
    for data in datasets:
        for inp, tgt in data:
            in_bytes = object_to_bytes(inp)
            tgt_bytes = object_to_bytes(tgt)
            bitstream += flatten_tensor_to_bitstream(bytes_to_tensors(in_bytes))
            bitstream += flatten_tensor_to_bitstream(bytes_to_tensors(tgt_bytes))

    return build_vocab(
        bitstream,
        min_len=min_len,
        max_len=max_len,
        max_size=max_size,
        start_id=start_id,
        min_occurrence=min_occurrence,
    )
