from typing import List, Tuple
from numpy.typing import NDArray

from numpy import (
    arange, array, argwhere, bool_, cumsum, empty, equal, hstack, insert, int_, round, ones, zeros
)
from numpy.random import shuffle


__all__ = ["create_span_corrupt_inputs"]


def get_noise_statistics(mask_prob: float, average_span_length: int, length: int) -> Tuple[int, int, int]:
    num_noise_tokens = max(1, int(round(length * mask_prob)))
    num_spans = max(1, int(round(num_noise_tokens / average_span_length)))

    num_noise_tokens = min(num_noise_tokens, length - num_spans)
    num_noise_tokens = max(num_noise_tokens, num_spans)

    num_nonnoise_tokens = length - num_noise_tokens

    return num_spans, num_noise_tokens, num_nonnoise_tokens


def get_span_lengths(length: int, num_spans: int) -> NDArray[int_]:
    stars = zeros((length - num_spans,), dtype=bool)
    bars = ones((num_spans - 1,), dtype=bool)
    stars_and_bars = hstack((stars, bars))

    shuffle(stars_and_bars)

    bars_idxs = argwhere(stars_and_bars).ravel() - arange(num_spans - 1)
    indices = hstack(([0], bars_idxs, [length - num_spans]))
    lengths = indices[1:] - indices[:-1] + 1

    return lengths


def create_noise_mask(length: int, noise_lengths: NDArray[int_], nonnoise_lengths: NDArray[int_]) -> NDArray[bool_]:
    """
    >>> from numpy import array
    >>> length = 10
    >>> nonnoise_lengths = array([3, 3])
    >>> noise_lengths = array([2, 2])
    >>> create_noise_mask(length, noise_lengths, nonnoise_lengths)
    array([False, False, False,  True,  True, False, False, False,  True,
            True])

    """
    lengths = empty((len(noise_lengths) + len(nonnoise_lengths),), dtype=noise_lengths.dtype)
    lengths[::2] = nonnoise_lengths
    lengths[1::2] = noise_lengths

    start_indices = cumsum(lengths)[:-1]
    start_mask = zeros((length,), dtype=bool)
    start_mask[start_indices] = True

    spread_mask = cumsum(start_mask)
    noise_mask = equal(spread_mask % 2, 1)

    return noise_mask


def get_ids(input_ids: NDArray[int_], mask: NDArray[bool_]) -> NDArray[int_]:
    return input_ids[mask]


def insert_sentinel_ids(input_ids: NDArray[int_], lengths: List[int], sentinel_id_start: int) -> NDArray[int_]:
    """
    For the first example,
        insert_idxs is [3, 5, 9]
        sentinel_ids is [250099, 250098, 250097]
    For the second example,
        insert_idxs = [0, 3, 5, 9]
        sentinel_ids is [250099, 250098, 250097, 250096]

    >>> input_ids = [0, 1, 2, 0, 1, 0, 1, 2, 3]
    >>> sentinel_id_start = 250099
    >>> lengths = [3, 2, 4]
    >>> insert_sentinel_ids(input_ids, lengths, sentinel_id_start)
    array([     0,      1,      2, 250099,      0,      1, 250098,      0,
                1,      2,      3, 250097])
    >>> lengths = [0] + lengths
    >>> insert_sentinel_ids(input_ids, lengths, sentinel_id_start)
    array([250099,      0,      1,      2, 250098,      0,      1, 250097,
                0,      1,      2,      3, 250096])

    """
    insert_idxs = cumsum(lengths)
    sentinel_ids = sentinel_id_start - arange(len(lengths))

    expanded_inputs = insert(input_ids, insert_idxs, sentinel_ids)

    return expanded_inputs


def create_span_corrupt_inputs(
    input_ids: List[int], mask_prob: float, average_span_length: int, sentinel_id_start: int
) -> Tuple[List[int], List[int]]:
    """
    Parameters
    ----------
    input_ids: List[int]
    mask_prob: float
    average_span_length: int
    sentinel_id_start: int
        This method will count down from the provided value e.g. t5 uses 
        "<extra_id_0>" as id 250099 and "<extra_id_1>" as id 250098. 

    Returns
    -------
    corrupted_input_ids: List[int]
    label_ids: List[int]

    >>> from numpy import arange
    >>> from numpy.random import seed as set_seed
    >>> set_seed(42)
    >>> input_ids = arange(50)
    >>> mask_prob, average_span_length, sentinel_id_start = 0.15, 3, 250099
    >>> corrupted_input_ids, label_ids = create_span_corrupt_inputs(input_ids, mask_prob, average_span_length, sentinel_id_start)
    >>> corrupted_input_ids
    array([     0, 250099,      4,      5,      6,      7,      8,      9,
               10,     11,     12,     13,     14,     15,     16,     17,
               18,     19,     20,     21,     22,     23,     24,     25,
               26,     27,     28,     29,     30,     31,     32,     33,
           250098,     38,     39,     40,     41,     42,     43,     44,
               45,     46,     47,     48, 250097])
    >>> label_ids
    array([250099,      1,      2,      3, 250098,     34,     35,     36,
               37, 250097,     49, 250096])
    >>> corrupted_input_ids, label_ids = create_span_corrupt_inputs(input_ids, mask_prob, average_span_length, sentinel_id_start)
    >>> corrupted_input_ids
    array([     0,      1,      2,      3,      4,      5,      6,      7,
                8,      9,     10, 250099,     12,     13,     14,     15,
               16,     17,     18,     19,     20,     21,     22,     23,
           250098,     28,     29,     30,     31,     32,     33,     34,
               35,     36,     37,     38,     39,     40,     41,     42,
               43,     44,     45,     46, 250097])
    >>> label_ids
    array([250099,     11, 250098,     24,     25,     26,     27, 250097,
               47,     48,     49, 250096])

    >>> from itertools import product
    >>> mask_probs = arange(0.1, 1, 0.1)
    >>> average_span_lengths = arange(2, 6)
    >>> input_ids = arange(50)
    >>> _ = [\
        create_span_corrupt_inputs(input_ids, mask_prob, average_span_length, sentinel_id_start)\
        for mask_prob, average_span_length in product(mask_probs, average_span_lengths)\
    ]

    """
    assert 0 < mask_prob < 1, f"Mask probability must be in (0, 1)"

    input_ids = array(input_ids)
    total_length = len(input_ids)

    num_spans, num_noise_tokens, num_nonnoise_tokens =\
        get_noise_statistics(mask_prob, average_span_length, total_length)

    noise_lengths = get_span_lengths(num_noise_tokens, num_spans)
    nonnoise_lengths = get_span_lengths(num_nonnoise_tokens, num_spans)

    noise_mask = create_noise_mask(total_length, noise_lengths, nonnoise_lengths)

    noise_ids = get_ids(input_ids, noise_mask)
    nonnoise_ids = get_ids(input_ids, ~noise_mask)

    corrupted_input_ids = insert_sentinel_ids(nonnoise_ids, nonnoise_lengths, sentinel_id_start)
    label_ids = insert_sentinel_ids(noise_ids, hstack(([0], noise_lengths)), sentinel_id_start)

    return corrupted_input_ids, label_ids


if __name__ == "__main__":
    import doctest
    doctest.testmod()
