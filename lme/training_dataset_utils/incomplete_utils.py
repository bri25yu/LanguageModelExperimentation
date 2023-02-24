
from typing import Dict, Sequence
from torch import randint



def add_prefix_truncated_output(inputs: Dict[str, Sequence], max_input_length: int) -> None:
    # Truncate the labels and append it to the input, taking the first part. 
    # [* * * * * * *] ---------------
    result_length = randint(len(inputs["labels"]), ())
    result_length = min(result_length, max_input_length - len(inputs["input_ids"]))

    # Add the prefix
    to_append = inputs["labels"][:result_length]

    inputs["input_ids"] = inputs["input_ids"] + to_append
    inputs["attention_mask"] = inputs["attention_mask"] + [1] * len(to_append)


def add_prefix_and_suffix_truncated_output(inputs: Dict[str, Sequence], max_input_length: int) -> None:
    # Truncate the labels and append it to the input, somewhere in the middle
    # [* * *] --------------- [* * * *]
    # result_length = randint(len(inputs["labels"]), ())
    # result_length = min(result_length, max_input_length - len(inputs["input_ids"]))

    # Randomize the total length of the sequence based on the max length
    # result_length = randint(max_input_length - len(inputs["input_ids"]), ())
    result_length = randint(len(inputs["labels"]), ())
    result_length = min(result_length, max_input_length - len(inputs["input_ids"]))

    # Create space for the sentinel token
    result_length -= 1

    if result_length <= 0:
        return

    # Generate suffix and prefix locations based on max length
    prefix_cutoff = randint(result_length, ())
    prefix = inputs["labels"][:prefix_cutoff]
    suffix_cutoff = result_length - prefix_cutoff
    suffix = inputs["labels"][(-suffix_cutoff):] if suffix_cutoff > 0 else []

    # Add the prefix and suffix and a sentinel token in between
    inputs["input_ids"] = inputs["input_ids"] + prefix
    inputs["input_ids"] = inputs["input_ids"] + [32087] # Extra input sentinal token
    inputs["input_ids"] = inputs["input_ids"] + suffix
    inputs["attention_mask"] = inputs["attention_mask"] + [1] * (len(prefix)+len(suffix)+1)


def add_middle_truncated_output(inputs: Dict[str, Sequence], max_input_length: int) -> None:
    # Truncate the labels and append it to the input, somewhere in the middle. 
    # -------- [* * * * * *] ---------
    result_length = randint(len(inputs["labels"]), ())
    result_length = min(result_length, max_input_length - len(inputs["input_ids"]))

    # Generate offset for starting position of the truncation
    offset = randint(len(inputs["labels"]) - result_length, ())
    
    # Add the truncated output in the middle
    to_append = inputs["labels"][offset:(offset + result_length)]

    inputs["input_ids"] = inputs["input_ids"] + to_append
    inputs["attention_mask"] = inputs["attention_mask"] + [1] * len(to_append)


def add_suffix_truncated_output(inputs: Dict[str, Sequence], max_input_length: int) -> None:
    # Truncate the labels and append it to the input, taking the last part. 
    # --------------- [* * * * * * *]
    result_length = randint(len(inputs["labels"]), ())
    result_length = min(result_length, max_input_length - len(inputs["input_ids"]))

    if result_length == 0:
        return

    # Add the suffix
    to_append = inputs["labels"][(-result_length):]

    inputs["input_ids"] = inputs["input_ids"] + to_append
    inputs["attention_mask"] = inputs["attention_mask"] + [1] * len(to_append)
