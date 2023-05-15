from typing import List

import json

from sacrebleu import CHRF
from sacrebleu.utils import sum_of_lists


chrf = CHRF(6, 2, 2)  # character n-gram order 6, word n-gram order 2, beta 2

def chrf_unreduced_to_str(hypothesis: str, reference: str):
    stats = chrf._extract_corpus_statistics([hypothesis], [[reference]])
    return json.dumps(stats[0])


def chrf_unreduced_str_to_aggregate(strs: List[str]) -> float:
    return chrf._compute_f_score(sum_of_lists([json.loads(s) for s in strs]))


if __name__ == "__main__":
    references = [
        ["bob joe", "i went to the store"],
    ]
    predictions = [
        "bob joe",
        "i went to the library",
    ]

    chrf_unreduced_strs = [chrf_unreduced_to_str(pred, ref) for pred, ref in zip(predictions, references[0])]
    chrf_unreduced_scores = [chrf_unreduced_str_to_aggregate([s]) for s in chrf_unreduced_strs]
    chrf_aggregate_score = chrf_unreduced_str_to_aggregate(chrf_unreduced_strs)

    print(chrf_unreduced_strs)
    print(chrf_unreduced_scores)
    print(chrf_aggregate_score)
