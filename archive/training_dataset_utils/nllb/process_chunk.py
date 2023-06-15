from typing import Any, Dict, Sequence

from tqdm.auto import tqdm

from lme.training_dataset_utils.nllb.nllb_lang_pairs import NLLB_PAIRS

from datasets import load_dataset, concatenate_datasets


def just_translations(examples: Dict[str, Sequence[Any]]) -> Dict[str, Sequence[str]]:
    res = {
        "lang1": [],
        "lang2": [],
        "sentence1": [],
        "sentence2": [],
    }
    for translation in examples["translation"]:
        keys = list(translation.keys())
        res["lang1"].append(keys[0])
        res["lang2"].append(keys[1])
        res["sentence1"].append(translation[keys[0]])
        res["sentence2"].append(translation[keys[1]])

    return res


def process_and_upload_100(chunk: int) -> None:
    batch_size = 100
    nllb_pairs_to_process = NLLB_PAIRS[batch_size * chunk: batch_size * (chunk + 1)]

    datasets_to_cat = []
    for language_pair in tqdm(nllb_pairs_to_process, desc="Filtering just translations"):
        dataset = load_dataset("allenai/nllb", "-".join(language_pair))["train"]
        dataset = dataset.map(just_translations, batched=True, num_proc=16, remove_columns=dataset.column_names)
        datasets_to_cat.append(dataset)

    dataset = concatenate_datasets(datasets_to_cat)
    dataset.push_to_hub(f"nllb_chunk{chunk}", private=True)


if __name__ == "__main__":
    CHUNK = None
    assert CHUNK
    process_and_upload_100(CHUNK)
