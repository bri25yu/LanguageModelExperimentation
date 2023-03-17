"""
A randomly selected set of training examples, of 8 languages:

Latin based: 
Pretrained on by MT5: English, Danish, 
Not pretrained on by MT5: Dyula, Southwestern Dinka

Character based: 
Pretrained on by MT5: Simplified Chinese, Japanese
Not pretrained on by MT5: Standard Tibetan, Yue Chinese

DatasetDict({
    train: Dataset({
        features: ['id', 'source_lang', 'target_lang', 'source', 'target'],
        num_rows: <number of target rows>
    })
})

"""

from datasets import DatasetDict, load_dataset

from lme.training_dataset_utils.flores.utils import select_n


BATCH_SIZE_PER_UPDATE = 2048
NUM_UPDATES = 10000

SEED = 42
DATASET_NAME = "flores200_8_baseline"

LANGUAGES_TO_SELECT = ['sentence_eng_Latn', 
    'sentence_dan_Latn', 
    'sentence_dyu_Latn', 
    'sentence_dik_Latn', 
    'sentence_zho_Hans', 
    'sentence_jpn_Jpan', 
    'sentence_bod_Tibt', 
    'sentence_yue_Hant']

OTHER_COLUMNS_TO_SELECT = ['id',
    'URL',
    'domain',
    'topic',
    'has_image',
    'has_hyperlink']


def main():
    total_set_size = BATCH_SIZE_PER_UPDATE * NUM_UPDATES

    raw_dataset = load_dataset("facebook/flores", "all")["dev"]
    columns_to_remove = tuple(set(raw_dataset.column_names) - set(OTHER_COLUMNS_TO_SELECT + LANGUAGES_TO_SELECT))
    raw_dataset = raw_dataset.remove_columns(columns_to_remove)
    dataset_dict = DatasetDict({
        "train": select_n(raw_dataset, total_set_size, SEED),
    })

    dataset_dict.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    main()
