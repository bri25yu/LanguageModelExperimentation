"""
A randomly selected set of 5000 val examples and 10000 test examples.

DatasetDict({
    val: Dataset({
        features: ['id', 'source_lang', 'target_lang', 'source', 'target'],
        num_rows: 5000
    })
    test: Dataset({
        features: ['id', 'source_lang', 'target_lang', 'source', 'target'],
        num_rows: 10000
    })
})
"""

from datasets import DatasetDict, load_dataset

from lme.training_dataset_utils.flores.utils import select_n


VAL_SET_SIZE = 5000
TEST_SET_SIZE = 10000

SEED = 42
DATASET_NAME = "flores200_8_val_test"

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
    total_set_size = VAL_SET_SIZE + TEST_SET_SIZE
    
    raw_dataset = load_dataset("facebook/flores", "all")["devtest"]
    columns_to_remove = tuple(set(raw_dataset.column_names) - set(OTHER_COLUMNS_TO_SELECT + LANGUAGES_TO_SELECT))
    raw_dataset = raw_dataset.remove_columns(columns_to_remove)

    total_dataset = select_n(raw_dataset, total_set_size, seed=SEED)

    dataset_dict = total_dataset.train_test_split(
        test_size=TEST_SET_SIZE,
        shuffle=True,
        seed=SEED,
    )
    dataset_dict = DatasetDict({
        "val": dataset_dict["train"],
        "test": dataset_dict["test"],
    })

    dataset_dict.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    main()
