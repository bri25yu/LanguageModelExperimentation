"""
A randomly selected set of training examples tokenized with the mT5 tokenizer with
one input but predicting the english and target language as outputs. 

The final translation task is just to translate between the two languages, but during 
training, the model is given the english output as a scaffold to help it learn to 
predict the target language, which it may or may not have a prior on.

Task: Translate Language A to Language B (LA, LB)
Input:
<LA Token> <Eng Token> <LB Token> <LA input>
Training target output:
<Eng output> <LB output>

DatasetDict({
    train: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 20480000
    })
    val: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 5000
    })
    test: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 10000
    })
})

"""

from datasets import DatasetDict, load_dataset

from lme.training_dataset_utils.flores.utils import select_n


BATCH_SIZE_PER_UPDATE = 1024
NUM_UPDATES = 20000

SEED = 42
DATASET_NAME = "flores200_eng_scaffolding_large"


def main():
    total_set_size = BATCH_SIZE_PER_UPDATE * NUM_UPDATES

    raw_dataset = load_dataset("facebook/flores", "all")["dev"]

    eng_data = raw_dataset["sentence_eng_Latn"]

    raw_dataset = raw_dataset.remove_columns(["sentence_eng_Latn"])

    dataset_dict = DatasetDict({
        "train": select_n(raw_dataset, total_set_size, SEED, eng_data=eng_data),
    })

    dataset_dict.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    main()
