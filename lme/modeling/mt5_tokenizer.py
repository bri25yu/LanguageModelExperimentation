from typing import Any, Callable, Iterator

from argparse import ArgumentParser, Namespace

from datasets import load_dataset, Dataset

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers import AutoTokenizer


class TrainMT5Tokenizer:
    OLD_TOKENIZER_NAME = "google/mt5-base"

    def train(self) -> None:
        args = self.parse_args()

        dataset = self.load_data()
        batch_iterator_fn = self.get_batch_iterator_fn()

        batch_iterator = batch_iterator_fn(dataset, args.batch_size)
        old_tokenizer = self.load_old_tokenizer()

        new_tokenizer = self.train_new_tokenizer(batch_iterator, old_tokenizer, len(dataset))

        self.save_new_tokenizer(new_tokenizer)

    def parse_args(self) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--batch_size", "-bs", default=1000
        )

        return parser.parse_args()

    def load_data(self) -> Dataset:
        # Loads a DatasetDict with Chinese, English, and Tibetan splits
        dataset = load_dataset("buddhist-nlp/pretraining_data", use_auth_token=True)

        # Since mt5 is already trained on Chinese and English, we only want to train our tokenizer on the Tibetan split
        dataset = dataset["tibetan"]

        return dataset

    def get_batch_iterator_fn(self) -> Callable[[Any], Iterator]:
        def batch_iterator_fn(dataset: Dataset, batch_size: int):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i: i + batch_size]["text"]

        return batch_iterator_fn

    def load_old_tokenizer(self) -> PreTrainedTokenizerBase:
        old_tokenizer_name = self.OLD_TOKENIZER_NAME

        old_tokenizer = AutoTokenizer.from_pretrained(old_tokenizer_name)

        return old_tokenizer

    def train_new_tokenizer(self, batch_iterator: Iterator, old_tokenizer: PreTrainedTokenizerBase, length: int) -> PreTrainedTokenizerBase:
        old_vocab_size = old_tokenizer.vocab_size
        new_vocab_size = old_vocab_size + 5000

        new_tokenizer = old_tokenizer.train_new_from_iterator(
            batch_iterator, new_vocab_size, length=length,
        )

        return new_tokenizer

    def save_new_tokenizer(self, new_tokenizer: PreTrainedTokenizerBase) -> None:
        new_tokenizer.push_to_hub("buddhist-nlp/mt5-tibetan-tokenizer")


if __name__ == "__main__":
    TrainMT5Tokenizer().train()
