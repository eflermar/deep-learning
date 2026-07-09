#!/usr/bin/env python3
# Author: Martin Efler
import argparse
import os

import npfl138
import torch
import torchmetrics

npfl138.require_version("2526.7")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=128, type=int, help="RNN layer dimension.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.1, type=float, help="Mask words with the given probability.")


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        word_ids = torch.tensor([self.dataset.words.string_vocab.index(w) for w in example["words"]])
        tag_ids = torch.tensor([self.dataset.tags.string_vocab.index(t) for t in example["tags"]])
        return word_ids, example["words"], tag_ids

    def collate(self, batch):
        word_ids, words, tag_ids = zip(*batch)
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True)
        unique_words, words_indices = self.dataset.cle_batch(words)
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True)
        return (word_ids, unique_words, words_indices), tag_ids


class Model(npfl138.TrainableModule):
    class MaskElements(torch.nn.Module):
        """A layer randomly masking elements with a given value."""

        def __init__(self, mask_probability, mask_value):
            super().__init__()
            self._mask_probability = mask_probability
            self._mask_value = mask_value

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if self.training and self._mask_probability:
                mask = torch.rand_like(inputs, dtype=torch.float32)
                inputs = torch.where(mask < self._mask_probability, torch.tensor(self._mask_value), inputs)
            return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._word_masking = self.MaskElements(args.word_masking, MorphoDataset.UNK)
        self._char_embedding = torch.nn.Embedding(len(train.words.char_vocab), args.cle_dim)
        self._char_rnn = torch.nn.GRU(
            input_size=args.cle_dim,
            hidden_size=args.cle_dim,
            bidirectional=True,
            num_layers=3,
            dropout=.2,
        )
        self._word_embedding = torch.nn.Embedding(len(train.words.string_vocab), args.we_dim)
        self._word_rnn = torch.nn.LSTM(
            input_size=args.we_dim + 2 * args.cle_dim,
            hidden_size=args.rnn_dim,
            bidirectional=True,
            num_layers=3,
            dropout=.2,
        )
        self._output_layer = torch.nn.Linear(args.rnn_dim, len(train.tags.string_vocab))

    def forward(self, word_ids: torch.Tensor, unique_words: torch.Tensor, word_indices: torch.Tensor) -> torch.Tensor:
        hidden = self._word_masking(word_ids)
        hidden = self._word_embedding(hidden)
        cle = self._char_embedding(unique_words)
        char_lengths = (unique_words != MorphoDataset.PAD).sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            cle, char_lengths, batch_first=True, enforce_sorted=False,
        )
        cle = self._char_rnn(packed)[1]
        cle = torch.cat([cle[0], cle[1]], dim=1)
        cle = torch.nn.functional.embedding(word_indices, cle)
        hidden = torch.cat([hidden, cle], dim=-1)
        lengths = (word_ids != MorphoDataset.PAD).sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            hidden, lengths, batch_first=True, enforce_sorted=False,
        )
        packed = self._word_rnn(packed)[0]
        hidden = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)[0]
        forward, backward = torch.chunk(hidden, 2, dim=-1)
        hidden = forward + backward
        hidden = self._output_layer(hidden).permute(0, 2, 1)

        return hidden


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True,
                                                      num_workers=args.threads)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size, num_workers=args.threads)
    test = TrainableDataset(morpho.test).dataloader(batch_size=args.batch_size, num_workers=args.threads)
    model = Model(args, morpho.train)
    model.configure(
        optimizer=torch.optim.AdamW(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=len(morpho.train.tags.string_vocab),
                                                   ignore_index=morpho.PAD)},
        logdir=npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args)),
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test, data_with_labels=True, as_numpy=True)

        for predicted_tags, words in zip(predictions, morpho.test.words.strings):
            for predicted_tag in predicted_tags[:, :len(words)].argmax(axis=0):
                print(morpho.train.tags.string_vocab.string(predicted_tag), file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
