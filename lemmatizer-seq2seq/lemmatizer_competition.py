#!/usr/bin/env python3
# Author: Martin Efler

import argparse
import itertools
import os

import npfl138
import torch
import torchmetrics
from npfl138.callbacks import KeepBestWeights

npfl138.require_version("2526.9")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--cle_dim", default=256, type=int, help="CLE embedding dimension.")
parser.add_argument("--rnn_dim", default=256, type=int, help="RNN layer dimension.")
parser.add_argument("--tie_embeddings", default=True, action=argparse.BooleanOptionalAction,
                    help="Tie target embeddings.")
parser.add_argument("--blocks", default=3, type=int, help="Number of residual blocks.")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--show_results_every_batch", default=0, type=int, help="Show results every given batch.")


class ResidualBiLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normed_x = self.norm(x)
        out = self.lstm(normed_x)[0]
        forward, backward = torch.chunk(out, 2, dim=-1)
        out = forward + backward
        out = self.dropout(out)
        return out + x


class WithAttention(torch.nn.Module):
    """A class adding Bahdanau attention to a given RNN cell."""

    def __init__(self, cell, attention_dim):
        super().__init__()
        self._encoded_projected = None
        self._encoded = None
        self._cell = cell

        self._project_encoder_layer = nn.Linear(cell.hidden_size, attention_dim)
        self._project_decoder_layer = nn.Linear(cell.hidden_size, attention_dim)
        self._output_layer = nn.Linear(attention_dim, 1)

    def setup_memory(self, encoded):
        self._encoded = encoded
        self._encoded_projected = self._project_encoder_layer(encoded)

    def forward(self, inputs, states):
        logits = self._encoded_projected + self._project_decoder_layer(states).unsqueeze(1)
        logits = nn.functional.tanh(logits)
        logits = self._output_layer(logits)
        logits = logits.squeeze(-1)
        logits[(self._encoded == MorphoDataset.PAD).all(dim=-1)] = -1e9
        weights = logits.softmax(dim=-1)
        attention = (self._encoded * weights.unsqueeze(-1)).sum(dim=1)
        return self._cell(torch.cat([inputs, attention], dim=-1), states)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._source_vocab = train.words.char_vocab
        self._target_vocab = train.lemmas.char_vocab

        self._source_embedding = nn.Embedding(len(self._source_vocab), args.cle_dim)
        self.dropout = nn.Dropout(args.dropout)

        self._source_encoder_blocks = nn.ModuleList([
            ResidualBiLSTMBlock(input_size=args.cle_dim, hidden_size=args.rnn_dim, dropout=args.dropout)
            for _ in range(args.blocks)
        ])

        self._target_rnn_cell = WithAttention(attention_dim=args.rnn_dim,
                                              cell=nn.GRUCell(args.cle_dim + args.rnn_dim, args.rnn_dim))

        self._target_output_layer = nn.Linear(args.rnn_dim, len(self._target_vocab))

        if not args.tie_embeddings:
            self._target_embedding = nn.Embedding(len(self._target_vocab), args.cle_dim)
        else:
            self._target_embedding = lambda x: nn.functional.embedding(x, self._target_output_layer.weight * torch.sqrt(
                torch.tensor(args.rnn_dim, dtype=torch.float32)))

        self._show_results_every_batch = args.show_results_every_batch
        self._batches = 0

    def forward(self, words: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(words)
        if targets is not None:
            return self.decoder_training(encoded, targets)
        else:
            return self.decoder_prediction(encoded, max_length=words.shape[1] + 10)

    def encoder(self, words: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self._source_embedding(words))

        for block in self._source_encoder_blocks:
            hidden = block(hidden)

        mask = (words != MorphoDataset.PAD).unsqueeze(-1).float()
        hidden = hidden * mask

        return hidden

    def decoder_training(self, encoded: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prepend = torch.full((targets.size(0), 1), MorphoDataset.BOW, device=targets.device)
        inputs = torch.cat((prepend, targets[..., :-1]), dim=1)
        self._target_rnn_cell.setup_memory(encoded)

        embedded = self.dropout(self._target_embedding(inputs))
        hidden = encoded[:, 0]
        outputs = []
        for i in range(embedded.shape[1]):
            hidden = self._target_rnn_cell(embedded[:, i, :], hidden)
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=1)
        out = self._target_output_layer(self.dropout(outputs))
        return out.permute(0, 2, 1)

    def decoder_prediction(self, encoded: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = encoded.shape[0]

        self._target_rnn_cell.setup_memory(encoded)

        index = 0
        inputs = torch.full((batch_size,), MorphoDataset.BOW, device=encoded.device)
        states = encoded[:, 0]
        results = []
        result_lengths = torch.full((batch_size,), max_length, device=encoded.device)

        while index < max_length and torch.any(result_lengths == max_length):
            hidden = self._target_embedding(inputs)
            states = self._target_rnn_cell(hidden, states)
            out = self._target_output_layer(states)
            predictions = torch.argmax(out, dim=-1)

            results.append(predictions)
            result_lengths[(predictions == MorphoDataset.EOW) & (result_lengths > index)] = index + 1
            inputs = predictions
            index += 1

        results = torch.stack(results, dim=1)
        return results

    def compute_metrics(self, y_pred, y, *xs):
        if self.training:
            y_pred = y_pred.argmax(dim=1)
        y_pred = y_pred[:, :y.shape[-1]]
        y_pred = torch.nn.functional.pad(y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=MorphoDataset.PAD)
        self.metrics["accuracy"].update(torch.all((y_pred == y) | (y == MorphoDataset.PAD), dim=-1))
        return self.metrics

    def train_step(self, xs, y):
        result = super().train_step(xs, y)
        self._batches += 1
        if self._show_results_every_batch and self._batches % self._show_results_every_batch == 0:
            self.log_console("{}: {} -> {}".format(
                self._batches,
                "".join(self._source_vocab.strings(xs[0][0][xs[0][0] != MorphoDataset.PAD].numpy(force=True))),
                "".join(self._target_vocab.strings(list(self.predict_step((xs[0][:1],)))[0].numpy(force=True)))))
        return result

    def test_step(self, xs, y):
        with torch.no_grad():
            y_pred = self(*xs)
            return self.compute_metrics(y_pred, y, *xs)

    def predict_step(self, xs):
        with torch.no_grad():
            for lemma in self(*xs):
                yield lemma[(lemma == MorphoDataset.EOW).cumsum(-1) == 0]


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: MorphoDataset.Dataset, training: bool) -> None:
        super().__init__(dataset)
        self._training = training

    def transform(self, example):
        return example["words"], example["lemmas"]

    def collate(self, batch):
        words, lemmas = zip(*batch)
        words = list(itertools.chain(*words))
        words = [torch.tensor([self.dataset.words.char_vocab.index(c) for c in w]) for w in words]
        words = torch.nn.utils.rnn.pad_sequence(words, batch_first=True)

        lemmas = list(itertools.chain(*lemmas))
        lemmas = [torch.tensor([self.dataset.lemmas.char_vocab.index(c) for c in l] + [MorphoDataset.EOW]) for l in
                  lemmas]
        lemmas = torch.nn.utils.rnn.pad_sequence(lemmas, batch_first=True)

        return ((words, lemmas), lemmas) if self._training else (words, lemmas)


def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    train = TrainableDataset(morpho.train, True).dataloader(args.batch_size, shuffle=True, num_workers=3)
    dev = TrainableDataset(morpho.dev, False).dataloader(args.batch_size, num_workers=3)
    test = TrainableDataset(morpho.test, False).dataloader(args.batch_size, num_workers=3)

    model = Model(args, morpho.train)
    optimizer = torch.optim.AdamW(model.parameters())
    early_stopping = KeepBestWeights(metric="dev:accuracy", patience=5, mode='max')

    model.configure(optimizer=optimizer,
                    loss=nn.CrossEntropyLoss(ignore_index=morpho.PAD),
                    metrics={"accuracy": torchmetrics.MeanMetric()},
                    logdir=logdir,
                    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                         eta_min=0,
                                                                         T_max=len(train) * args.epochs),
                    )

    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[early_stopping])
    model.load_state_dict(early_stopping.best_state_dict)

    correct = 0
    total = 0
    dev_predictions = iter(model.predict(dev, data_with_labels=True))

    for sentence_words, sentence_lemmas in zip(morpho.dev.words.strings, morpho.dev.lemmas.strings):
        for word, true_lemma in zip(sentence_words, sentence_lemmas):
            lemma_indices = next(dev_predictions)
            pred_lemma = "".join(morpho.dev.lemmas.char_vocab.strings(lemma_indices))

            valid_analyses = analyses.get(word)
            if valid_analyses:
                valid_lemmas = [a.lemma for a in valid_analyses]
                if pred_lemma not in valid_lemmas:
                    pred_lemma = valid_lemmas[0]

            if pred_lemma == true_lemma:
                correct += 1
            total += 1

    print(f"Dev Accuracy (with analyses): {100 * correct / total:.2f}%")

    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "lemmatizer_competition.txt"), "w", encoding="utf-8") as predictions_file:
        test_predictions = iter(model.predict(test, data_with_labels=True))

        for sentence in morpho.test.words.strings:
            for word in sentence:
                lemma_indices = next(test_predictions)
                pred_lemma = "".join(morpho.test.lemmas.char_vocab.strings(lemma_indices))

                valid_analyses = analyses.get(word)
                if valid_analyses:
                    valid_lemmas = [a.lemma for a in valid_analyses]
                    if pred_lemma not in valid_lemmas:
                        pred_lemma = valid_lemmas[0]

                print(pred_lemma, file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
