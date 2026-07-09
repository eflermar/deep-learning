#!/usr/bin/env python3
# Author: Martin Efler

import argparse
import os

import torchmetrics
from npfl138 import TensorOrTensors
from npfl138.callbacks import KeepBestWeights
from transformers import get_cosine_schedule_with_warmup

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress the LOAD REPORT with weight discrepancies.

import torch
import transformers
from torch import nn
import npfl138

npfl138.require_version("2526.10")
from npfl138.datasets.reading_comprehension_dataset import ReadingComprehensionDataset  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=4, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate.")


class SpanAccuracy(torchmetrics.Metric):
    def __init__(self, mode="exact"):
        super().__init__()
        # mode can be "start", "end", or "exact"
        self.mode = mode
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: TensorOrTensors, target: TensorOrTensors):
        start_logits, end_logits = preds
        start_target, end_target = target

        start_pred = start_logits.argmax(dim=-1)
        end_pred = end_logits.argmax(dim=-1)

        if self.mode == "start":
            match = start_pred == start_target
        elif self.mode == "end":
            match = end_pred == end_target
        else:
            match = (start_pred == start_target) & (end_pred == end_target)

        self.correct += match.sum()
        self.total += match.numel()

    def compute(self):
        return self.correct.float() / self.total


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: ReadingComprehensionDataset.Dataset, tokenizer, max_examples=None) -> None:
        flatten_dataset = []
        for paragraph in dataset.paragraphs:
            for qa in paragraph["qas"]:
                answers = qa.get("answers") or [None]
                flatten_dataset.append({"context": paragraph["context"], "question": qa["question"], "answers": answers})
        if max_examples is not None:
            flatten_dataset = flatten_dataset[:max_examples]
        super().__init__(flatten_dataset)
        self._tokenizer = tokenizer

    @staticmethod
    def transform(example):
        return example["context"], example["question"], example["answers"][0]

    def collate(self, batch):
        contexts, questions, answers_list = zip(*batch)
        encoding = self._tokenizer(contexts, questions, truncation="only_first", padding=True, return_offsets_mapping=True, return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        if answers_list[0] is not None:
            starts = [encoding.char_to_token(i, ans["start"]) for i, ans in enumerate(answers_list)]
            ends = [encoding.char_to_token(i, ans["start"] + len(ans["text"]) - 1) for i, ans in enumerate(answers_list)]
            if missing := sum(s is None for s in starts) > 0:
                print(missing)
            starts = [s if s is not None else -42 for s in starts]
            ends = [e if e is not None else -42 for e in ends]
            return (input_ids, attention_mask), (torch.tensor(starts), torch.tensor(ends))

        return (input_ids, attention_mask), (contexts, encoding["offset_mapping"])


class Model(npfl138.TrainableModule):
    def __init__(self, args, robeczech, sep_token_id=None):
        super().__init__()
        self.cross = nn.CrossEntropyLoss(ignore_index=-42)
        self.sep_token_id = sep_token_id
        self.robeczech = robeczech
        self.qa_outputs = nn.Linear(robeczech.config.hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.robeczech(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.qa_outputs(hidden)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        return start_logits, end_logits

    def predict(self, input_ids, attention_mask, contexts, offsets):
        start_logits, end_logits = self(input_ids, attention_mask)
        start_logits = start_logits.masked_fill(attention_mask == 0, -10000.0)
        end_logits = end_logits.masked_fill(attention_mask == 0, -10000.0)
        answers = []
        for i in range(len(contexts)):
            start_idx = start_logits[i].argmax().item()
            end_idx = end_logits[i].argmax().item()

            if start_idx > end_idx:
                answers.append("start_idx > end_idx")
                continue

            start_char = offsets[i][start_idx][0].item()
            end_char = offsets[i][end_idx][1].item()

            answers.append(contexts[i][start_char:end_char])
        return answers

    def compute_loss(self, y_pred: TensorOrTensors, y: TensorOrTensors, *xs: TensorOrTensors) -> torch.Tensor:
        start_logits, end_logits = y_pred
        target_starts, target_ends = y
        start_loss = self.cross(start_logits, target_starts)
        end_loss = self.cross(end_logits, target_ends)
        return (start_loss + end_loss) / 2


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the pre-trained RobeCzech model.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.AutoModel.from_pretrained("ufal/robeczech-base")

    # Load the data.
    dataset = ReadingComprehensionDataset()
    train = TrainableDataset(dataset.train, tokenizer).dataloader(args.batch_size, shuffle=True, num_workers=args.threads)
    dev = TrainableDataset(dataset.dev, tokenizer).dataloader(args.batch_size, num_workers=args.threads)
    test = TrainableDataset(dataset.test, tokenizer).dataloader(args.batch_size, num_workers=args.threads)

    model = Model(args, robeczech)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    total_steps = len(train) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    model.configure(
        optimizer=optimizer,
        metrics={
            "accuracy": SpanAccuracy(mode="exact"),
            "start_acc": SpanAccuracy(mode="start"),
            "end_acc": SpanAccuracy(mode="end"),
        },
        logdir=logdir,
        scheduler=scheduler,
    )

    early_stopping = KeepBestWeights(metric="dev:accuracy", patience=2, mode="max")
    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[early_stopping])
    model.load_state_dict(early_stopping.best_state_dict)
    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        with torch.inference_mode():
            for (input_ids, attention_mask), (contexts, offsets) in test:
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)
                contexts = contexts.to(model.device)
                offsets = offsets.to(model.device)
                for answer in model.predict(input_ids, attention_mask, contexts, offsets):
                    print(answer, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
