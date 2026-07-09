#!/usr/bin/env python3
# Author: Martin Efler
import argparse
import os

import npfl138
import torch
import torchmetrics
from npfl138.callbacks import KeepBestWeights
from torch import nn
from transformers import get_cosine_schedule_with_warmup

npfl138.require_version("2526.11")
from npfl138.datasets.modelnet import ModelNet

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=2**5, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=32, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")


class Dataset(npfl138.TransformedDataset):
    def __init__(self, dataset: ModelNet.Dataset) -> None:
        super().__init__(dataset)

    @staticmethod
    def transform(example):
        return example["grid"].float(), example["label"]


class DenseLayer(nn.Module):
    def __init__(self, growth_rate=16):
        super().__init__()
        self.bot = nn.Sequential(
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
            nn.LazyConv3d(4 * growth_rate, kernel_size=1, bias=False),
        )
        self.main = nn.Sequential(
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
            nn.LazyConv3d(growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.main(self.bot(x))], dim=1)


class TransitionLayer(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.trans = nn.Sequential(
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
            nn.LazyConv3d(out_channels, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.trans(x)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        self.model = nn.Sequential(
            nn.LazyConv3d(32, kernel_size=3, padding=1, bias=False),
            *[DenseLayer(8) for _ in range(4)], TransitionLayer(48),
            *[DenseLayer(8) for _ in range(6)], TransitionLayer(96),
            *[DenseLayer(8) for _ in range(8)], TransitionLayer(128),
            *[DenseLayer(8) for _ in range(6)],

            nn.LazyBatchNorm3d(), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.LazyLinear(10)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data.
    modelnet = ModelNet(args.modelnet)
    train = torch.utils.data.DataLoader(Dataset(modelnet.train), args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(modelnet.dev), args.batch_size)
    test = torch.utils.data.DataLoader(Dataset(modelnet.test), args.batch_size)

    model = Model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    model.configure(optimizer=optimizer,
                    loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
                    metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=modelnet.LABELS)},
                    logdir=logdir,
                    scheduler=scheduler,
                    )

    early_stopping = KeepBestWeights(metric="dev:accuracy", patience=4, mode='max')
    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[early_stopping])
    model.load_state_dict(early_stopping.best_state_dict)
    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        for prediction in model.predict(test, data_with_labels=True):
            print(prediction.argmax().item(), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
