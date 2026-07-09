#!/usr/bin/env python3
# Author: Martin Efler

import argparse
import os
import torch
import torch.nn as nn
import torchmetrics
from torchvision.transforms import v2

import npfl138
npfl138.require_version("2526.4")
from npfl138.datasets.cifar10 import CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--weight_decay", default=0.0001, type=float, help="Adam weight decay.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset: CIFAR10.Dataset, is_train: bool = True):
        self.images = cifar_dataset.data["images"]
        self.labels = cifar_dataset.data["labels"]
        self.is_train = is_train
        if self.is_train:
            self.transform = v2.Compose([
                v2.RandomCrop(32, padding=4, padding_mode='reflect'),
                v2.RandomHorizontalFlip(),
                v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
                v2.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index] / 255.0, self.labels[index]

    def __getitems__(self, indices):
        return (self.transform(self.images[indices]) / 255.0) if self.is_train else self.images[indices] / 255.0, \
            self.labels[indices]

    @staticmethod
    def collate(batch):
        return batch


class DenseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bot = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(128, kernel_size=1, bias=False)
        )
        self.main = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(32, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.main(self.bot(x))], dim=1)


class TransitionLayer(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.trans = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.trans(x)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        self.model = nn.Sequential(
                    nn.LazyConv2d(64, kernel_size=3, padding=1, bias=False),
                    
                    *[DenseLayer() for _ in range(6)],
                    TransitionLayer(out_channels=128),
                    
                    *[DenseLayer() for _ in range(12)],
                    TransitionLayer(out_channels=256),
                    
                    *[DenseLayer() for _ in range(24)],
                    TransitionLayer(out_channels=512),
                    
                    *[DenseLayer() for _ in range(16)],
                    
                    nn.LazyBatchNorm2d(),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.LazyLinear(10)
                )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data.
    cifar = CIFAR10()
    train = torch.utils.data.DataLoader(Dataset(cifar.train, is_train=True), args.batch_size, collate_fn=Dataset.collate, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(cifar.dev, is_train=False), args.batch_size, collate_fn=Dataset.collate)
    test = torch.utils.data.DataLoader(Dataset(cifar.test, is_train=False), args.batch_size, collate_fn=Dataset.collate)

    model = Model(args)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.learning_rate, 
        momentum=0.9, 
        weight_decay=args.weight_decay,
        nesterov=True
    )
    model.configure(
        optimizer=optimizer,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=10)},
        logdir=logdir,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.0001 ,T_max=len(train) * args.epochs)
    )
    model.fit(train, dev=dev, epochs=args.epochs, log_config=vars(args), log_graph=False, console=3)
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for prediction in model.predict(test, data_with_labels=True):
            print(prediction.argmax().item(), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)