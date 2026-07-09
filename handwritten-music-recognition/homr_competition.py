#!/usr/bin/env python3
# Author: Martin Efler
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T

import torchvision.transforms.functional as TF
import npfl138
from npfl138.callbacks import KeepBestWeights
from transformers import get_cosine_schedule_with_warmup

import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')  # avoid the fd limit

npfl138.require_version("2526.12")
from npfl138.datasets.homr_dataset import HOMRDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=90, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=6, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=5e-4, type=float)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument("--rnn_layers", default=4, type=int)
parser.add_argument("--dropout", default=0.3, type=float)
parser.add_argument("--patience", default=12, type=int)
parser.add_argument("--weight_decay", default=0.01, type=float)

VOCAB_SIZE = 938

train_augmentation = T.Compose([
    T.RandomAffine(degrees=2.5, translate=(0.02, 0.04), scale=(0.85, 1.15),
                   shear=8, fill=255),
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    T.RandomErasing(p=0.15, scale=(0.005, 0.02), ratio=(0.3, 3.3), value=255),
    T.RandomApply([T.ElasticTransform(alpha=30.0, sigma=4.0, fill=255)], p=0.3),
])


class Dataset(npfl138.TransformedDataset):
    def __init__(self, dataset: HOMRDataset, augmentation_fn=None) -> None:
        super().__init__(dataset)
        self._augmentation_fn = augmentation_fn

    def transform(self, example):
        img = example["image"]
        _, H, W = img.shape
        new_W = max(1, int(round(W * 64 / H)))
        img = TF.resize(img, [64, new_W], antialias=True)
        if self._augmentation_fn is not None:
            img = self._augmentation_fn(img)
        img = img.float() / 255.0
        return img, example["marks"]

    def collate(self, batch):
        images, marks = zip(*batch)
        widths = torch.tensor([img.shape[-1] for img in images], dtype=torch.long)
        lengths = torch.tensor([m.shape[0] for m in marks], dtype=torch.long)
        max_w, max_l = int(widths.max()), int(lengths.max())
        imgs = torch.stack([F.pad(img, (0, max_w - img.shape[-1])) for img in images])
        tgts = torch.stack([F.pad(m, (0, max_l - m.shape[0]), value=0) for m in marks])
        return imgs, tgts


class CTCLossWrapper(nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=True)

    def forward(self, y_pred, y_true):
        log_probs = y_pred.log_softmax(-1).transpose(0, 1)
        B, T = y_pred.size(0), y_pred.size(1)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=y_pred.device)
        target_lengths = (y_true != 0).sum(dim=1)
        return self.ctc(log_probs, y_true, input_lengths, target_lengths)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.hidden = args.hidden
        self.rnn_layers = args.rnn_layers
        self.dropout = args.dropout

        def block(in_c, out_c, n_convs, pool):
            layers = []
            for i in range(n_convs):
                layers += [
                    nn.Conv2d(in_c if i == 0 else out_c, out_c, 3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.MaxPool2d(pool))
            return nn.Sequential(*layers)

        self.cnn = nn.Sequential(
            block(1, 32, 1, (2, 2)),
            block(32, 64, 1, (2, 1)),
            block(64, 128, 2, (2, 1)),
            block(128, 256, 2, (2, 1)),
            block(256, 256, 2, (4, 1)),
        )

        self.rnn = nn.GRU(
            input_size=256, hidden_size=self.hidden, num_layers=self.rnn_layers,
            bidirectional=True, dropout=self.dropout, batch_first=True,
        )

        self.head = nn.Linear(2 * self.hidden, VOCAB_SIZE)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.head(x)

    @staticmethod
    def ctc_decode_logits(logits, blank=0):
        """CTC decode: argmax, drop repeats, drop blanks."""
        ids = logits.argmax(-1).cpu().tolist()
        out = []
        for row in ids:
            seq, prev = [], -1
            for t in row:
                if t != blank and t != prev:
                    seq.append(t)
                prev = t
            out.append(seq)
        return out

    def predict_step(self, xs):
        x = xs[0]
        with torch.no_grad():
            logits = self(x)
        yield from self.ctc_decode_logits(logits)

    def compute_metrics(self, y_pred, y, *xs):
        decoded = self.ctc_decode_logits(y_pred)
        golds = y.cpu().tolist()
        for metric in self.metrics.values():
            metric.update(decoded, golds)
        return self.metrics


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[1, HEIGHT, WIDTH]` tensor of `torch.uint8` values in [0-255] range,
    # - "marks", a `[num_marks]` tensor with indices of marks on the image.
    # Using `decode_on_demand=True` loads just the raw dataset (~500MB of undecoded PNG images)
    # and then decodes them on every access. Using `decode_on_demand=False` decodes the images
    # during loading, resulting in much faster access, but requires ~5GB of memory.
    homr = HOMRDataset(decode_on_demand=False)
    train = Dataset(homr.train, augmentation_fn=train_augmentation).dataloader(args.batch_size, shuffle=True,
                                                                               seed=args.seed, num_workers=args.threads)
    dev = Dataset(homr.dev).dataloader(args.batch_size)
    test = Dataset(homr.test).dataloader(args.batch_size)

    model = Model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    model.configure(optimizer=optimizer,
                    loss=CTCLossWrapper(blank=0),
                    metrics={"edit_distance": HOMRDataset.EditDistanceMetric(ignore_index=0)},
                    logdir=logdir,
                    scheduler=scheduler,
                    )

    early_stopping = KeepBestWeights(metric="dev:edit_distance", patience=args.patience, mode='min')
    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[early_stopping], console=2)
    model.load_state_dict(early_stopping.best_state_dict)
    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test)
        for sequence in predictions:
            print(" ".join(HOMRDataset.MARKS_VOCAB.strings(sequence)), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
