#!/usr/bin/env python3
# Author: Martin Efler
import argparse
import math
import os
from typing import Iterable

import npfl138
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision.transforms.v2 as v2
from npfl138 import STOP_TRAINING, TensorOrTensors
from npfl138.trainable_module import ProgressLogger
from torchvision.ops import sigmoid_focal_loss

from bboxes_utils import bboxes_from_rcnn, bboxes_training

npfl138.require_version("2526.6")
from npfl138.datasets.svhn import SVHN

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=60, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--weight_decay", default=0.001, type=float, help="Adam weight decay.")
parser.add_argument("--keep_thr", default=0.2, type=float, help="Threshold for keeping the prediction.")
parser.add_argument('--use_scheduler', action='store_true', help="Whether to use lr scheduler.")


def generate_anchors(feature_size: int = 14,
                     image_size: int = 224,
                     aspect_ratios: tuple = ((16, 16), (24, 16), (32, 24),
                                             (48, 32), (64, 48))) -> tuple[torch.Tensor, int]:
    anchor_size = image_size / feature_size  # 224 / 14 = 16 size of each grid cell
    grid = (torch.arange(feature_size, dtype=torch.float32) + .5) * anchor_size
    centers = torch.cartesian_prod(grid, grid)  # [num_centers, 2]
    ratios = torch.tensor(aspect_ratios, dtype=torch.float32)  # [num_ratios, 2]

    # Compute anchors with one stack and broadcasting
    anchors = torch.stack([
        centers[:, 0:1] - ratios[:, 0] / 2,  # y_min (Top)
        centers[:, 1:2] - ratios[:, 1] / 2,  # x_min (Left)
        centers[:, 0:1] + ratios[:, 0] / 2,  # y_max (Bottom)
        centers[:, 1:2] + ratios[:, 1] / 2,  # x_max (Right)
    ], dim=-1)

    # flatten from [num_centers, num_ratios, 4] to [num_centers * num_ratios, 4]
    anchors = torch.relu(anchors.view(-1, 4))

    return anchors, ratios.shape[0]


ANCHORS, ANCHORS_PER_LOC = generate_anchors()


class SVHNSegmentationDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, image_transform, augment):
        super().__init__(dataset)
        self.image_transform = image_transform
        self.augment = augment

    def transform(self, item):
        scale = 224.0 / item["image"].shape[-1]
        image = item["image"]
        if self.augment:
            aug = v2.Compose([
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                v2.RandomRotation(degrees=15),
                v2.GaussianBlur(kernel_size=3),
            ])
            image = aug(image)

        if len(item["bboxes"]) > 0:
            resized_bboxes = item["bboxes"].float() * scale
            anchor_classes, anchor_bboxes = bboxes_training(
                anchors=ANCHORS,
                gold_bboxes=resized_bboxes,
                gold_classes=item["classes"],
                iou_threshold=.5,
            )
        else:
            anchor_classes = torch.zeros(len(ANCHORS), dtype=torch.long)
            anchor_bboxes = torch.zeros(len(ANCHORS), 4, dtype=torch.float32)

        target = {"bboxes": anchor_bboxes, "classes": anchor_classes, "orig_size": item["image"].shape[1]}
        return self.image_transform(image), target


class SVHNModel(npfl138.TrainableModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)

        self.bbox_head = nn.Sequential(
            *[layer for _ in range(4) for layer in (nn.LazyConv2d(256, 3, padding='same'), nn.ReLU(inplace=True))],
            nn.Conv2d(256, ANCHORS_PER_LOC * 4, kernel_size=3, padding='same'),
        )

        self.cls_head = nn.Sequential(
            *[layer for _ in range(4) for layer in (nn.LazyConv2d(256, 3, padding='same'), nn.ReLU(inplace=True))],
            nn.Conv2d(256, ANCHORS_PER_LOC * SVHN.LABELS, kernel_size=3, padding='same'),
        )

        # fix for focal loss (bias to only 1% of anchors have objects)
        pi = 0.01
        bias_value = -math.log((1 - pi) / pi)
        torch.nn.init.constant_(self.cls_head[-1].bias, bias_value)
        torch.nn.init.normal_(self.cls_head[-1].weight, std=0.01)

    def forward(self, x):
        out, feats = self.backbone.forward_intermediates(x)
        bbox_gampley = self.bbox_head(feats[-2])
        cls_gampley = self.cls_head(feats[-2])
        return bbox_gampley, cls_gampley

    @staticmethod
    def _reshape_pred(x, last_channel):
        B = x.shape[0]
        return x.permute(0, 2, 3, 1).contiguous().view(B, -1, last_channel)

    @staticmethod
    def _classification_loss(cls_pred, cls_true, num_classes, num_pos):
        cls_true_oh = F.one_hot(cls_true, num_classes=num_classes + 1).float()
        cls_true_oh = cls_true_oh[..., 1:]

        loss = sigmoid_focal_loss(cls_pred, cls_true_oh, reduction="sum")

        return loss / num_pos

    @staticmethod
    def _bbox_loss(bb_pred, bb_true, pos_mask, num_pos):
        if pos_mask.any():
            loss = F.smooth_l1_loss(bb_pred[pos_mask], bb_true[pos_mask], reduction="sum")
            return loss / num_pos

        return bb_pred.sum() * 0.0

    def compute_loss(self, y_pred, y_true, *inputs):
        bb_pred, cls_pred = y_pred
        bb_true = y_true["bboxes"]
        cls_true = y_true["classes"]

        bb_pred = self._reshape_pred(bb_pred, 4)
        cls_pred = self._reshape_pred(cls_pred, SVHN.LABELS)

        pos_mask = cls_true > 0  # (B, N)
        num_pos = pos_mask.sum().clamp(min=1).float()

        cls_loss = self._classification_loss(cls_pred, cls_true, SVHN.LABELS, num_pos)
        bbox_loss = self._bbox_loss(bb_pred, bb_true, pos_mask, num_pos)

        return cls_loss + bbox_loss

    def predict(self,
                dataloader: torch.utils.data.DataLoader,
                *,
                data_with_labels: bool = False,
                whole_batches: bool = False,
                as_numpy: bool = False,
                console: int | None = None,
                keep_thr: float = .2) -> Iterable[TensorOrTensors]:

        self.eval()
        with torch.inference_mode():
            for batch in ProgressLogger(dataloader, "Prediction"):
                if data_with_labels:
                    images, targets = batch
                else:
                    images = batch[0]

                bbox_preds, cls_preds = self(images.to(self.device))

                B = bbox_preds.shape[0]
                bbox_preds = self._reshape_pred(bbox_preds, 4)
                cls_preds = self._reshape_pred(cls_preds, SVHN.LABELS)

                for i in range(B):
                    cls_pred = cls_preds[i]
                    reg_pred = bbox_preds[i]

                    probs = torch.sigmoid(cls_pred)
                    scores, class_ids = probs.max(dim=-1)

                    keep = scores > keep_thr
                    if keep.sum() == 0:
                        yield torch.tensor([]), torch.tensor([])
                        continue

                    scores, class_ids, reg_pred = scores[keep], class_ids[keep], reg_pred[keep]

                    # Decode from RCNN space
                    bboxes = bboxes_from_rcnn(ANCHORS.to(self.device)[keep], reg_pred).clamp(0, 224)

                    # Apply NMS
                    bboxes_xyxy = bboxes[:, [0, 1, 2, 3]]
                    keep_idx = ops.batched_nms(bboxes_xyxy, scores, class_ids, 0.5)

                    # Scale back to original image size
                    orig_size = targets["orig_size"][i].item() if data_with_labels else 224
                    scale = orig_size / 224.0

                    yield class_ids[keep_idx], bboxes[keep_idx] * scale


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, SIZE, SIZE]` tensor of `torch.uint8` values in [0-255] range,
    # - "classes", a `[num_digits]` PyTorch vector with classes of image digits,
    # - "bboxes", a `[num_digits, 4]` PyTorch vector with bounding boxes of image digits.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    svhn = SVHN(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # takes a lot of time, orig passes baseline so cba
    # efficientnetv2_b0 = timm.create_model("beitv2_large_patch16_224", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Resize(224, interpolation=v2.InterpolationMode(efficientnetv2_b0.pretrained_cfg["interpolation"])),
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    train = SVHNSegmentationDataset(svhn.train, image_transform=preprocessing, augment=True).dataloader(
        batch_size=args.batch_size,
        num_workers=args.threads,
        shuffle=True)
    dev = SVHNSegmentationDataset(svhn.dev, image_transform=preprocessing, augment=False).dataloader(
        batch_size=args.batch_size,
        num_workers=args.threads)
    test = SVHNSegmentationDataset(svhn.test, image_transform=preprocessing, augment=False).dataloader(
        batch_size=args.batch_size,
        num_workers=args.threads)

    model = SVHNModel(efficientnetv2_b0)

    class EarlyStopping:
        def __init__(self, patience: int = 3, save_path: str = "best_model.pt", save_to_disk: bool = False,
                     keep_thr: float = .2):
            self.best_acc = -1
            self.patience = patience
            self.counter = 0
            self.best_model = None
            self.save_path = save_path
            self.save_to_disk = save_to_disk
            self.keep_thr = keep_thr

        def __call__(self, module, epoch, logs):
            acc = self.evaluate_dev_iou(module, logs, self.keep_thr)

            if acc >= self.best_acc:
                self.best_acc = acc
                self.counter = 0
                self.best_model = module.state_dict()
                if self.save_to_disk:
                    torch.save(module.state_dict(), self.save_path)
                    print(f"Model saved to {self.save_path}")
            else:
                self.counter += 1
                print(f"Loosing patience... {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    print(f"Early stopping after {self.counter} epochs...")
                    return STOP_TRAINING
            return None

        @staticmethod
        def evaluate_dev_iou(module, logs, keep_thr: float):
            predictions = []
            for pred_cls, pred_bboxes in module.predict(dev, data_with_labels=True, keep_thr=keep_thr):
                predictions.append((pred_cls.cpu().tolist(), pred_bboxes.cpu().tolist()))

            accuracy = SVHN.evaluate(svhn.dev, predictions)
            if logs is not None:
                logs["dev_accuracy"] = accuracy
            return accuracy

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.configure(
        optimizer=optimizer,
        logdir=logdir,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.00001,
                                                             T_max=len(
                                                                 train) * args.epochs) if args.use_scheduler else None,
    )
    early_stopping = EarlyStopping(5)
    model.fit(train, epochs=args.epochs, log_graph=True, callbacks=[early_stopping], )

    if early_stopping.best_model is not None:
        print(80 * '=')
        print(f"Best model restored from memory.")
        print(f"Best dev acc = {early_stopping.best_acc:.2%}")
        model.load_state_dict(early_stopping.best_model)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        for predicted_classes, predicted_bboxes in model.predict(test, data_with_labels=True, keep_thr=args.keep_thr):
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [int(label)] + list(map(float, bbox))
            print(*output, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
