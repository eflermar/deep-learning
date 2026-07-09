#!/usr/bin/env python3
# Author: Martin Efler
import argparse
import unittest
from math import log
from typing import Callable

import torch

# Bounding boxes and anchors are expected to be PyTorch tensors,
# where the last dimension has size 4.

# For bounding boxes in pixel coordinates, the 4 values correspond to
# (top, left, bottom, right) with top <= bottom and left <= right.
TOP: int = 0
LEFT: int = 1
BOTTOM: int = 2
RIGHT: int = 3


def bboxes_area(bboxes: torch.Tensor) -> torch.Tensor:
    """Compute area of given set of bboxes.

    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    If the bboxes.shape is [..., 4], the output shape is bboxes.shape[:-1].
    """
    return torch.relu(bboxes[..., BOTTOM] - bboxes[..., TOP]) \
        * torch.relu(bboxes[..., RIGHT] - bboxes[..., LEFT])


def bboxes_iou(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Compute IoU of corresponding pairs from two sets of bboxes `xs` and `ys`.

    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    Note that broadcasting is supported, so passing inputs with
    `xs.shape=[num_xs, 1, 4]` and `ys.shape=[1, num_ys, 4]` produces an output with
    shape `[num_xs, num_ys]`, computing IoU for all pairs of bboxes from `xs` and `ys`.
    Formally, the output shape is `torch.broadcast_shapes(xs.shape, ys.shape)[:-1]`.
    """
    intersections = torch.stack([
        torch.maximum(xs[..., TOP], ys[..., TOP]),
        torch.maximum(xs[..., LEFT], ys[..., LEFT]),
        torch.minimum(xs[..., BOTTOM], ys[..., BOTTOM]),
        torch.minimum(xs[..., RIGHT], ys[..., RIGHT]),
    ], dim=-1)

    xs_area, ys_area, intersections_area = bboxes_area(xs), bboxes_area(ys), bboxes_area(intersections)

    return intersections_area / (xs_area + ys_area - intersections_area)


def bboxes_to_rcnn(anchors: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
    """Convert `bboxes` to a R-CNN-like representation relative to `anchors`.

    The `anchors` and `bboxes` are arrays of four-tuples (top, left, bottom, right);
    you can use the TOP, LEFT, BOTTOM, RIGHT constants as indices of the
    respective coordinates.

    The resulting representation of a single bbox is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - log(bbox_height / anchor_height)
    - log(bbox_width / anchor_width)

    If the `anchors.shape` is `[anchors_len, 4]` and `bboxes.shape` is `[anchors_len, 4]`,
    the output shape is `[anchors_len, 4]`.
    """

    anchor_heights = anchors[..., BOTTOM] - anchors[..., TOP]
    anchor_widths = anchors[..., RIGHT] - anchors[..., LEFT]
    anchor_y_centers = anchors[..., TOP] + anchor_heights / 2
    anchor_x_centers = anchors[..., LEFT] + anchor_widths / 2

    bbox_heights = bboxes[..., BOTTOM] - bboxes[..., TOP]
    bbox_widths = bboxes[..., RIGHT] - bboxes[..., LEFT]
    bbox_y_centers = bboxes[..., TOP] + bbox_heights / 2
    bbox_x_centers = bboxes[..., LEFT] + bbox_widths / 2

    return torch.stack([(bbox_y_centers - anchor_y_centers) / anchor_heights,
                        (bbox_x_centers - anchor_x_centers) / anchor_widths,
                        torch.log(bbox_heights / anchor_heights),
                        torch.log(bbox_widths / anchor_widths)], dim=-1)


def bboxes_from_rcnn(anchors: torch.Tensor, rcnns: torch.Tensor) -> torch.Tensor:
    """Convert R-CNN-like representation relative to `anchor` to a `bbox`.

    If the `anchors.shape` is `[anchors_len, 4]` and `rcnns.shape` is `[anchors_len, 4]`,
    the output shape is `[anchors_len, 4]`.
    """
    anchor_heights = anchors[..., BOTTOM] - anchors[..., TOP]
    anchor_widths = anchors[..., RIGHT] - anchors[..., LEFT]
    anchor_y_centers = anchors[..., TOP] + anchor_heights / 2
    anchor_x_centers = anchors[..., LEFT] + anchor_widths / 2

    bbox_y_centers = rcnns[..., TOP] * anchor_heights + anchor_y_centers
    bbox_x_centers = rcnns[..., LEFT] * anchor_widths + anchor_x_centers
    bbox_heights = torch.exp(rcnns[..., BOTTOM]) * anchor_heights
    bbox_widths = torch.exp(rcnns[..., RIGHT]) * anchor_widths

    return torch.stack([
        bbox_y_centers - bbox_heights / 2,
        bbox_x_centers - bbox_widths / 2,
        bbox_y_centers + bbox_heights / 2,
        bbox_x_centers + bbox_widths / 2], dim=-1)


def bboxes_training(
        anchors: torch.Tensor, gold_classes: torch.Tensor, gold_bboxes: torch.Tensor, iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute training data for object detection.

    Arguments:
    - `anchors` is an array of four-tuples (top, left, bottom, right)
    - `gold_classes` is an array of zero-based classes of the gold objects
    - `gold_bboxes` is an array of four-tuples (top, left, bottom, right)
      of the gold objects
    - `iou_threshold` is a given threshold

    Returns:
    - `anchor_classes` contains for every anchor either 0 for background
      (if no gold object is assigned) or `1 + gold_class` if a gold object
      with `gold_class` is assigned to it
    - `anchor_bboxes` contains for every anchor a four-tuple
      `(center_y, center_x, height, width)` representing the gold bbox of
      a chosen object using parametrization of R-CNN; zeros if no gold object
      was assigned to the anchor
    If the `anchors` shape is `[anchors_len, 4]`, the `anchor_classes` shape
    is `[anchors_len]` and the `anchor_bboxes` shape is `[anchors_len, 4]`.

    Algorithm:
    - First, for each gold object, assign it to an anchor with the largest IoU
      (the anchor with the smaller index if there are several). In case several
      gold objects are assigned to a single anchor, use the gold object with
      the smaller index.
    - For each unused anchor, find the gold object with the largest IoU
      (again the gold object with the smaller index if there are several),
      and if the IoU is >= iou_threshold, assign the object to the anchor.
    """
    anchor_classes = torch.zeros(len(anchors), dtype=torch.long)
    anchor_bboxes = torch.zeros(len(anchors), 4)
    all_ious = bboxes_iou(anchors.unsqueeze(1), gold_bboxes.unsqueeze(0))
    best_ious, best_golds_for_anchors = all_ious.max(dim=1)

    # Assign every anchor whose best-matching gold object clears the IoU threshold.
    threshold_mask = best_ious >= iou_threshold
    gold_indices = best_golds_for_anchors[threshold_mask]
    anchor_classes[threshold_mask] = gold_classes[gold_indices] + 1
    anchor_bboxes[threshold_mask] = bboxes_to_rcnn(anchors[threshold_mask],
                                                   gold_bboxes[gold_indices])

    # Overwrite with each gold object's single best anchor, processing gold objects
    # in reverse so that the lowest-indexed gold object wins ties on a shared anchor.
    best_anchor_per_gold = all_ious.argmax(dim=0)
    reverse_indices = torch.arange(len(gold_bboxes) - 1, -1, -1)  # to [len, ..., 0]

    best_anchors_rev = best_anchor_per_gold[reverse_indices]
    anchor_classes[best_anchors_rev] = gold_classes[reverse_indices] + 1
    anchor_bboxes[best_anchors_rev] = bboxes_to_rcnn(
        anchors[best_anchors_rev],
        gold_bboxes[reverse_indices],
    )

    return anchor_classes, anchor_bboxes


def main(args: argparse.Namespace) -> tuple[Callable, Callable, Callable]:
    return bboxes_to_rcnn, bboxes_from_rcnn, bboxes_training


class Tests(unittest.TestCase):
    def test_bboxes_to_from_rcnn(self):
        data = [
            [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
            [[0, 0, 10, 10], [5, 0, 15, 10], [.5, 0, 0, 0]],
            [[0, 0, 10, 10], [0, 5, 10, 15], [0, .5, 0, 0]],
            [[0, 0, 10, 10], [0, 0, 20, 30], [.5, 1, log(2), log(3)]],
            [[0, 9, 10, 19], [2, 10, 5, 16], [-0.15, -0.1, -1.20397, -0.51083]],
            [[5, 3, 15, 13], [7, 7, 10, 9], [-0.15, 0, -1.20397, -1.60944]],
            [[7, 6, 17, 16], [9, 10, 12, 13], [-0.15, 0.05, -1.20397, -1.20397]],
            [[5, 6, 15, 16], [7, 7, 10, 10], [-0.15, -0.25, -1.20397, -1.20397]],
            [[6, 3, 16, 13], [8, 5, 12, 8], [-0.1, -0.15, -0.91629, -1.20397]],
            [[5, 2, 15, 12], [9, 6, 12, 8], [0.05, 0, -1.20397, -1.60944]],
            [[2, 10, 12, 20], [6, 11, 8, 17], [0, -0.1, -1.60944, -0.51083]],
            [[10, 9, 20, 19], [12, 13, 17, 16], [-0.05, 0.05, -0.69315, -1.20397]],
            [[6, 7, 16, 17], [10, 11, 12, 14], [0, 0.05, -1.60944, -1.20397]],
            [[2, 2, 12, 12], [3, 5, 8, 8], [-0.15, -0.05, -0.69315, -1.20397]],
        ]
        # First run on individual anchors, and then on all together
        for anchors, bboxes, rcnns in [map(lambda x: [x], row) for row in data] + [zip(*data)]:
            anchors, bboxes, rcnns = [torch.tensor(data, dtype=torch.float32) for data in [anchors, bboxes, rcnns]]
            torch.testing.assert_close(bboxes_to_rcnn(anchors, bboxes), rcnns, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(bboxes_from_rcnn(anchors, rcnns), bboxes, atol=1e-3, rtol=1e-3)

    def test_bboxes_training(self):
        anchors = torch.tensor([[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]])
        for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
            [[1], [[14, 14, 16, 16]], [0, 0, 0, 2], [[0, 0, 0, 0]] * 3 + [[0, 0, log(.2), log(.2)]], 0.5],
            [[2], [[0, 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, log(2), log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
            [[2], [[0, 0, 20, 20]], [3, 3, 3, 3],
             [[y, x, log(2), log(2)] for y in [.5, -.5] for x in [.5, -.5]], 0.24],
            [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 0, 1],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [-0.35, -0.45, 0.53062, 0.40546]], 0.5],
            [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 2, 1],
             [[0, 0, 0, 0], [0, 0, 0, 0], [-0.1, 0.6, -0.22314, 0.69314], [-0.35, -0.45, 0.53062, 0.40546]], 0.3],
            [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 1, 2, 1],
             [[0, 0, 0, 0], [0.65, -0.45, 0.53062, 0.40546], [-0.1, 0.6, -0.22314, 0.69314],
              [-0.35, -0.45, 0.53062, 0.40546]], 0.17],
        ]:
            gold_classes, anchor_classes = torch.tensor(gold_classes), torch.tensor(anchor_classes)
            gold_bboxes, anchor_bboxes = torch.tensor(gold_bboxes), torch.tensor(anchor_bboxes)
            computed_classes, computed_bboxes = bboxes_training(anchors, gold_classes, gold_bboxes, iou)
            torch.testing.assert_close(computed_classes, anchor_classes, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(computed_bboxes, anchor_bboxes, atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
