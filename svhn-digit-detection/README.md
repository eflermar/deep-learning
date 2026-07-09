# SVHN Multi-Digit Detection

Anchor-based object detector that finds and classifies house-number digits in Street View House
Numbers (SVHN) images — a simplified single-stage detector in the spirit of RetinaNet.

## Approach

- **Backbone**: pretrained EfficientNetV2-B0 (via `timm`), frozen, used as a feature extractor at
  14x14 resolution (`forward_intermediates`).
- **Anchors**: a fixed grid of 5 aspect ratios per spatial location, matched to ground-truth boxes
  by IoU (`bboxes_utils.py` — vectorized best-anchor-per-object and best-object-per-anchor
  assignment, threshold 0.5).
- **Heads**: two parallel conv stacks on top of the backbone features — a classification head
  (sigmoid focal loss, bias-initialized per Lin et al. 2017 so ~1% of anchors start as
  "foreground") and a bounding-box regression head (smooth L1 loss, R-CNN-style
  center/size parametrization).
- **Inference**: per-anchor sigmoid scores thresholded (`--keep_thr`), decoded back to
  image-space boxes, then class-aware NMS (`torchvision.ops.batched_nms`).
- **Training**: AdamW, optional cosine LR schedule, early stopping on dev mean-IoU-matched
  accuracy (`SVHN.evaluate`).

## Results

The best full run (60 epochs, AdamW, lr 1e-4, cosine schedule) converged with:

| Metric | Value |
|---|---|
| Dev loss (final, focal + smooth-L1) | ~0.35 |

Evaluation on SVHN's actual objective — mean digit-sequence accuracy after IoU-matching detected
boxes to ground truth — was tracked live during training (`SVHN.evaluate`, used for early
stopping) but only the loss curve survived in the saved logs; the final competition submission
used greedy-searched hyperparameters (`--keep_thr 0.2`, cosine schedule) on top of this setup.

## Running it

```bash
pip install -r requirements.txt
python svhn_competition.py --epochs 60 --use_scheduler
```

Uses [`npfl138`](https://pypi.org/project/npfl138/) for the training loop and `SVHN` dataset
loader; `bboxes_utils.py` provides the shared anchor/IoU math and has its own unit tests
(`python bboxes_utils.py` runs them via `unittest`).
