# CIFAR-10 Image Classification

Image classifier for CIFAR-10 (10 classes, 32x32 RGB images), built around a custom
DenseNet-style CNN trained from scratch.

## Approach

- **Architecture**: a DenseNet variant — dense blocks of `BatchNorm -> ReLU -> 1x1 conv ->
  BatchNorm -> ReLU -> 3x3 conv` with channel concatenation, separated by transition layers
  (`1x1 conv` + average pooling) that control channel growth. 6/12/24/16 dense layers per stage,
  growth rate 32.
- **Augmentation**: random crop (reflect padding), random horizontal flip, AutoAugment
  (CIFAR-10 policy), and random erasing — applied batch-wise via `torchvision.transforms.v2` in
  a custom `__getitems__` for speed.
- **Optimization**: SGD with Nesterov momentum, cosine-annealed learning rate, label smoothing
  (0.1) in the cross-entropy loss.

## Results

Final training run (200 epochs, batch size 64, lr 0.1, weight decay 1e-4):

| Metric | Value |
|---|---|
| Dev accuracy (best, epoch 188) | **97.06%** |
| Dev accuracy (final, epoch 200) | 97.02% |
| Train accuracy (final) | ~97-98% |

## Running it

```bash
pip install -r requirements.txt
python cifar_competition.py --epochs 200
```

Uses [`npfl138`](https://pypi.org/project/npfl138/), a PyTorch training-loop wrapper
(`TrainableModule`, dataset helpers) from the NPFL138 Deep Learning course at Charles University.
`CIFAR10` data downloads automatically on first run.
