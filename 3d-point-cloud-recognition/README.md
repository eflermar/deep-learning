# 3D Object Recognition (ModelNet)

Classifies 3D CAD models from the ModelNet dataset, represented as voxel occupancy grids, into
10 object categories.

## Approach

- **Input**: 32^3 voxel grids (`ModelNet(32)`).
- **Architecture**: a 3D DenseNet — the same dense-block/transition-layer pattern as the CIFAR
  project, but with `Conv3d`/`BatchNorm3d`/`AvgPool3d` throughout: 4/6/8/6 dense layers per
  stage, growth rate 8, channel counts 32 -> 48 -> 96 -> 128.
- **Optimization**: AdamW with a cosine schedule and linear warmup (10% of total steps, via
  HuggingFace `transformers.get_cosine_schedule_with_warmup`), label smoothing (0.1), best-weight
  checkpointing on dev accuracy with patience-based early stopping.

## Running it

```bash
pip install -r requirements.txt
python 3d_recognition.py --epochs 20 --modelnet 32
```

Uses [`npfl138`](https://pypi.org/project/npfl138/) for the training loop and the `ModelNet`
dataset loader, which downloads the voxelized data automatically.

*No training log survived from the original coursework runs for this project (only the raw
test-set predictions were saved), so no dev-accuracy number is quoted here — the architecture and
training setup above reflect the actual code that was submitted and graded.*
