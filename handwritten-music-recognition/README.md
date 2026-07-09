# Handwritten Optical Music Recognition (HOMR)

Reads a photographed line of handwritten sheet music and transcribes it into a sequence of music
symbols ("marks") — a CTC-based sequence recognition model, structurally similar to handwritten
text recognition.

## Approach

- **Preprocessing**: images resized to a fixed height (64px) with proportional width, normalized
  to [0, 1].
- **Augmentation**: random affine transforms (small rotation/translation/scale/shear), color
  jitter, Gaussian blur, random erasing, and occasional elastic distortion — applied only at
  training time, tuned to look like natural handwriting/scan variation without destroying symbol
  shape.
- **Architecture**: a CNN feature extractor (5 conv blocks, progressively downsampling height
  while preserving width resolution) feeding a 4-layer bidirectional GRU, followed by a linear
  classifier over a 938-symbol vocabulary.
- **Loss/decoding**: CTC loss (blank-token alignment-free training), greedy CTC decoding
  (argmax + collapse repeats + drop blanks) at inference.
- **Training**: AdamW with cosine schedule + warmup, best-weight checkpointing on dev edit
  distance with patience-based early stopping. `torch.multiprocessing` sharing strategy set to
  `file_system` to avoid file-descriptor exhaustion from many parallel data-loading workers.

## Results

Final run (90 epochs, batch size 128, hidden 384, 4 RNN layers, dropout 0.3):

| Metric | Value |
|---|---|
| Dev edit distance (final, epoch 90) | **0.0026** (~0.26%) |
| Train edit distance (final) | 0.0265 |

## Running it

```bash
pip install -r requirements.txt
python homr_competition.py --epochs 90 --hidden 384 --rnn_layers 4
```

Uses [`npfl138`](https://pypi.org/project/npfl138/) for the training loop and the `HOMRDataset`
loader (downloaded automatically; ~500MB of handwritten score images).
