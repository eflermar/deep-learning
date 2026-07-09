# Deep Learning Coursework

Seven deep learning projects built with PyTorch, originally completed as coursework for the
NPFL138 Deep Learning course at Charles University. Each project is a self-contained model
trained end-to-end on a real dataset — image classification, object detection, 3D recognition,
sequence tagging, sequence-to-sequence generation, handwritten music recognition, and transformer
fine-tuning for extractive QA.

All projects share the [`npfl138`](https://pypi.org/project/npfl138/) course package, a thin
PyTorch wrapper (`TrainableModule`, dataset loaders, logging utilities) built around the standard
`model.fit()`/`model.predict()` pattern. Larger models (HOMR, SVHN, CIFAR) were trained on
[MetaCentrum](https://metacentrum.cz/en/), the Czech national HPC/grid infrastructure.

## Projects

| Project | Task | Key techniques | Result |
|---|---|---|---|
| [`cifar-image-classification/`](cifar-image-classification) | Classify CIFAR-10 images (10 classes) | Custom DenseNet, heavy augmentation, cosine LR, label smoothing | 97.06% dev accuracy |
| [`svhn-digit-detection/`](svhn-digit-detection) | Detect and classify house-number digits in Street View images | Anchor-based single-stage detector, EfficientNetV2 backbone, focal loss, NMS | ~0.35 final dev loss |
| [`3d-point-cloud-recognition/`](3d-point-cloud-recognition) | Classify 3D CAD models (ModelNet, voxelized) | 3D DenseNet, warmup + cosine schedule | — |
| [`lemmatizer-seq2seq/`](lemmatizer-seq2seq) | Predict the dictionary lemma of Czech words | Residual BiLSTM encoder, attention-based GRU decoder, tied embeddings | — |
| [`pos-tagger/`](pos-tagger) | Tag Czech words with part-of-speech tags | Word + character-level (BiGRU) embeddings, BiLSTM tagger, word masking | 96.18% dev accuracy |
| [`handwritten-music-recognition/`](handwritten-music-recognition) | Transcribe handwritten sheet music into symbol sequences | CNN + BiGRU, CTC loss, greedy decoding | 0.26% dev edit distance |
| [`reading-comprehension-qa/`](reading-comprehension-qa) | Extract answer spans from Czech text given a question | Fine-tuned RobeCzech transformer, span-prediction head | 64.79% dev exact-match |

Projects without a quoted result had their training logs lost along the way (only the raw
test-set predictions were saved) — the README in each still describes the actual architecture and
training setup that was submitted and graded.

## Running the projects

Each project folder is self-contained with its own `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r <project-folder>/requirements.txt
python <project-folder>/<script>.py
```

Datasets are downloaded automatically by `npfl138`'s dataset loaders on first run.
