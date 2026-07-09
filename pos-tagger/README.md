# Czech POS Tagger

Part-of-speech tagger for Czech (PDT treebank), combining word-level and character-level
representations through stacked bidirectional RNNs.

## Approach

- **Word representations**: trainable word embeddings, with random word masking (`--word_masking`,
  replacing a word with `UNK` with some probability during training) to force reliance on
  character-level signal for rare/unseen words.
- **Character representations**: a 3-layer bidirectional GRU over each word's characters,
  producing a per-word character-level embedding (CLE) that's concatenated with the word
  embedding.
- **Tagger**: a 3-layer bidirectional LSTM over the concatenated word representations
  (forward/backward summed rather than concatenated), followed by a linear tag classifier.
  Packed sequences are used throughout to skip padding compute.
- **Training**: AdamW, cross-entropy loss (padding ignored), `torchmetrics.Accuracy` on non-pad
  tokens.

## Results

10-epoch run (batch size 64, cle_dim 64, rnn_dim 128, we_dim 128, word_masking 0.1):

| Metric | Value |
|---|---|
| Dev accuracy (final, epoch 10) | **96.18%** |

## Running it

```bash
pip install -r requirements.txt
python tagger_competition.py --epochs 10
```

Uses [`npfl138`](https://pypi.org/project/npfl138/) for the training loop and the `MorphoDataset`
/`MorphoAnalyzer` loaders (Czech PDT treebank, downloaded automatically).
