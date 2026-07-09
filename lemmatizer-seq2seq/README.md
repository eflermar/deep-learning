# Czech Lemmatizer (Seq2Seq)

Character-level sequence-to-sequence lemmatizer for Czech (PDT treebank): given a word, predicts
its dictionary base form (lemma) one character at a time.

## Approach

- **Encoder**: character embeddings fed through a stack of residual bidirectional LSTM blocks
  (pre-norm, forward+backward sum, dropout, residual connection) — deeper than a single BiLSTM
  without the usual vanishing-gradient cost.
- **Decoder**: a GRU cell with Bahdanau-style additive attention over the encoded character
  sequence, decoding the lemma character-by-character. Output embeddings are optionally *tied* to
  the output projection layer's weights (scaled by `sqrt(rnn_dim)`), roughly halving the decoder's
  parameter count.
- **Post-processing**: predictions are checked against a morphological analyzer
  (`MorphoAnalyzer`) — if the predicted lemma isn't one of the analyzer's valid lemmas for that
  word, it's replaced by the analyzer's top candidate, trading a small amount of model creativity
  for morphological validity.
- **Training**: teacher forcing during training, greedy decoding at inference, dev accuracy
  computed as exact character-sequence match (`torchmetrics.MeanMetric` over per-word matches),
  best-weight checkpointing.

## Running it

```bash
pip install -r requirements.txt
python lemmatizer_competition.py --epochs 30 --blocks 3 --tie_embeddings
```

Uses [`npfl138`](https://pypi.org/project/npfl138/) for the training loop and the `MorphoDataset`
/`MorphoAnalyzer` loaders (Czech PDT treebank + morphological analyses, downloaded automatically).

*No training log survived from the original coursework runs for this project (only the raw
test-set predictions were saved), so no dev-accuracy number is quoted here — the architecture and
training setup above reflect the actual code that was submitted and graded.*
