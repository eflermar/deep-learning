# Extractive Reading Comprehension (Czech QA)

Given a Czech paragraph and a question about it, predicts the answer as a contiguous span of
text extracted from the paragraph — fine-tuning a pretrained transformer for extractive QA.

## Approach

- **Base model**: [RobeCzech](https://huggingface.co/ufal/robeczech-base) (a Czech RoBERTa),
  fine-tuned end-to-end with a linear span-prediction head on top of the final hidden states,
  predicting separate start/end token logits.
- **Tokenization**: context + question packed into one sequence (`truncation="only_first"`, so
  only the context is truncated if too long), character-to-token offset mapping used to convert
  character-level gold answer spans into token-level start/end labels.
- **Loss**: average of start-token and end-token cross-entropy.
- **Custom metric**: `SpanAccuracy`, a `torchmetrics.Metric` with three modes — exact span match,
  start-only match, and end-only match — used both for monitoring and for best-weight checkpoint
  selection (exact match on dev).
- **Optimization**: AdamW with weight-decay exclusion for biases/LayerNorm (standard transformer
  fine-tuning practice), cosine schedule with warmup.

## Results

Best run (3 epochs, batch size 12, lr 3e-5):

| Metric | Value |
|---|---|
| Dev exact-span accuracy (final, epoch 3) | **64.79%** |
| Dev start-token accuracy | ~71-72% (from a related run) |
| Dev end-token accuracy | ~71-72% (from a related run) |

## Running it

```bash
pip install -r requirements.txt
python reading_comprehension.py --epochs 4 --batch_size 16 --lr 2e-5
```

Uses [`npfl138`](https://pypi.org/project/npfl138/) for the training loop and the
`ReadingComprehensionDataset` loader, plus HuggingFace `transformers` for the pretrained
RobeCzech model and tokenizer (downloaded automatically on first run).
