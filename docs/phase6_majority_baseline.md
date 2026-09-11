# Phase 6 — Majority-Class Baseline

The majority baseline always predicts the most frequent manually reviewed
intent in the development annotation file. It is intentionally simple and
provides a lower-bound reference for later classifiers.

The development queue now contains a first-pass manual-review label for all 540
examples. The labels are not golden evaluation labels; they are used only for
development benchmarking. Ambiguous cases are recorded in `annotator_notes`.

After manually labeling the queue, run:

```powershell
python scripts/run_majority_baseline.py
```

The result is written to `evaluation/results/majority_baseline.json`.

Current first-pass result: accuracy `0.1481`, macro F1 `0.0287`, with
`feedback_or_other` as the majority class.