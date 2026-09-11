# Phase 12 — Evaluation Harness

`src/evaluation.py` evaluates saved predictions against manually labeled golden
data. It reports intent accuracy, macro F1, macro precision, macro recall,
per-intent metrics, and a confusion matrix. It also reports escalation accuracy,
precision, recall, F1, and false auto-handling rate.

Run it only after `data/golden_set.csv` has complete `gold_intent` and
`gold_decision` fields and a system prediction file exists:

```powershell
python scripts/evaluate.py --predictions path/to/predictions.csv
```

Required prediction columns are `id`, `predicted_intent`, and
`predicted_decision`. The harness refuses incomplete labels or partial
prediction coverage and never creates placeholder metrics.