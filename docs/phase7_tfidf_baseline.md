# Phase 7 — TF-IDF + Logistic Regression

This baseline uses word and bigram TF-IDF features followed by Logistic
Regression. The labeled development annotation queue is split reproducibly
into 80% training and 20% validation using seed `42`.

The baseline reports accuracy, macro F1, macro precision, macro recall,
per-intent precision/recall/F1, a confusion matrix, and validation
predictions. It never reads the golden pool.

The development queue now has first-pass reviewed labels. These labels support
development benchmarking, but they are not the locked golden evaluation set and
should receive adjudication before strong claims are made.

After annotation, install dependencies and run:

```powershell
python -m pip install -r requirements.txt
python scripts/run_tfidf_baseline.py
```

Outputs are written under `evaluation/results/`.

Current first-pass validation result: accuracy `0.6389`, macro F1 `0.5999`,
macro precision `0.6541`, and macro recall `0.5912` on 108 validation examples.
Account-access recall was `0.0000` in this split, which is an important risk
signal rather than a result to hide behind aggregate accuracy.