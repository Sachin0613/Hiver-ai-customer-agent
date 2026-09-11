# Phase 1 inspection status

**Date:** 2026-09-11

**Local inspection of tweet rows: blocked.**

The workspace `C:\Users\HP\Documents\havir` had no project files and no Kaggle dump. Searches of this folder, Downloads, Desktop, and common Kaggle cache paths found no `twcs.csv` / `sample.csv`.

Python 3.11+ and Git were not available on PATH (`python` resolved to the Windows Store stub; `git` was missing). The inspection script therefore could not be executed against real rows.

## Official dataset (documentation only — not local measurements)

These facts come from the public Kaggle page, not from a local `read_csv`:

| Item | Value |
| --- | --- |
| Dataset | [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) |
| Owner / slug | `thoughtvector/customer-support-on-twitter` |
| Version (page) | Version 10, **516.53 MB** |
| Files listed | folder `twcs`, file `sample.csv` |
| Row unit | one tweet per CSV row |
| Documented columns | `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id` |

Kaggle states every included conversation has at least one consumer request and one company response. Company vs customer is recovered from `inbound` plus `author_id` (company handles are readable; consumer IDs are anonymized). PII in `text` may be masked as `__email__` / similar.

**Do not treat the ~2.8M row figure from third-party notebooks as our measurement until `scripts/analyze_dataset.py` prints it from disk.**

## What to do next

1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ and tick **Add python.exe to PATH**.
2. Download the full dataset (not only `sample.csv`) and extract so `twcs.csv` is under `data/raw/`. See `data/raw/README.md`.
3. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/analyze_dataset.py
```

4. Open `docs/phase1_report.md` and `evaluation/results/phase1_stats.json`. Those files will contain real shapes, samples, missingness, duplicates, and brand rankings.

Until step 3 succeeds, candidate brands and a single recommended brand stay **unassigned**.
