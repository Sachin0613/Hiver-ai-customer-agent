# Hiver support agent (take-home)

Python AI customer-support agent over the Kaggle [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) corpus.

**Current phase: 20 — implementation and report complete; empirical evaluation remains pending human labels.**

## Status on this machine (2026-09-11)

| Check | Result |
| --- | --- |
| Project files | Phases 1–20 implemented with honest evaluation gates |
| `data/raw/twcs.csv` | Present; full local corpus |
| Python 3.11+ | Python 3.12 available |

Phase 2 selected `AmazonHelp` from the local data. See [docs/report.md](docs/report.md).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the Kaggle files as described in `data/raw/README.md`, then run:

```powershell
python scripts/analyze_dataset.py
python scripts/analyze_brands.py
```

The project includes:

- `data/processed/brand_statistics.csv` — full-corpus brand statistics
- `data/processed/candidate_conversations.csv` — small candidate sample
- `docs/brand_selection.md` — criteria, scores, themes, and recommendation
- `evaluation/results/phase2_brand_selection.json` — reproducible results
- `docs/report.md` — final project report
- `docs/decision_log.md` — 15 interview-facing technical decisions

## Layout (Phases 1–2)

```
.
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── configs/config.yaml
├── data/raw/          # you add twcs.csv here
├── data/processed/    # generated Phase 2 tables
├── src/data_loader.py
├── src/brand_analysis.py
├── scripts/analyze_dataset.py
├── scripts/analyze_brands.py
├── evaluation/results/
└── docs/
```

## License note

The Kaggle dataset is **CC BY-NC-SA 4.0**. Keep the CSV out of git (already gitignored).
