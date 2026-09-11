# Phase 14 — Human vs LLM-Judge Agreement

The agreement harness compares the LLM judge with independent human ratings
using the same five 1–5 criteria: relevance, groundedness, helpfulness, tone,
and unsupported claims.

Provide approximately 40–50 generated replies in both files:

- `evaluation/judge_scores.csv`
- `evaluation/human_ratings.csv`

Each file must contain the same unique `id` values. Human columns are named
`human_relevance`, `human_groundedness`, `human_helpfulness`, `human_tone`, and
`human_unsupported_claims`. The script reports exact agreement, within-one-point
agreement, mean absolute difference, Pearson correlation, and Spearman
correlation for each criterion.

The repository contains an empty `human_ratings.csv` header as a schema
template. It contains no fabricated ratings. Run:

```powershell
python scripts/measure_human_agreement.py
```

The script refuses to report agreement until at least 40 complete human ratings
are present.