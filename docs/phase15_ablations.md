# Phase 15 — Ablations and Experiments

The ablation harness compares four real-output variants:

1. `llm_without_retrieval`
2. `llm_with_retrieval`
3. `retrieval_without_intent_filter`
4. `retrieval_with_intent_filter`

Each row in `evaluation/ablation_scores.csv` must contain a unique `id`, one
variant, and the five Phase 13 judge scores: relevance, groundedness,
helpfulness, tone, and unsupported claims. The harness validates scores from 1
to 5 and requires all four variants before summarizing means and sample counts.

Run:

```powershell
python scripts/run_ablations.py
```

No ablation scores are currently present. The script refuses to create results
without real generated replies and judge scores. The golden set must remain
locked while variants and prompts are developed; final comparison should use
the labeled evaluation protocol without tuning on those results.