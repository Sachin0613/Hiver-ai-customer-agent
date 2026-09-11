# Phase 13 — LLM-as-Judge

`evaluation/judge.py` evaluates a generated reply using the customer message
and retrieved historical evidence. The judge does not receive system scores,
intent confidence, or escalation predictions.

It returns 1–5 scores for:

- Relevance
- Groundedness
- Helpfulness
- Tone
- Unsupported claims, where 5 means no unsupported claims

It also returns short reasoning. Responses must be valid JSON and all scores
must be integers from 1 through 5. The judge requires `OPENAI_API_KEY` at
runtime and makes no API call in tests.

Judge tuning must not be performed against final evaluation results. Human
agreement measurement is a separate Phase 14 activity.