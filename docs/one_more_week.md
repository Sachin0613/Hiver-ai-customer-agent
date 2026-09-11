# Phase 18 — What I Would Do With One More Week

The highest-value work is the work that turns the current implementation into
an honestly measurable system. I would prioritize it in this order:

## 1. Complete human labels and quality review

Finish the 200-example golden annotation and review the 540-example development
intent queue. Resolve multi-intent and ambiguous cases with a second annotator
or adjudication pass. This is the immediate blocker for meaningful classifier
and evaluation metrics.

## 2. Finish and verify the retrieval index

Complete the Sentence Transformer download/index build, record the model and
embedding dimensions, and test retrieval on manually inspected development
examples. Add checks for weak similarity and irrelevant evidence before any
reply is generated.

## 3. Run both baselines on labeled development data

Run the majority and TF-IDF baselines with a fixed validation split. Review
per-intent confusion and rare-class recall before tuning the main system.

## 4. Integrate one minimal end-to-end agent path

Connect intent prediction, top-k retrieval, deterministic escalation, and
structured grounded response generation. Keep the interface small and log the
evidence IDs, confidence, decision, and reply for every run.

## 5. Evaluate the locked golden set once

Generate predictions without using gold labels for tuning. Run the Phase 12
harness, measure false auto-handling separately, and record real failure cases.
Do not repeatedly tune against the golden results.

## 6. Validate the LLM judge with human ratings

Select 40–50 generated replies by a fixed rule, obtain independent human
ratings, and run the Phase 14 agreement analysis. If agreement is weak, treat
judge scores as exploratory rather than headline evidence.

## 7. Add confidence calibration and temporal robustness

Calibrate intent and retrieval thresholds on development data only. Run the
newest-conversation holdout as a separate robustness experiment to measure
historical distribution shift.

## 8. Improve reliability, then optimize cost and latency

Add retries and structured logging around API calls, cache embeddings and
responses where appropriate, measure latency and token cost, and batch embedding
work. These improvements come after correctness and leakage checks.

## Deliberately deferred

I would not spend the week adding a frontend, database, orchestration platform,
or a larger model before the labels, leakage controls, and risk metrics are
validated. Those additions would make the project harder to explain without
answering whether the support behavior works.