# Phase 11 — Golden Evaluation Set

The golden set contains 200 examples sampled from the isolated Phase 4 golden
pool. It has now been first-pass manually labeled with `gold_intent` and
`gold_decision`.
The labels are held out from development and retrieval.

## Current distribution

- `AUTO_HANDLE`: 165
- `ESCALATE`: 35
- All 200 rows have an intent label.

The labels are human task judgments, not model predictions. Ambiguous cases and
high-risk cases include notes. They should not be used to tune prompts,
thresholds, or model selection after looking at final results.

## Baseline evaluation

Run the held-out intent baseline evaluation with:

```powershell
python scripts/evaluate_baselines_on_golden.py
```

This trains the baselines only on the reviewed development queue and evaluates
them once on the labeled golden set.# Phase 11 — Golden Evaluation Set

## Sampling method

`data/golden_set.csv` is sampled only from the isolated Phase 4
`data/splits/golden_pool.csv`. The development split and retrieval corpus are
not used. Sampling uses seed `42`, keeps one customer example per conversation,
and targets 200 examples.

The sampler balances broad keyword-derived coverage buckets and short, medium,
and long writing styles. These buckets are sampling aids only; they are not
intent labels.

## Labeling process

The file contains blank `gold_intent`, `gold_decision`,
`gold_response_characteristics`, and `notes` fields. A human annotator must
fill them using [docs/intent_taxonomy.md](intent_taxonomy.md) and the rules
below. No model predictions should be consulted during labeling.

For each example:

1. Read the full customer message.
2. Assign the primary intent from the taxonomy.
3. Record `AUTO_HANDLE` or `ESCALATE` based on the support task, not on a model.
4. Note secondary intents, ambiguity, missing context, or unusual language.
5. Record reply characteristics only when useful, such as asking for details,
   explaining a process, or acknowledging a complaint.

## Challenging cases

The sample intentionally retains short messages, long messages, multilingual
messages, complaints, and multi-intent candidates. `sampling_bucket` is not a
gold label and must not override human judgment.

## Isolation and limitations

The golden conversations were already excluded from development and retrieval
by Phase 4. This file is an annotation artifact, not an evaluation result.
Metrics must not be reported until the labels are manually completed. The
dataset is historical Twitter support data, so some messages lack context and
actual business resolution cannot be verified.

## Command

```powershell
python scripts/create_golden_set.py
```