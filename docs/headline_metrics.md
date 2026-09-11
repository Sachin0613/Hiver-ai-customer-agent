# Phase 17 — What Is Misleading About My Headline Number?

A headline such as “86% intent accuracy” is incomplete and must not be used
without the surrounding evidence. This project does not currently have a
completed labeled golden set or final model results, so no headline metric is
reported yet.

When results exist, the headline number can mislead for several reasons:

## Class imbalance

Accuracy can be dominated by common intents. A system may perform well on
delivery questions while failing rare account or security cases. Macro F1,
per-intent recall, and the confusion matrix are necessary companions.

## Easy and difficult examples

A random sample can contain many short, explicit requests. Ambiguous,
multi-intent, multilingual, emotional, and context-poor messages may be harder
and more important operationally. The golden set therefore includes challenge
and writing-style metadata.

## Rare intents and label uncertainty

Rare intents have noisy estimates when only a few examples are labeled. Human
annotators may also disagree about the primary intent of a multi-intent message.
Per-intent support counts and annotation notes must accompany the metric.

## Distribution shift

The Twitter corpus is historical. Product policies, delivery practices,
language, and customer behavior may change. A random conversation split is the
main evaluation design, while the newest-15%-conversation temporal analysis is
available as a robustness comparison.

## Historical-data limitations

An observed company reply does not prove the customer's issue was resolved.
Public tweets may omit private context, include masked PII, or end with a DM
request. Retrieval quality therefore cannot be inferred from response presence
alone.

## Leakage risk

Tweet-level splitting, duplicate customer messages, or retrieving from a golden
conversation can make results look better than they are. This project assigns
conversations to one split and checks golden-pool exclusion. Repeated messages
are measured rather than silently deleted.

## False auto-handling risk

Accuracy does not show whether the system automatically answered a case that
should have gone to a human. False auto-handling rate, escalation recall, and
high-risk failure cases should be treated as production-risk metrics.

## LLM-judge limitations

LLM judge scores are not ground truth. They may be sensitive to wording,
position, and model bias. The project measures judge agreement with independent
human ratings on approximately 40–50 examples before making strong claims about
reply quality.

## Metrics that matter more than one headline

The final report should show accuracy alongside macro F1, per-intent recall,
confusion matrix, escalation recall, false auto-handling rate, groundedness,
unsupported-claim scores, human-vs-judge agreement, sample counts, and observed
failure modes. If those measures disagree with accuracy, the risk-oriented
measures should drive the conclusion.