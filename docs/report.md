# Hiver AI Customer Support Agent

## 1. Executive Summary

This project prepares an AI customer-support agent for AmazonHelp using the
Kaggle Customer Support on Twitter dataset. The system design combines
brand-specific intent classification, historical response retrieval, grounded
reply generation, and deterministic escalation.

AmazonHelp was selected empirically because it has 82,556 conversations,
203,830 customer messages, 169,840 company responses, and a 76.0% direct
response rate in the local corpus. The current repository contains the data
pipeline and component contracts, plus first-pass development baseline results,
but it does **not** yet contain final end-to-end reply-quality metrics.

## 2. Problem Framing

Good support behavior means identifying the customer's primary issue, using
historical evidence without inventing policy or account facts, and escalating
high-risk or weakly supported cases. False auto-handling is more serious than a
simple wrong-topic prediction.

The project deliberately does not add a frontend, database, cloud deployment,
or orchestration platform. The focus is a reproducible Python pipeline that can
be explained in an interview.

## 3. Data

The verified local schema contains tweet ID, author ID, inbound flag, timestamp,
text, response tweet IDs, and parent tweet ID. The raw corpus contains
2,811,774 tweets.

Phase 3 preserves raw text, creates cleaned text, identifies customer/company
roles, and reconstructs deterministic conversation IDs from parent links. The
AmazonHelp processed data contains 82,556 conversations, 374,042 selected-brand
messages, 203,598 customer messages, 169,840 company responses, and 168,814
direct customer/company pairs.

Phase 4 assigns conversations exclusively to splits:

| Split | Conversations | Customer messages | Company responses |
| --- | ---: | ---: | ---: |
| Development | 57,789 | 143,133 | 119,144 |
| Retrieval | 12,383 | 30,189 | 25,262 |
| Golden pool | 12,384 | 30,276 | 25,434 |

The golden pool is excluded from development and retrieval. Exact repeated
customer messages are measured rather than silently deleted.

## 4. System

```text
Customer message
        |
        v
Intent classifier -----> intent + confidence
        |
        v
Historical FAISS retrieval -----> evidence pairs
        |
        v
Deterministic escalation rules
        |                         \
        | AUTO_HANDLE              \ ESCALATE + reason
        v                          \
Grounded LLM reply ----------------> human review
```

The provisional AmazonHelp taxonomy contains delivery tracking, order issues,
returns/refunds, payment/billing, Prime membership, account access,
product/listing, seller/customer service, and feedback/other. It is derived
from development data and requires manual review.

Historical retrieval uses normalized Sentence Transformer embeddings and an
exact FAISS inner-product index. The response generator requires structured JSON
with the reply, evidence indexes, uncertainty, and unsupported-claim notes.
Escalation rules cover legal threats, security incidents, sensitive information,
severe complaints, weak evidence, conflicting evidence, and low confidence.

## 5. Evaluation

The repository includes:

- Majority-class baseline
- TF-IDF plus Logistic Regression baseline
- Intent accuracy, macro F1, per-intent metrics, and confusion matrix
- Escalation precision, recall, F1, and false auto-handling rate
- Ablation harness for retrieval and intent filtering

The first-pass development benchmark reports majority accuracy `0.1481` and
macro F1 `0.0287`. TF-IDF + Logistic Regression reports validation accuracy
`0.6389` and macro F1 `0.5999` on 108 examples. On the locked 200-example
golden set, majority accuracy is `0.1100` with macro F1 `0.0220`, while TF-IDF
accuracy is `0.5400` with macro F1 `0.5020`. These are intent-only baseline
results. Account-access recall is `0.0476` for TF-IDF, so aggregate accuracy is
not sufficient evidence of safe support behavior. End-to-end reply quality,
escalation metrics, judge agreement, and ablations still require real system
predictions and ratings.

## 6. LLM Judge Validation

The LLM judge scores relevance, groundedness, helpfulness, tone, and
unsupported claims on a 1–5 scale. The human-agreement harness requires
approximately 40–50 independent human ratings and reports exact agreement,
within-one-point agreement, mean absolute difference, and correlations.

No judge or human-rating results exist yet.

## 7. Failure Analysis

The failure-analysis pipeline requires real cases containing the customer
message, expected output, actual output, failure category, explanation,
hypothesis, and proposed fix. No failure cases are reported until the golden
set has been labeled and predictions have been generated.

## 8. What Is Misleading About My Headline Number?

An accuracy headline can hide class imbalance, rare-intent failure, ambiguous
messages, temporal shift, label uncertainty, retrieval leakage, and false
auto-handling. Macro F1, per-intent recall, escalation recall, false
auto-handling rate, groundedness, unsupported-claim scores, judge agreement,
and actual failure modes provide better production context.

The full discussion is in [docs/headline_metrics.md](headline_metrics.md).

## 9. What I Would Do With One More Week

1. Complete and adjudicate development and golden labels.
2. Finish and manually verify the retrieval index.
3. Run both baselines and inspect rare-intent confusion.
4. Connect the minimal end-to-end agent path.
5. Evaluate once on the locked golden set.
6. Measure LLM-judge agreement with independent human ratings.
7. Add calibration and temporal robustness checks.
8. Optimize cost and latency only after correctness is established.

## 10. Conclusion

The project has a reproducible data foundation, leakage-aware splits, a
brand-specific provisional taxonomy, retrieval and generation contracts,
deterministic escalation rules, and evaluation tooling. It is not yet
production-ready and should not claim accuracy or reply-quality success until
human labels, real predictions, and independent ratings are completed.