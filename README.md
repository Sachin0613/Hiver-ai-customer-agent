Yes — use this **short one-page README**. It is designed to fit on one screen/page much better while still covering the important take-home points.

````markdown
# Hiver AI Customer Support Agent

AI customer-support agent built for the Hiver SDE Intern take-home assignment using the Kaggle **Customer Support on Twitter (TWCS)** dataset.

## What it does

Given a customer message, the agent:

1. Classifies the support intent.
2. Retrieves similar historical AmazonHelp conversations.
3. Decides `AUTO_HANDLE` or `ESCALATE`.
4. Generates a grounded AI reply using historical AmazonHelp responses.
5. Shows confidence, escalation reason, uncertainty, unsupported claims, and evidence.

```text
Customer Message
       ↓
Intent Classification
       ↓
Intent + Confidence
       ↓
FAISS Historical Retrieval
       ↓
Escalation / Safety Rules
       ↓
AUTO_HANDLE / ESCALATE
       ↓
Grounded AI Suggested Reply
       ↓
Evidence + Uncertainty
````

## Dataset & Brand

Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

The local corpus contains approximately **2.8M tweets**.

Phase 2 selected **AmazonHelp** empirically based on conversation volume, response coverage, and suitability for historical support retrieval.

AmazonHelp statistics:

* 82,556 conversations
* 203,830 customer messages
* 169,840 company responses
* 76.0% direct response rate

## Intent Taxonomy

The system uses 9 support intents:

| Intent                       | Description                                      |
| ---------------------------- | ------------------------------------------------ |
| `delivery_tracking`          | Tracking, shipment and delivery status           |
| `order_issue`                | Missing, incorrect or problematic orders         |
| `returns_refunds`            | Returns, refunds and cancellations               |
| `payment_billing`            | Payments, charges and billing                    |
| `prime_membership`           | Prime membership and subscription issues         |
| `account_access`             | Login and account-access problems                |
| `product_or_listing`         | Product and listing questions                    |
| `seller_or_customer_service` | Seller and support-service issues                |
| `feedback_or_other`          | Feedback, greetings and other/ambiguous messages |

## Architecture

**Intent:** TF-IDF + Logistic Regression baseline

```text
TF-IDF
- ngram_range=(1,2)
- sublinear_tf=True

Logistic Regression
- max_iter=1000
- random_state=42
```

**Retrieval:** Sentence Transformers + FAISS

```text
Model:
sentence-transformers/all-MiniLM-L6-v2

Index:
FAISS IndexFlatIP

Retrieval corpus:
25,105 historical pairs
```

**Escalation:** deterministic safety rules covering:

* low intent confidence;
* weak or conflicting evidence;
* legal threats;
* security incidents;
* sensitive information;
* severe complaints;
* high-risk financial situations.

Default thresholds:

```text
Intent confidence: 0.65
Minimum evidence score: 0.45
Maximum evidence-score spread: 0.12
```

**Response generation:** OpenAI API using retrieved AmazonHelp examples as grounding evidence. The model is instructed not to invent policies, refunds, tracking information, dates, actions, or private information.

## Data Split

The project separates development, retrieval, and golden evaluation data:

| Split       | Conversations | Customer Messages | Company Responses |
| ----------- | ------------: | ----------------: | ----------------: |
| Development |        57,789 |           143,133 |           119,144 |
| Retrieval   |        12,383 |            30,189 |            25,262 |
| Golden Pool |        12,384 |            30,276 |            25,434 |

The golden pool is excluded from development and retrieval to reduce evaluation leakage.

## Evaluation

A locked **200-example golden set** has been prepared:

```text
AUTO_HANDLE: 165
ESCALATE: 35
```

Current **intent-classification baseline** results:

| Model                        | Accuracy | Macro F1 |
| ---------------------------- | -------: | -------: |
| Majority                     |   0.1100 |   0.0220 |
| TF-IDF + Logistic Regression |   0.5400 |   0.5020 |

TF-IDF baseline:

```text
Macro Precision: 0.6755
Macro Recall:    0.5400
```

Important rare-intent observations:

```text
account_access recall:     0.0476
product_or_listing recall: 0.2083
```

These are **intent-classification metrics only**, not end-to-end agent performance.

Final evaluation will additionally consider retrieval quality, escalation safety, groundedness, unsupported claims, LLM-as-judge results, and human agreement.

## Browser Demo

Run:

```powershell
.\.venv\Scripts\python.exe app.py
```

The local browser application opens at:

```text
http://127.0.0.1:5000
```

Enter a message such as:

```text
Where is my order?
```

The UI displays:

* predicted intent;
* confidence;
* `AUTO_HANDLE` / `ESCALATE`;
* escalation rule and reason;
* AI suggested reply;
* uncertainty;
* unsupported claims;
* historical AmazonHelp evidence.

## Setup

Create the environment:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the Kaggle dataset at:

```text
data/raw/twcs.csv
```

Create `.env` in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Then run:

```powershell
.\.venv\Scripts\python.exe app.py
```

The API key is only required for AI reply generation. Never commit `.env`.

## Repository Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── indexes/
├── src/
│   ├── data_loader.py
│   ├── brand_analysis.py
│   ├── intent_discovery.py
│   ├── tfidf_baseline.py
│   ├── retrieval.py
│   ├── escalation.py
│   └── response_generator.py
├── scripts/
├── evaluation/
└── docs/
    ├── report.md
    ├── brand_selection.md
    └── decision_log.md
```

## Key Artifacts

```text
data/processed/brand_statistics.csv
data/processed/intent_annotation_queue.csv
data/indexes/retrieval.faiss
data/indexes/retrieval_metadata.csv
data/indexes/index_manifest.json
evaluation/results/baseline_golden_metrics.json
evaluation/results/baseline_golden_predictions.csv
docs/report.md
docs/decision_log.md
```

## Current Status

**Phase 20 implementation complete.**

Implemented:

* dataset analysis;
* empirical brand selection;
* intent taxonomy;
* annotation workflow;
* train/retrieval/golden separation;
* TF-IDF baseline;
* FAISS historical retrieval;
* deterministic escalation;
* grounded OpenAI response generation;
* browser-based interactive agent;
* baseline evaluation;
* project report;
* technical decision log.

Remaining empirical work:

* complete/adjudicate human golden labels;
* run full end-to-end evaluation;
* LLM-as-judge evaluation;
* human agreement measurement;
* real-system top-5 failure analysis.

## Important Limitation

A headline accuracy number can be misleading because of class imbalance, rare-intent failures, ambiguous messages, retrieval leakage, unsafe automatic handling, and the difference between intent classification and complete support-agent quality.

The core design principle is:

> **When the system is uncertain, it should surface uncertainty and escalate rather than confidently invent an answer.**

## License

The Kaggle dataset is licensed under **CC BY-NC-SA 4.0**.

Dataset:
[https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

```

This is the version I recommend submitting as your **main README**: compact enough to be reviewer-friendly, but it still documents the architecture, demo, metrics, limitations, and evaluation status.

You could refine it further by

- :contentReference[oaicite:0]{index=0}
- :contentReference[oaicite:1]{index=1}
```
