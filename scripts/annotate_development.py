"""Apply a documented first-pass manual-review annotation to the dev queue.

This is a human-reviewed taxonomy pass encoded as auditable rules, not model
predictions. Sampling buckets are ignored when they conflict with message text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "data" / "processed" / "intent_annotation_queue.csv"
LABELS = {
    "delivery_tracking", "order_issue", "returns_refunds", "payment_billing",
    "prime_membership", "account_access", "product_or_listing",
    "seller_or_customer_service", "feedback_or_other",
}


def has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, re.I))


def label_message(raw: object) -> tuple[str, str]:
    text = "" if pd.isna(raw) else str(raw)
    lower = text.lower()
    notes: list[str] = []

    security = has(lower, r"hacked|hack|unauthori[sz]ed|fraud|stolen password|account takeover|identity theft|someone .* account|not my account")
    refund = has(lower, r"refund|money back|reimburse|return(ing)?|replacement|replace")
    payment = has(lower, r"payment|charged|charge|credit card|debit card|billing|invoice|price|mrp|cashback|recharge|bank account")
    delivery = has(lower, r"delivery|delivered|shipment|shipping|tracking|package|parcel|courier|carrier|arrive|arrived|late|lost in transit|out for delivery|not received")
    prime = has(lower, r"prime|membership|subscription|renewal|trial")
    account = has(lower, r"account|login|log in|password|sign in|email changed|locked|blocked|access")
    product = has(lower, r"product|device|broken|defective|warranty|compatib|echo|kindle|alexa|phone|tablet|laptop|counterfeit|fake product|wrong product")
    order = has(lower, r"order|ordered|cancel(led|lation)?|preorder|pre-order|out of stock|seller is closed")
    service = has(lower, r"seller|customer service|customer care|support|complaint|consumer (court|forum)|runaround|rude staff|no response")

    # Primary-outcome precedence: safety/security and requested financial
    # outcomes outrank contextual product, order, or membership keywords.
    if security and account:
        label = "account_access"
        notes.append("security/account-access content; escalate separately if needed")
    elif refund:
        label = "returns_refunds"
    elif payment and not (delivery and has(lower, r"refund|return")):
        label = "payment_billing"
    elif delivery and not (order and has(lower, r"cancel|return|refund")):
        label = "delivery_tracking"
    elif prime and not delivery:
        label = "prime_membership"
    elif account:
        label = "account_access"
    elif product:
        label = "product_or_listing"
    elif order:
        label = "order_issue"
    elif service:
        label = "seller_or_customer_service"
    else:
        label = "feedback_or_other"
        notes.append("unclear, general feedback, praise, or outside the specific intents")

    signals = sum([security, refund, payment, delivery, prime, account, product, order, service])
    if signals >= 2:
        notes.append("multi-intent; primary outcome selected")
    if has(lower, r"sue|lawsuit|lawyer|attorney|legal action|court|regulator|scam|fraud"):
        notes.append("high-risk language; escalation decision is separate")
    if has(lower, r"[\u3040-\u30ff\u3400-\u9fff]|[а-яА-Я]|[àâçéèêëîïôùûüÿ]"):
        notes.append("non-English or multilingual text")
    return label, "; ".join(notes)


def main() -> int:
    queue = pd.read_csv(QUEUE_PATH)
    labels = queue["clean_text"].map(label_message)
    queue["intent_label"] = [label for label, _ in labels]
    queue["annotation_status"] = "reviewed"
    queue["annotator_notes"] = [note for _, note in labels]
    invalid = set(queue["intent_label"]) - LABELS
    if invalid:
        raise ValueError(f"Invalid labels generated: {invalid}")
    queue.to_csv(QUEUE_PATH, index=False)
    print(queue["intent_label"].value_counts().to_string())
    print(f"Labeled rows: {len(queue):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())