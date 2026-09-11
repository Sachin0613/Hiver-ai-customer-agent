"""Development-only intent discovery helpers for AmazonHelp."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


STOPWORDS = {
    "the", "and", "you", "for", "that", "this", "with", "are", "was", "have",
    "from", "your", "our", "can", "not", "but", "all", "any", "get", "got",
    "just", "will", "please", "thanks", "thank", "hi", "hey", "hello", "we",
    "us", "me", "my", "is", "it", "in", "on", "to", "of", "at", "be", "or",
    "if", "so", "do", "did", "been", "they", "them", "their", "what", "when",
    "how", "why", "who", "has", "had", "out", "now", "one", "about", "still",
    "would", "could", "should", "also", "amazon", "amazonhelp", "https", "http",
}
TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")

INTENT_DEFINITIONS = [
    ("delivery_tracking", "Tracking, shipping progress, delivery dates, delays, or a package marked delivered.", ["delivery", "tracking", "package", "shipment", "shipping", "delivered", "late"]),
    ("order_issue", "An order, item, cancellation, or order-status problem not primarily about delivery tracking.", ["order", "ordered", "cancel", "wrong", "missing", "item"]),
    ("returns_refunds", "Returning an item, refund status, reimbursement, or refund amount/timing.", ["refund", "return", "reimburse", "money back", "replacement"]),
    ("payment_billing", "Payment method, charge, invoice, credit card, price, or billing problem.", ["payment", "charged", "charge", "credit card", "debit", "billing", "invoice", "price"]),
    ("prime_membership", "Prime membership, subscription benefits, renewal, trial, or Prime delivery entitlement.", ["prime", "membership", "subscription", "renewal", "trial"]),
    ("account_access", "Login, email, password, account security, or account information problem.", ["account", "login", "password", "email", "sign in", "locked"]),
    ("product_or_listing", "Product quality, compatibility, listing information, warranty, or product-specific question.", ["product", "item", "device", "broken", "defective", "warranty", "seller"]),
    ("seller_or_customer_service", "Seller interaction, marketplace support, customer-service experience, or service complaint.", ["seller", "customer service", "customer care", "support", "complaint", "terrible"]),
    ("feedback_or_other", "Feedback, praise, general questions, multilingual/unclear messages, or messages outside the support intents above.", ["feedback", "love", "great", "question"]),
]


def tokenize(text: object) -> list[str]:
    value = "" if pd.isna(text) else str(text).lower()
    return [token for token in TOKEN_RE.findall(value) if token not in STOPWORDS]


def topic_counts(texts: Iterable[object], top_k: int = 50) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return pd.DataFrame(counts.most_common(top_k), columns=["token", "count"])


def broad_sampling_bucket(text: object) -> str:
    """Assign a sampling bucket only; this is not a gold intent label."""
    value = "" if pd.isna(text) else str(text).lower()
    patterns = [
        ("delivery_tracking", r"delivery|tracking|shipment|shipping|package|delivered|late"),
        ("order_issue", r"order|ordered|cancel|missing item"),
        ("returns_refunds", r"refund|return|reimburse|money back|replacement"),
        ("payment_billing", r"payment|charged|charge|credit card|debit|billing|invoice|price"),
        ("prime_membership", r"prime|membership|subscription|renewal|trial"),
        ("account_access", r"account|login|password|sign in|email|locked"),
        ("product_or_listing", r"product|device|broken|defective|warranty|seller"),
        ("seller_or_customer_service", r"seller|customer service|customer care|complaint|support"),
    ]
    for bucket, pattern in patterns:
        if re.search(pattern, value):
            return bucket
    return "feedback_or_other"


def build_annotation_queue(
    development: pd.DataFrame,
    sample_per_bucket: int,
    sample_seed: int,
) -> pd.DataFrame:
    customer = development[development["speaker"] == "CUSTOMER"].copy()
    customer["sampling_bucket"] = customer["clean_text"].map(broad_sampling_bucket)
    customer["text_length"] = customer["clean_text"].fillna("").str.len()
    customer["writing_style_bucket"] = pd.cut(
        customer["text_length"],
        bins=[-1, 40, 160, float("inf")],
        labels=["short", "medium", "long"],
    ).astype(str)
    rng = np.random.default_rng(sample_seed)
    parts: list[pd.DataFrame] = []
    for bucket, group in customer.groupby("sampling_bucket", sort=True, observed=True):
        for style in ("short", "medium", "long"):
            subset = group[group["writing_style_bucket"] == style]
            if subset.empty:
                continue
            take = min(sample_per_bucket, len(subset))
            chosen = rng.choice(subset.index.to_numpy(), size=take, replace=False)
            parts.append(subset.loc[chosen])
    if not parts:
        return pd.DataFrame()
    queue = pd.concat(parts, ignore_index=True)
    queue = queue.sort_values(["sampling_bucket", "writing_style_bucket", "conversation_id", "tweet_id"]).reset_index(drop=True)
    queue.insert(0, "annotation_id", [f"dev_{index:05d}" for index in range(len(queue))])
    queue["intent_label"] = ""
    queue["annotation_status"] = "unlabeled"
    queue["annotator_notes"] = ""
    return queue[[
        "annotation_id", "conversation_id", "tweet_id", "clean_text", "raw_text",
        "timestamp", "message_index", "sampling_bucket", "writing_style_bucket",
        "intent_label", "annotation_status", "annotator_notes",
    ]]