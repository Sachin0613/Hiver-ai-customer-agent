"""Chunked brand statistics for the Twitter customer-support corpus.

Memory strategy: never keep tweet text for the full file. First pass reads only
id / author / inbound / parent columns in chunks, then concatenates a slim table.
Conversation ids use union-find on those ids. Text is read in a second pass for
a small sample of candidate-brand tweets only.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from src.data_loader import column_map, parse_inbound_series

SLIM_DOC_COLUMNS = (
    "tweet_id",
    "author_id",
    "inbound",
    "in_response_to_tweet_id",
    "response_tweet_id",
)
TEXT_DOC_COLUMNS = ("tweet_id", "author_id", "inbound", "created_at", "text")

TOKEN_RE = re.compile(r"[a-z]{3,}")
DM_RE = re.compile(
    r"\b(dm|dms|direct message|private message|send us a (?:pm|message)|click message)\b",
    re.I,
)
STOPWORDS = {
    "the", "and", "you", "for", "that", "this", "with", "are", "was", "have",
    "from", "your", "our", "can", "not", "but", "all", "any", "get", "got",
    "just", "will", "please", "thanks", "thank", "hi", "hey", "hello", "we",
    "us", "me", "my", "is", "it", "in", "on", "to", "of", "at", "be", "or",
    "if", "so", "do", "did", "been", "they", "them", "their", "what", "when",
    "how", "why", "who", "has", "had", "out", "now", "one", "about", "still",
    "would", "could", "should", "also", "via", "amp", "http", "https", "www",
    "com", "twitter",
}


@dataclass(frozen=True)
class BrandRow:
    brand: str
    total_tweets: int
    customer_messages: int
    company_responses: int
    conversation_count: int
    resolved_conversations: int
    mixed_brand_conversations: int
    avg_conversation_length: float
    median_conversation_length: float
    max_conversation_length: int
    response_rate: float
    resolved_conversation_rate: float
    multi_turn_resolved_rate: float
    dm_template_rate: float
    useful_reply_rate: float
    ngram_diversity: float
    volume_score: float
    resolution_score: float
    quality_score: float
    diversity_score: float
    response_usefulness_score: float
    overall_score: float


SCORE_WEIGHTS = {
    "volume": 0.20,
    "resolution": 0.25,
    "quality": 0.20,
    "diversity": 0.15,
    "response_usefulness": 0.20,
}


def required_columns(mapping: dict[str, str], names: Iterable[str]) -> list[str]:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(
            "Unable to reliably calculate brand statistics; missing columns: "
            + ", ".join(missing)
        )
    return [mapping[name] for name in names]


def iter_csv_chunks(
    csv_path: Path,
    usecols: list[str],
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
    )


def slim_chunk(chunk: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    inbound = parse_inbound_series(chunk[mapping["inbound"]])
    # Official IDs fit in int64; numeric storage is substantially smaller than
    # keeping millions of repeated Python string objects in the slim table.
    tweet_id = pd.to_numeric(chunk[mapping["tweet_id"]], errors="raise").astype("int64")
    parent = pd.to_numeric(
        chunk[mapping["in_response_to_tweet_id"]], errors="coerce"
    ).astype("Int64")
    # Brand handles are values such as ``AmazonHelp`` rather than numeric IDs.
    author = chunk[mapping["author_id"]].astype("string").str.strip()
    brand = author.where(inbound == False, pd.NA)  # noqa: E712
    has_listed_response = chunk[mapping["response_tweet_id"]].notna()
    return pd.DataFrame(
        {
            "tweet_id": tweet_id,
            "parent_id": parent,
            "inbound": inbound,
            "brand": brand,
            "has_listed_response": has_listed_response.astype(bool),
        }
    )


def load_slim_table(csv_path: Path, mapping: dict[str, str], chunksize: int) -> pd.DataFrame:
    usecols = required_columns(mapping, SLIM_DOC_COLUMNS)
    parts = [slim_chunk(chunk, mapping) for chunk in iter_csv_chunks(csv_path, usecols, chunksize)]
    if not parts:
        raise ValueError("CSV contained no rows.")
    slim = pd.concat(parts, ignore_index=True)
    slim["brand"] = slim["brand"].astype("category")
    return slim


def conversation_labels(tweet_id: pd.Series, parent_id: pd.Series) -> np.ndarray:
    """Union-find: a tweet and its parent belong to the same conversation."""
    n = len(tweet_id)
    parent = np.arange(n, dtype=np.int32)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    pos = pd.Series(np.arange(n, dtype=np.int32), index=tweet_id.to_numpy())
    parent_pos = parent_id.map(pos)
    valid = parent_pos.notna()
    child_idx = np.flatnonzero(valid.to_numpy())
    parent_idx = parent_pos.to_numpy()
    for i in child_idx:
        union(int(i), int(parent_idx[i]))

    labels = np.empty(n, dtype=np.int32)
    for i in range(n):
        labels[i] = find(i)
    return labels


def _minmax(values: pd.Series) -> pd.Series:
    vmin, vmax = float(values.min()), float(values.max())
    if math.isclose(vmin, vmax):
        return pd.Series(1.0, index=values.index)
    return (values - vmin) / (vmax - vmin)


def tokenize(text: str) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]


def bigrams(tokens: list[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


def ngram_diversity(texts: list[str]) -> float:
    """Share of sampled messages covered by relatively common content bigrams.

    Higher means recurring issue language rather than one-off unique wording.
    """
    if not texts:
        return 0.0
    counts: Counter[str] = Counter()
    per_doc: list[set[str]] = []
    for text in texts:
        grams = set(bigrams(tokenize(text)))
        per_doc.append(grams)
        counts.update(grams)
    if not counts:
        return 0.0
    min_df = max(3, int(0.005 * len(texts)))
    recurring = {gram for gram, n in counts.items() if n >= min_df}
    if not recurring:
        return 0.0
    covered = sum(1 for grams in per_doc if grams & recurring)
    return covered / len(texts)


def is_dm_template(text: str) -> bool:
    return bool(DM_RE.search(text))


def is_useful_reply(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 40:
        return False
    if is_dm_template(stripped) and len(stripped) < 120:
        return False
    return True


def add_conversation_ids(slim: pd.DataFrame) -> pd.DataFrame:
    work = slim.copy()
    work["conv_id"] = conversation_labels(work["tweet_id"], work["parent_id"])
    work["is_customer"] = work["inbound"] == True  # noqa: E712
    work["is_company"] = work["inbound"] == False  # noqa: E712
    return work


def aggregate_brands(slim: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per-brand stats. `slim` must already include `conv_id`."""
    work = slim if "conv_id" in slim.columns else add_conversation_ids(slim)
    company = work.loc[work["is_company"] & work["brand"].notna(), ["conv_id", "brand", "tweet_id"]]
    if company.empty:
        return pd.DataFrame()

    conv_brand = company[["conv_id", "brand"]].drop_duplicates()
    n_brands = conv_brand.groupby("conv_id").size().rename("n_brands")
    conv_len = work.groupby("conv_id").size().rename("length")
    conv_customers = work.groupby("conv_id")["is_customer"].sum().rename("n_cust")
    brand_replies_in_conv = company.groupby(["conv_id", "brand"]).size().rename("n_brand_replies")

    conv_meta = (
        conv_brand.merge(n_brands, on="conv_id")
        .merge(conv_len, on="conv_id")
        .merge(conv_customers, on="conv_id")
        .merge(brand_replies_in_conv.reset_index(), on=["conv_id", "brand"])
    )
    conv_meta["resolved"] = (conv_meta["n_cust"] > 0) & (conv_meta["n_brand_replies"] > 0)
    conv_meta["mixed"] = conv_meta["n_brands"] > 1
    conv_meta["multi_turn_resolved"] = conv_meta["resolved"] & (conv_meta["length"] >= 3)

    direct_reply = work.loc[
        work["is_company"] & work["parent_id"].notna(),
        ["parent_id", "brand"],
    ].drop_duplicates()
    direct_reply = direct_reply.rename(columns={"parent_id": "tweet_id", "brand": "reply_brand"})

    tagged = work.merge(conv_brand.rename(columns={"brand": "conv_brand"}), on="conv_id", how="inner")
    customers = tagged.loc[tagged["is_customer"], ["tweet_id", "conv_brand"]].merge(
        direct_reply, on="tweet_id", how="left"
    )
    customers["replied_by_brand"] = customers["reply_brand"] == customers["conv_brand"]

    cust_agg = customers.groupby("conv_brand").agg(
        customer_messages=("tweet_id", "size"),
        customer_with_reply=("replied_by_brand", "sum"),
    )
    tweet_agg = tagged.groupby("conv_brand").size().rename("total_tweets")
    company_responses = company.groupby("brand").size().rename("company_responses")

    conv_agg = conv_meta.groupby("brand").agg(
        conversation_count=("conv_id", "nunique"),
        resolved_conversations=("resolved", "sum"),
        mixed_brand_conversations=("mixed", "sum"),
        avg_conversation_length=("length", "mean"),
        median_conversation_length=("length", "median"),
        max_conversation_length=("length", "max"),
        resolved_conversation_rate=("resolved", "mean"),
        multi_turn_resolved_rate=("multi_turn_resolved", "mean"),
    )

    stats = (
        conv_agg.join(tweet_agg, how="left")
        .join(company_responses, how="left")
        .join(cust_agg, how="left")
        .reset_index()
        .rename(columns={"index": "brand"})
    )
    if "brand" not in stats.columns:
        stats = stats.rename(columns={stats.columns[0]: "brand"})
    stats["customer_messages"] = stats["customer_messages"].fillna(0).astype(int)
    stats["customer_with_reply"] = stats["customer_with_reply"].fillna(0).astype(int)
    stats["company_responses"] = stats["company_responses"].fillna(0).astype(int)
    stats["total_tweets"] = stats["total_tweets"].fillna(0).astype(int)
    stats["response_rate"] = np.where(
        stats["customer_messages"] > 0,
        stats["customer_with_reply"] / stats["customer_messages"],
        0.0,
    )
    stats = stats.drop(columns=["customer_with_reply"])
    return stats.sort_values("resolved_conversations", ascending=False).reset_index(drop=True)


def attach_text_metrics(
    stats: pd.DataFrame,
    reply_texts: dict[str, list[str]],
    customer_texts: dict[str, list[str]],
) -> pd.DataFrame:
    stats = stats.copy()
    dm_rates = []
    useful_rates = []
    diversities = []
    for brand in stats["brand"]:
        replies = reply_texts.get(brand, [])
        customers = customer_texts.get(brand, [])
        if replies:
            dm_rates.append(sum(is_dm_template(t) for t in replies) / len(replies))
            useful_rates.append(sum(is_useful_reply(t) for t in replies) / len(replies))
        else:
            dm_rates.append(float("nan"))
            useful_rates.append(float("nan"))
        diversities.append(ngram_diversity(customers) if customers else float("nan"))
    stats["dm_template_rate"] = dm_rates
    stats["useful_reply_rate"] = useful_rates
    stats["ngram_diversity"] = diversities
    return stats


def score_brands(stats: pd.DataFrame) -> pd.DataFrame:
    """Min-max scores among the rows passed in (typically the ranking table)."""
    if stats.empty:
        return stats
    out = stats.copy()
    volume_raw = np.log1p(out["resolved_conversations"].astype(float))
    out["volume_score"] = _minmax(pd.Series(volume_raw, index=out.index))
    out["resolution_score"] = out["response_rate"].astype(float).clip(0, 1)

    median = out["median_conversation_length"].astype(float)
    # 3–6 messages is a usable support thread; 1-message or huge threads score lower.
    sweet = 1.0 - ((median - 4.0).abs() / 4.0).clip(0, 1)
    multi = out["multi_turn_resolved_rate"].astype(float).clip(0, 1)
    out["quality_score"] = (0.5 * sweet + 0.5 * multi).clip(0, 1)

    div = out["ngram_diversity"].astype(float)
    out["diversity_score"] = _minmax(div.fillna(div.median() if div.notna().any() else 0.0))

    useful = out["useful_reply_rate"].astype(float)
    filled_useful = useful.fillna(useful.median() if useful.notna().any() else 0.0)
    out["response_usefulness_score"] = filled_useful.clip(0, 1)

    out["overall_score"] = (
        SCORE_WEIGHTS["volume"] * out["volume_score"]
        + SCORE_WEIGHTS["resolution"] * out["resolution_score"]
        + SCORE_WEIGHTS["quality"] * out["quality_score"]
        + SCORE_WEIGHTS["diversity"] * out["diversity_score"]
        + SCORE_WEIGHTS["response_usefulness"] * out["response_usefulness_score"]
    )
    return out.sort_values("overall_score", ascending=False).reset_index(drop=True)


def pick_candidates(stats: pd.DataFrame, n: int = 5, min_resolved: int = 5000) -> list[str]:
    eligible = stats[stats["resolved_conversations"] >= min_resolved]
    if eligible.empty:
        eligible = stats
    ranked = eligible.assign(
        candidate_priority=(
            eligible["resolved_conversations"].astype(float)
            * eligible["response_rate"].astype(float)
        )
    )
    return ranked.sort_values(
        ["candidate_priority", "resolved_conversations"],
        ascending=False,
    ).head(n)["brand"].tolist()


def sample_conversation_ids(
    slim: pd.DataFrame,
    brand: str,
    n: int,
    rng: np.random.Generator,
) -> list[int]:
    company = slim[(slim["brand"] == brand) & (slim["inbound"] == False)]  # noqa: E712
    conv_ids = company["conv_id"].unique()
    if len(conv_ids) == 0:
        return []
    lengths = slim[slim["conv_id"].isin(conv_ids)].groupby("conv_id").size()
    buckets = {
        "short": lengths[lengths <= 2].index.to_numpy(),
        "medium": lengths[(lengths >= 3) & (lengths <= 6)].index.to_numpy(),
        "long": lengths[lengths >= 7].index.to_numpy(),
    }
    chosen: list[int] = []
    per_bucket = max(1, n // 3)
    for key in ("short", "medium", "long"):
        pool = buckets[key]
        if len(pool) == 0:
            continue
        take = min(per_bucket, len(pool))
        picked = rng.choice(pool, size=take, replace=False)
        chosen.extend(int(x) for x in picked)
    remaining = [int(c) for c in conv_ids if int(c) not in chosen]
    if len(chosen) < n and remaining:
        extra = min(n - len(chosen), len(remaining))
        picked = rng.choice(np.array(remaining), size=extra, replace=False)
        chosen.extend(int(x) for x in np.atleast_1d(picked))
    return chosen[:n]


def collect_text_samples(
    csv_path: Path,
    mapping: dict[str, str],
    tweet_ids: set[str],
    chunksize: int,
) -> pd.DataFrame:
    if not tweet_ids:
        return pd.DataFrame()
    usecols = required_columns(mapping, TEXT_DOC_COLUMNS)
    parts: list[pd.DataFrame] = []
    for chunk in iter_csv_chunks(csv_path, usecols, chunksize):
        ids = chunk[mapping["tweet_id"]].astype(str)
        hit = chunk.loc[ids.isin(tweet_ids)].copy()
        if hit.empty:
            continue
        rename = {
            mapping["tweet_id"]: "tweet_id",
            mapping["author_id"]: "author_id",
            mapping["inbound"]: "inbound",
            mapping["created_at"]: "created_at",
            mapping["text"]: "text",
        }
        parts.append(hit.rename(columns=rename)[list(rename.values())])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def theme_table(texts: list[str], top_k: int = 15) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(bigrams(tokenize(text)))
    rows = [{"theme_bigram": gram, "count": n} for gram, n in counts.most_common(top_k)]
    return pd.DataFrame(rows)


def reservoir_tweet_ids(
    slim: pd.DataFrame,
    brand: str,
    kind: str,
    n: int,
    rng: np.random.Generator,
) -> list[str]:
    if kind == "customer":
        mask = (slim["conv_id"].isin(slim.loc[slim["brand"] == brand, "conv_id"])) & (
            slim["inbound"] == True  # noqa: E712
        )
    else:
        mask = (slim["brand"] == brand) & (slim["inbound"] == False)  # noqa: E712
    ids = slim.loc[mask, "tweet_id"].astype(str).to_numpy()
    if len(ids) == 0:
        return []
    take = min(n, len(ids))
    return [str(x) for x in rng.choice(ids, size=take, replace=False)]
