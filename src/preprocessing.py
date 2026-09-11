"""Phase 3 cleaning and conversation preparation for the selected brand."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from src.brand_analysis import conversation_labels, required_columns
from src.data_loader import parse_inbound_series

RELATION_COLUMNS = (
    "tweet_id",
    "author_id",
    "inbound",
    "in_response_to_tweet_id",
)
MESSAGE_COLUMNS = RELATION_COLUMNS + (
    "created_at",
    "text",
    "response_tweet_id",
)
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_numeric_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def load_relationship_table(
    csv_path: Path,
    mapping: dict[str, str],
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load only relationship fields and report invalid/duplicate IDs."""
    usecols = required_columns(mapping, RELATION_COLUMNS)
    parts: list[pd.DataFrame] = []
    invalid_ids = 0
    for chunk in iter_csv_chunks(csv_path, usecols, chunksize):
        tweet_id = normalize_numeric_id(chunk[mapping["tweet_id"]])
        valid = tweet_id.notna()
        invalid_ids += int((~valid).sum())
        inbound = parse_inbound_series(chunk[mapping["inbound"]])
        parent_id = normalize_numeric_id(chunk[mapping["in_response_to_tweet_id"]])
        parts.append(
            pd.DataFrame(
                {
                    "tweet_id": tweet_id,
                    "inbound": inbound,
                    "parent_id": parent_id,
                    "brand_account": normalize_id(chunk[mapping["author_id"]]).where(
                        inbound == False  # noqa: E712
                    ),
                }
            ).loc[valid]
        )
    if not parts:
        raise ValueError("CSV contained no rows.")
    table = pd.concat(parts, ignore_index=True)
    duplicate_ids = int(table["tweet_id"].duplicated(keep=False).sum())
    table = table.drop_duplicates("tweet_id", keep="first").reset_index(drop=True)
    table["brand_account"] = table["brand_account"].astype("category")
    return table, {
        "invalid_tweet_id_count": invalid_ids,
        "duplicate_tweet_id_count": duplicate_ids,
    }


def add_conversation_ids(table: pd.DataFrame) -> pd.DataFrame:
    work = table.copy()
    labels = conversation_labels(work["tweet_id"], work["parent_id"])
    work["conversation_id"] = work["tweet_id"].iloc[labels].to_numpy()
    return work


def selected_conversation_ids(table: pd.DataFrame, selected_brand: str) -> set[int]:
    mask = (table["inbound"] == False) & (table["brand_account"] == selected_brand)  # noqa: E712
    return set(table.loc[mask, "conversation_id"].astype("int64"))


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = CONTROL_RE.sub(" ", str(value))
    return WHITESPACE_RE.sub(" ", text).strip()


def collect_selected_messages(
    csv_path: Path,
    mapping: dict[str, str],
    relationship_table: pd.DataFrame,
    selected_conversations: set[int],
    selected_brand: str,
    chunksize: int,
) -> tuple[pd.DataFrame, int]:
    """Read text only for rows in selected conversations."""
    usecols = required_columns(mapping, MESSAGE_COLUMNS)
    conversation_lookup = relationship_table.set_index("tweet_id")["conversation_id"]
    parts: list[pd.DataFrame] = []
    missing_text = 0
    for chunk in iter_csv_chunks(csv_path, usecols, chunksize):
        ids = normalize_numeric_id(chunk[mapping["tweet_id"]])
        conversations = ids.map(conversation_lookup)
        hit = conversations.isin(selected_conversations)
        missing_text += int(chunk.loc[hit, mapping["text"]].isna().sum())
        if not hit.any():
            continue
        part = pd.DataFrame(
            {
                "tweet_id": ids.loc[hit].to_numpy(),
                "author_id": normalize_id(chunk.loc[hit, mapping["author_id"]]).to_numpy(),
                "inbound": parse_inbound_series(chunk.loc[hit, mapping["inbound"]]).to_numpy(),
                "parent_id": normalize_numeric_id(
                    chunk.loc[hit, mapping["in_response_to_tweet_id"]]
                ).to_numpy(),
                "response_tweet_ids": normalize_id(
                    chunk.loc[hit, mapping["response_tweet_id"]]
                ).to_numpy(),
                "created_at": chunk.loc[hit, mapping["created_at"]].to_numpy(),
                "raw_text": chunk.loc[hit, mapping["text"]].to_numpy(),
                "conversation_id": conversations.loc[hit].to_numpy(),
            }
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(), missing_text

    messages = pd.concat(parts, ignore_index=True)
    messages["timestamp"] = pd.to_datetime(messages["created_at"], utc=True, errors="coerce")
    messages["clean_text"] = messages["raw_text"].map(clean_text)
    messages["speaker"] = np.where(
        messages["inbound"].eq(True),
        "CUSTOMER",
        np.where(messages["author_id"].eq(selected_brand), "COMPANY", "OTHER_COMPANY"),
    )
    messages["brand"] = selected_brand
    messages["is_low_quality"] = messages["clean_text"].eq("") | messages["clean_text"].str.len().lt(8)
    messages = messages.sort_values(
        ["conversation_id", "timestamp", "tweet_id"],
        na_position="last",
    ).reset_index(drop=True)
    messages["message_index"] = messages.groupby("conversation_id", sort=False).cumcount()
    return messages, missing_text


def build_conversation_summary(messages: pd.DataFrame, selected_brand: str) -> pd.DataFrame:
    if messages.empty:
        return pd.DataFrame()
    grouped = messages.groupby("conversation_id", sort=False)
    summary = grouped.agg(
        message_count=("tweet_id", "size"),
        first_message_time=("timestamp", "min"),
        last_message_time=("timestamp", "max"),
    )
    summary["customer_message_count"] = grouped["speaker"].apply(lambda values: (values == "CUSTOMER").sum())
    summary["company_message_count"] = grouped["speaker"].apply(lambda values: (values == "COMPANY").sum())
    summary["has_customer_message"] = summary["customer_message_count"] > 0
    summary["has_company_response"] = summary["company_message_count"] > 0
    first_customer = messages.loc[messages["speaker"] == "CUSTOMER"].groupby("conversation_id")["timestamp"].min()
    first_company = messages.loc[messages["speaker"] == "COMPANY"].groupby("conversation_id")["timestamp"].min()
    summary["is_resolved_candidate"] = (
        summary["has_customer_message"]
        & summary["has_company_response"]
        & first_company.gt(first_customer).reindex(summary.index, fill_value=False)
    )
    summary["conversation_duration_hours"] = (
        summary["last_message_time"] - summary["first_message_time"]
    ).dt.total_seconds() / 3600
    summary["brand"] = selected_brand
    return summary.reset_index()


def build_customer_company_pairs(messages: pd.DataFrame, selected_brand: str) -> pd.DataFrame:
    customers = messages[messages["speaker"] == "CUSTOMER"]
    companies = messages[messages["speaker"] == "COMPANY"]
    pairs = customers.merge(
        companies,
        left_on="tweet_id",
        right_on="parent_id",
        suffixes=("_customer", "_company"),
    )
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "conversation_id", "customer_tweet_id", "customer_message",
                "company_tweet_id", "company_response", "timestamp", "brand",
            ]
        )
    return pd.DataFrame(
        {
            "conversation_id": pairs["conversation_id_customer"],
            "customer_tweet_id": pairs["tweet_id_customer"],
            "customer_message": pairs["clean_text_customer"],
            "company_tweet_id": pairs["tweet_id_company"],
            "company_response": pairs["clean_text_company"],
            "timestamp": pairs["timestamp_company"],
            "brand": selected_brand,
        }
    )


def build_relationship_index(
    csv_path: Path,
    mapping: dict[str, str],
    index_path: Path,
    chunksize: int,
) -> dict[str, int]:
    """Persist parent links on disk so the full corpus is not held in RAM."""
    if index_path.exists():
        index_path.unlink()
    connection = sqlite3.connect(index_path)
    connection.execute("CREATE TABLE tweets (tweet_id INTEGER PRIMARY KEY, parent_id INTEGER, inbound INTEGER, brand TEXT)")
    connection.execute("CREATE INDEX idx_brand ON tweets(brand)")
    connection.execute("CREATE INDEX idx_parent ON tweets(parent_id)")
    invalid_ids = 0
    raw_count = 0
    for chunk in iter_csv_chunks(csv_path, required_columns(mapping, RELATION_COLUMNS), chunksize):
        raw_count += len(chunk)
        tweet_ids = pd.to_numeric(chunk[mapping["tweet_id"]], errors="coerce")
        invalid_ids += int(tweet_ids.isna().sum())
        parents = pd.to_numeric(chunk[mapping["in_response_to_tweet_id"]], errors="coerce")
        inbound = parse_inbound_series(chunk[mapping["inbound"]])
        authors = normalize_id(chunk[mapping["author_id"]])
        rows = zip(
            tweet_ids,
            parents,
            inbound,
            authors.where(inbound == False),  # noqa: E712
        )
        connection.executemany(
            "INSERT OR IGNORE INTO tweets(tweet_id,parent_id,inbound,brand) VALUES (?,?,?,?)",
            [
                (int(tweet_id), None if pd.isna(parent) else int(parent), int(bool(flag)), brand)
                for tweet_id, parent, flag, brand in rows
                if pd.notna(tweet_id) and pd.notna(flag)
            ],
        )
        connection.commit()
    connection.close()
    return {"raw_record_count": raw_count, "invalid_tweet_id_count": invalid_ids}


def selected_roots(index_path: Path, selected_brand: str) -> set[int]:
    connection = sqlite3.connect(index_path)
    rows = connection.execute(
        """
        WITH RECURSIVE ancestry(tweet_id, root_id) AS (
            SELECT tweet_id, tweet_id FROM tweets WHERE brand = ?
            UNION ALL
            SELECT ancestry.tweet_id, tweets.parent_id
            FROM ancestry JOIN tweets ON tweets.tweet_id = ancestry.root_id
                        WHERE tweets.parent_id IS NOT NULL
                            AND EXISTS (SELECT 1 FROM tweets AS parent WHERE parent.tweet_id = tweets.parent_id)
        )
        SELECT DISTINCT root_id FROM ancestry
        """,
        (selected_brand,),
    ).fetchall()
    connection.close()
    return {int(row[0]) for row in rows}


def selected_relationship_table(index_path: Path, roots: set[int]) -> pd.DataFrame:
    if not roots:
        return pd.DataFrame(columns=["tweet_id", "inbound", "parent_id", "brand_account", "conversation_id"])
    connection = sqlite3.connect(index_path)
    placeholders = ",".join("?" for _ in roots)
    rows = connection.execute(
        f"""
        WITH RECURSIVE descendants(tweet_id, root_id) AS (
            SELECT tweet_id, tweet_id FROM tweets WHERE tweet_id IN ({placeholders})
            UNION ALL
            SELECT tweets.tweet_id, descendants.root_id
            FROM descendants JOIN tweets ON tweets.parent_id = descendants.tweet_id
        )
        SELECT tweets.tweet_id, tweets.inbound, tweets.parent_id, tweets.brand,
               descendants.root_id
        FROM descendants JOIN tweets ON tweets.tweet_id = descendants.tweet_id
        """,
        tuple(roots),
    ).fetchall()
    connection.close()
    return pd.DataFrame(
        rows,
        columns=["tweet_id", "inbound", "parent_id", "brand_account", "conversation_id"],
    )