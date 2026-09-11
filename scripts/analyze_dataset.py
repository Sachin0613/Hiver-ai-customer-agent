"""Phase 1: inspect local files and write a factual dataset report.

Run from the project root after placing the Kaggle CSV under data/raw/:

    python scripts/analyze_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    choose_main_csv,
    column_map,
    discover_data_files,
    load_config,
    load_tweets_csv,
    parse_inbound_series,
    resolve_path,
)


def _md_table(rows: list[list[str]]) -> str:
    if not rows:
        return "_none_"
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _escape_cell(value: object, limit: int = 120) -> str:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    text = text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def reconstruct_conversation_ids(
    tweet_id: pd.Series,
    parent_id: pd.Series,
) -> pd.Series:
    """Walk each tweet to its root via in_response_to_tweet_id.

    Tweets whose parent is missing from the file become their own root.
    This is a simple parent-pointer walk, not a full graph library.
    """
    parent_lookup = dict(zip(tweet_id.tolist(), parent_id.tolist()))
    roots: dict[str, str] = {}

    def find_root(node: str) -> str:
        seen: list[str] = []
        current = node
        while True:
            if current in roots:
                root = roots[current]
                break
            if current in seen:
                root = current
                break
            seen.append(current)
            parent = parent_lookup.get(current)
            if parent is None or (isinstance(parent, float) and pd.isna(parent)):
                root = current
                break
            parent_str = str(parent).strip()
            if parent_str == "" or parent_str.lower() == "nan" or parent_str not in parent_lookup:
                root = current
                break
            current = parent_str
        for item in seen:
            roots[item] = root
        return root

    return tweet_id.map(lambda value: find_root(str(value)))


def analyze() -> int:
    config = load_config()
    paths = config["paths"]
    inspection = config["inspection"]

    raw_dir = resolve_path(paths["raw_dir"])
    reports_dir = resolve_path(paths["reports_dir"])
    results_dir = resolve_path(paths["results_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    files = discover_data_files(raw_dir)
    main_csv = choose_main_csv(files)

    report_lines: list[str] = [
        "# Phase 1 dataset inspection",
        "",
        f"Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "All numbers below come from files on disk. They are not estimates.",
        "",
        "## 1. Files found",
        "",
    ]

    payload: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "files": [
            {"path": str(item.path.relative_to(ROOT)), "size_bytes": item.size_bytes}
            for item in files
        ],
        "main_csv": None,
        "blocked": False,
    }

    if not files:
        payload["blocked"] = True
        payload["block_reason"] = "No CSV/ZIP files under data/raw/"
        report_lines.extend(
            [
                "**No dataset files found.**",
                "",
                f"Looked in: `{raw_dir}`",
                "",
                "Download the Kaggle dataset `thoughtvector/customer-support-on-twitter` ",
                "and place `twcs.csv` under `data/raw/` (see `data/raw/README.md`).",
                "",
                "Until that file exists, brand counts, samples, and a brand recommendation ",
                "**cannot** be produced without fabricating results.",
                "",
            ]
        )
        (reports_dir / "phase1_report.md").write_text("\n".join(report_lines), encoding="utf-8")
        (results_dir / "phase1_stats.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print("\n".join(report_lines))
        return 1

    file_rows = [["relative_path", "size_bytes", "size_mb"]]
    for item in files:
        rel = str(item.path.relative_to(ROOT))
        file_rows.append(
            [rel, str(item.size_bytes), f"{item.size_bytes / (1024 * 1024):.2f}"]
        )
    report_lines.append(_md_table(file_rows))
    report_lines.append("")

    if main_csv is None:
        payload["blocked"] = True
        payload["block_reason"] = "Found archives but no CSV to inspect"
        report_lines.extend(
            [
                "**No CSV file found.** If you downloaded a zip, extract it into `data/raw/`.",
                "",
            ]
        )
        (reports_dir / "phase1_report.md").write_text("\n".join(report_lines), encoding="utf-8")
        (results_dir / "phase1_stats.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print("\n".join(report_lines))
        return 1

    payload["main_csv"] = str(main_csv.relative_to(ROOT))
    report_lines.extend(
        [
            f"**Main file used for analysis:** `{main_csv.relative_to(ROOT)}`",
            "",
            "## 2. Shape and columns",
            "",
        ]
    )

    df = load_tweets_csv(main_csv)
    mapping = column_map(list(df.columns))
    payload["n_rows"] = int(len(df))
    payload["n_columns"] = int(df.shape[1])
    payload["columns"] = list(df.columns)
    payload["documented_columns_present"] = mapping
    payload["documented_columns_missing"] = sorted(
        name for name in (
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        )
        if name not in mapping
    )

    report_lines.append(f"- Rows: **{len(df):,}**")
    report_lines.append(f"- Columns: **{df.shape[1]}**")
    report_lines.append("")
    report_lines.append("### Actual header")
    report_lines.append("")
    dtype_rows = [["column", "non_null", "null", "example_dtype_after_str_load"]]
    for col in df.columns:
        nulls = int(df[col].isna().sum())
        dtype_rows.append([col, str(int(df[col].notna().sum())), str(nulls), str(df[col].dtype)])
    report_lines.append(_md_table(dtype_rows))
    report_lines.append("")
    report_lines.append("### Mapping to Kaggle-documented names")
    report_lines.append("")
    report_lines.append(
        "These names come from the public dataset page. They are used only if present."
    )
    report_lines.append("")
    map_rows = [["documented_name", "actual_column"]]
    for documented, actual in mapping.items():
        map_rows.append([documented, actual])
    if len(map_rows) == 1:
        report_lines.append("_No documented names matched the header._")
    else:
        report_lines.append(_md_table(map_rows))
    if payload["documented_columns_missing"]:
        report_lines.append("")
        report_lines.append(
            "Missing documented names: "
            + ", ".join(f"`{name}`" for name in payload["documented_columns_missing"])
        )
    report_lines.append("")

    report_lines.extend(["## 3. Sample records", ""])
    sample_n = min(int(inspection["sample_rows"]), len(df))
    sample = df.head(sample_n)
    payload["sample_records"] = sample.fillna("").to_dict(orient="records")
    for i, row in sample.iterrows():
        report_lines.append(f"### Row index {i}")
        report_lines.append("")
        report_lines.append("```")
        for col in df.columns:
            report_lines.append(f"{col}: {_escape_cell(row[col], limit=400)}")
        report_lines.append("```")
        report_lines.append("")

    report_lines.extend(["## 4. Missing values", ""])
    missing_rows = [["column", "n_missing", "pct_missing"]]
    missing_payload = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct = 100.0 * n_missing / max(len(df), 1)
        missing_rows.append([col, f"{n_missing:,}", f"{pct:.2f}%"])
        missing_payload[col] = {"n_missing": n_missing, "pct_missing": round(pct, 4)}
    payload["missing"] = missing_payload
    report_lines.append(_md_table(missing_rows))
    report_lines.append("")

    report_lines.extend(["## 5. Duplicates", ""])
    full_dupes = int(df.duplicated().sum())
    payload["duplicate_full_rows"] = full_dupes
    report_lines.append(f"- Duplicate full rows: **{full_dupes:,}**")
    tweet_col = mapping.get("tweet_id")
    if tweet_col:
        id_dupes = int(df[tweet_col].duplicated().sum())
        n_unique = int(df[tweet_col].nunique(dropna=False))
        payload["duplicate_tweet_id"] = id_dupes
        payload["unique_tweet_id"] = n_unique
        report_lines.append(f"- Duplicate `{tweet_col}` values: **{id_dupes:,}**")
        report_lines.append(f"- Unique `{tweet_col}`: **{n_unique:,}**")
    report_lines.append("")

    author_col = mapping.get("author_id")
    inbound_col = mapping.get("inbound")
    parent_col = mapping.get("in_response_to_tweet_id")
    response_col = mapping.get("response_tweet_id")
    text_col = mapping.get("text")
    created_col = mapping.get("created_at")

    report_lines.extend(
        [
            "## 6. Brand / account fields",
            "",
            "There is no separate `brand` column in the documented schema. ",
            "Brand accounts are expected to appear in the author field on **outbound** ",
            "(company) tweets. Customer accounts are typically anonymized IDs.",
            "",
        ]
    )

    inbound_parsed = None
    if inbound_col:
        inbound_parsed = parse_inbound_series(df[inbound_col])
        inbound_counts = inbound_parsed.value_counts(dropna=False)
        payload["inbound_value_counts"] = {
            str(k): int(v) for k, v in inbound_counts.items()
        }
        report_lines.append(f"`{inbound_col}` value counts (normalized):")
        report_lines.append("")
        inbound_rows = [["inbound", "count"]]
        for key, value in inbound_counts.items():
            inbound_rows.append([str(key), f"{int(value):,}"])
        report_lines.append(_md_table(inbound_rows))
        report_lines.append("")
        report_lines.append(
            "Interpretation used in this script: `inbound=True` → customer message "
            "to a company; `inbound=False` → company/support reply."
        )
        report_lines.append("")

    report_lines.extend(["## 7. Customer vs company tweets", ""])
    brand_counts = pd.Series(dtype="int64")
    if author_col and inbound_parsed is not None:
        company_mask = inbound_parsed == False  # noqa: E712
        customer_mask = inbound_parsed == True  # noqa: E712
        payload["n_company_tweets"] = int(company_mask.sum())
        payload["n_customer_tweets"] = int(customer_mask.sum())
        payload["n_inbound_unknown"] = int(inbound_parsed.isna().sum())
        report_lines.append(f"- Customer (inbound True): **{payload['n_customer_tweets']:,}**")
        report_lines.append(f"- Company (inbound False): **{payload['n_company_tweets']:,}**")
        report_lines.append(f"- Inbound unparseable: **{payload['n_inbound_unknown']:,}**")
        report_lines.append("")

        brand_counts = df.loc[company_mask, author_col].value_counts()
        top_n = int(inspection["top_brands"])
        top = brand_counts.head(top_n)
        payload["top_company_authors"] = {str(k): int(v) for k, v in top.items()}
        payload["n_distinct_company_authors"] = int(brand_counts.shape[0])

        report_lines.append(f"Distinct company `author_id` values: **{brand_counts.shape[0]:,}**")
        report_lines.append("")
        report_lines.append(f"### Top {top_n} company authors by outbound tweet count")
        report_lines.append("")
        brand_rows = [["rank", "author_id", "company_tweets"]]
        for i, (author, count) in enumerate(top.items(), start=1):
            brand_rows.append([str(i), str(author), f"{int(count):,}"])
        report_lines.append(_md_table(brand_rows))
        report_lines.append("")
    else:
        report_lines.append(
            "Could not split customer/company tweets because `author_id` and/or "
            "`inbound` are missing from the file."
        )
        report_lines.append("")

    report_lines.extend(["## 8. Conversation / thread reconstruction", ""])
    if tweet_col and parent_col:
        parent_missing = df[parent_col].isna() | (df[parent_col].astype(str).str.strip() == "")
        payload["n_root_like_tweets"] = int(parent_missing.sum())
        report_lines.append(
            f"- Tweets with empty `{parent_col}` (thread roots or missing parent): "
            f"**{payload['n_root_like_tweets']:,}**"
        )
        if response_col:
            has_response = df[response_col].notna() & (df[response_col].astype(str).str.strip() != "")
            payload["n_tweets_with_response_id"] = int(has_response.sum())
            report_lines.append(
                f"- Tweets with non-empty `{response_col}`: "
                f"**{payload['n_tweets_with_response_id']:,}**"
            )
        report_lines.append("")
        report_lines.append(
            "Threads are reconstructed by following `{parent}` until the parent is "
            "missing or not in the file. That root `tweet_id` is the conversation id."
            .replace("{parent}", parent_col)
        )
        report_lines.append("")
        report_lines.append("Computing conversation ids (this can take a minute on the full corpus)...")
        print("Computing conversation ids...")
        conv_ids = reconstruct_conversation_ids(df[tweet_col], df[parent_col])
        n_conv = int(conv_ids.nunique())
        sizes = conv_ids.value_counts()
        payload["n_conversations"] = n_conv
        payload["conversation_size"] = {
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "median": float(sizes.median()),
            "mean": float(sizes.mean()),
        }
        report_lines.append(f"- Distinct conversations: **{n_conv:,}**")
        report_lines.append(
            "- Tweets per conversation: "
            f"min={sizes.min()}, median={sizes.median():.1f}, "
            f"mean={sizes.mean():.2f}, max={sizes.max()}"
        )
        report_lines.append("")

        if author_col and inbound_parsed is not None:
            df = df.copy()
            df["_conv_id"] = conv_ids
            df["_inbound"] = inbound_parsed
            company_by_conv: dict[str, set[str]] = defaultdict(set)
            company_mask = df["_inbound"] == False  # noqa: E712
            for conv_id, author in zip(
                df.loc[company_mask, "_conv_id"],
                df.loc[company_mask, author_col],
            ):
                company_by_conv[str(conv_id)].add(str(author))

            brand_conversations: dict[str, int] = defaultdict(int)
            mixed_conversations = 0
            for authors in company_by_conv.values():
                if len(authors) == 1:
                    brand_conversations[next(iter(authors))] += 1
                elif len(authors) > 1:
                    mixed_conversations += 1
            payload["n_conversations_with_company_reply"] = len(company_by_conv)
            payload["n_mixed_brand_conversations"] = mixed_conversations
            payload["conversations_per_brand_top"] = dict(
                sorted(brand_conversations.items(), key=lambda kv: kv[1], reverse=True)[:25]
            )
            report_lines.append(
                f"- Conversations with at least one company tweet: "
                f"**{len(company_by_conv):,}**"
            )
            report_lines.append(
                f"- Conversations with more than one company author: **{mixed_conversations:,}**"
            )
            report_lines.append("")
            report_lines.append("### Top brands by conversation count (single-company threads)")
            report_lines.append("")
            conv_rows = [["rank", "author_id", "conversations"]]
            top_conv = sorted(brand_conversations.items(), key=lambda kv: kv[1], reverse=True)[:25]
            for i, (author, count) in enumerate(top_conv, start=1):
                conv_rows.append([str(i), author, f"{count:,}"])
            report_lines.append(_md_table(conv_rows) if len(conv_rows) > 1 else "_none_")
            report_lines.append("")
    else:
        report_lines.append(
            "Cannot reconstruct threads: need `tweet_id` and `in_response_to_tweet_id`."
        )
        report_lines.append("")

    report_lines.extend(["## 9. Extra quality checks", ""])
    quality_notes: list[str] = []
    if text_col:
        empty_text = int(df[text_col].isna().sum() + (df[text_col].astype(str).str.strip() == "").sum())
        payload["n_empty_text"] = empty_text
        quality_notes.append(f"Empty `{text_col}`: {empty_text:,}")
        masked = int(df[text_col].astype(str).str.contains(r"__email__|__phone__", regex=True).sum())
        payload["n_rows_with_pii_masks"] = masked
        quality_notes.append(f"Rows with `__email__` / `__phone__` masks: {masked:,}")
    if created_col:
        parsed_dates = pd.to_datetime(df[created_col], errors="coerce", utc=True)
        n_bad_dates = int(parsed_dates.isna().sum())
        payload["n_unparseable_created_at"] = n_bad_dates
        quality_notes.append(f"Unparseable `{created_col}`: {n_bad_dates:,}")
        if parsed_dates.notna().any():
            payload["created_at_min"] = str(parsed_dates.min())
            payload["created_at_max"] = str(parsed_dates.max())
            quality_notes.append(
                f"`{created_col}` range: {parsed_dates.min()} → {parsed_dates.max()}"
            )
    for note in quality_notes:
        report_lines.append(f"- {note}")
    if not quality_notes:
        report_lines.append("- No extra checks ran (text/created_at columns missing).")
    report_lines.append("")

    min_company = int(inspection["min_company_tweets"])
    report_lines.extend(
        [
            "## 10. Candidate brands",
            "",
            f"Rule used here: company authors with at least **{min_company:,}** outbound tweets.",
            "",
        ]
    )
    candidates: list[dict] = []
    if not brand_counts.empty:
        sufficient = brand_counts[brand_counts >= min_company]
        payload["min_company_tweets_threshold"] = min_company
        payload["n_brands_meeting_volume"] = int(sufficient.shape[0])
        cand_rows = [["author_id", "company_tweets"]]
        for author, count in sufficient.items():
            cand_rows.append([str(author), f"{int(count):,}"])
            candidates.append({"author_id": str(author), "company_tweets": int(count)})
        payload["brands_meeting_volume"] = candidates
        report_lines.append(_md_table(cand_rows) if len(cand_rows) > 1 else "_none met the threshold_")
        report_lines.append("")
        report_lines.append(
            "Final 2–3 interview candidates should also consider: English-heavy text, "
            "a coherent product domain (easier intent taxonomy), mix of common vs rare "
            "issues, and enough customer↔agent pairs for retrieval. Those filters belong "
            "in Phase 2 once this table is real."
        )
        report_lines.append("")
        report_lines.append("## 11. Brand recommendation (from local counts)")
        report_lines.append("")
        if candidates:
            top_author = candidates[0]["author_id"]
            report_lines.append(
                f"Highest outbound volume in this file: **{top_author}** "
                f"({candidates[0]['company_tweets']:,} company tweets)."
            )
            report_lines.append("")
            report_lines.append(
                "Volume alone is not a complete reason to lock a brand. Use Phase 2 to "
                "compare reply coverage, language, and issue diversity among the top 2–3."
            )
            payload["highest_volume_brand"] = top_author
        else:
            report_lines.append("No brand met the volume threshold.")
            payload["highest_volume_brand"] = None
    else:
        report_lines.append("Brand ranking skipped (author/inbound unavailable).")
        payload["highest_volume_brand"] = None
    report_lines.append("")

    report_lines.extend(
        [
            "## 12. Proposed sampling strategy (not executed yet)",
            "",
            "Unit of sampling: **conversation** (root tweet id), not individual tweets. ",
            "This avoids leaking the same thread into both retrieval and evaluation.",
            "",
            "After a brand is locked in Phase 2:",
            "",
            "1. Keep only conversations that include that brand's company author.",
            "2. Drop mixed-brand threads.",
            "3. Hash the conversation id with a fixed seed (`random_seed` in `configs/config.yaml`).",
            "4. Split conversations into development / retrieval index / held-out evaluation.",
            "5. Draw the golden set (~200) **only** from the held-out evaluation conversations.",
            "6. Stratify later by intent once intents exist (Phase 5); until then, sample ",
            "   by conversation length and inbound-message diversity so rare long threads remain.",
            "",
        ]
    )

    report_path = reports_dir / "phase1_report.md"
    json_path = results_dir / "phase1_stats.json"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nWrote {report_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(analyze())
