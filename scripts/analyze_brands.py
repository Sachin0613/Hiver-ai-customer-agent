"""Phase 2: brand statistics, candidates, and a data-driven recommendation.

    python scripts/analyze_brands.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.brand_analysis import (  # noqa: E402
    SCORE_WEIGHTS,
    add_conversation_ids,
    attach_text_metrics,
    collect_text_samples,
    load_slim_table,
    pick_candidates,
    reservoir_tweet_ids,
    sample_conversation_ids,
    score_brands,
    theme_table,
    aggregate_brands,
)
from src.data_loader import (  # noqa: E402
    choose_main_csv,
    column_map,
    discover_data_files,
    load_config,
    resolve_path,
)


def _read_header_mapping(csv_path: Path) -> dict[str, str]:
    header = pd.read_csv(csv_path, nrows=0)
    return column_map(list(header.columns))


def _fmt_table(df: pd.DataFrame, columns: list[str], n: int = 20) -> str:
    view = df[columns].head(n).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return view.to_string(index=False)


def main() -> int:
    config = load_config()
    paths = config["paths"]
    brand_cfg = config.get("brand_analysis", {})
    chunksize = int(brand_cfg.get("chunksize", 200_000))
    top_n = int(brand_cfg.get("top_n_table", 20))
    n_candidates = int(brand_cfg.get("n_candidates", 5))
    min_resolved = int(brand_cfg.get("min_resolved_conversations", 8000))
    sample_convs = int(brand_cfg.get("sample_conversations_per_brand", 12))
    text_sample = int(brand_cfg.get("text_sample_per_brand", 4000))
    seed = int(config["inspection"]["random_seed"])
    rng = np.random.default_rng(seed)

    raw_dir = resolve_path(paths["raw_dir"])
    processed_dir = resolve_path(paths["processed_dir"])
    reports_dir = resolve_path(paths["reports_dir"])
    results_dir = resolve_path(paths["results_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    files = discover_data_files(raw_dir)
    csv_path = choose_main_csv(files)
    if csv_path is None:
        print("No CSV found under data/raw/. See data/raw/README.md")
        return 1

    print(f"Dataset: {csv_path.relative_to(ROOT)}")
    print(f"Size: {csv_path.stat().st_size / (1024 * 1024):.1f} MB")
    mapping = _read_header_mapping(csv_path)
    print("Columns mapped:", mapping)

    print(f"Pass 1: slim columns in chunks of {chunksize:,} (no tweet text)...")
    slim = load_slim_table(csv_path, mapping, chunksize)
    print(f"Slim rows: {len(slim):,}")
    print("Building conversation ids (union-find)...")
    slim = add_conversation_ids(slim)
    print(f"Conversations: {slim['conv_id'].nunique():,}")

    print("Aggregating brand statistics...")
    stats = aggregate_brands(slim)
    if stats.empty:
        print("Unable to reliably calculate brand statistics from this file.")
        return 1

    stats_path = processed_dir / "brand_statistics.csv"
    stats.to_csv(stats_path, index=False)
    print(f"Wrote {stats_path.relative_to(ROOT)} ({len(stats)} brands)")

    table_cols = [
        "brand",
        "total_tweets",
        "customer_messages",
        "company_responses",
        "conversation_count",
        "resolved_conversations",
        "avg_conversation_length",
        "median_conversation_length",
        "response_rate",
    ]
    print("\nTop brands by resolved conversations:\n")
    print(_fmt_table(stats, table_cols, n=top_n))

    candidates = pick_candidates(stats, n=n_candidates, min_resolved=min_resolved)
    print("\nCandidate brands:", ", ".join(candidates))

    cand_stats = stats[stats["brand"].isin(candidates)].copy()

    print("Pass 2: sample tweet text for candidates only...")
    sample_ids: set[str] = set()
    conv_samples: dict[str, list[int]] = {}
    for brand in candidates:
        conv_samples[brand] = sample_conversation_ids(slim, brand, sample_convs, rng)
        sample_ids.update(
            slim.loc[slim["conv_id"].isin(conv_samples[brand]), "tweet_id"].astype(str)
        )
        sample_ids.update(reservoir_tweet_ids(slim, brand, "customer", text_sample, rng))
        sample_ids.update(reservoir_tweet_ids(slim, brand, "company", text_sample, rng))

    texts = collect_text_samples(csv_path, mapping, sample_ids, chunksize)
    if texts.empty:
        print("Unable to collect text samples for candidates.")
        reply_texts: dict[str, list[str]] = {}
        customer_texts: dict[str, list[str]] = {}
        conv_rows = pd.DataFrame()
    else:
        texts["tweet_id"] = texts["tweet_id"].astype(str)
        texts["inbound_bool"] = texts["inbound"].astype(str).str.lower().isin(
            {"true", "1", "yes", "t"}
        )
        slim_ids = slim[["tweet_id", "conv_id", "brand", "is_customer", "is_company"]].copy()
        slim_ids["tweet_id"] = slim_ids["tweet_id"].astype(str)
        texts = texts.merge(slim_ids, on="tweet_id", how="left")

        reply_texts = {}
        customer_texts = {}
        for brand in candidates:
            brand_convs = slim.loc[slim["brand"] == brand, "conv_id"]
            customer_texts[brand] = (
                texts.loc[
                    texts["inbound_bool"] & texts["conv_id"].isin(brand_convs),
                    "text",
                ]
                .fillna("")
                .astype(str)
                .tolist()
            )
            reply_texts[brand] = (
                texts.loc[texts["brand"] == brand, "text"].fillna("").astype(str).tolist()
            )

        conv_parts = []
        for brand, conv_ids in conv_samples.items():
            part = texts.loc[texts["conv_id"].isin(conv_ids)].copy()
            part["sample_brand"] = brand
            conv_parts.append(part)
        conv_rows = pd.concat(conv_parts, ignore_index=True) if conv_parts else pd.DataFrame()
        if not conv_rows.empty:
            conv_rows = conv_rows.sort_values(["sample_brand", "conv_id", "created_at"])

    cand_stats = attach_text_metrics(cand_stats, reply_texts, customer_texts)
    scored = score_brands(cand_stats)

    score_cols = [
        "brand",
        "volume_score",
        "resolution_score",
        "quality_score",
        "diversity_score",
        "response_usefulness_score",
        "overall_score",
        "resolved_conversations",
        "response_rate",
        "useful_reply_rate",
        "dm_template_rate",
        "ngram_diversity",
    ]
    print("\nCandidate scores (weights: "
          + ", ".join(f"{k}={v:.0%}" for k, v in SCORE_WEIGHTS.items())
          + "):\n")
    print(_fmt_table(scored, [c for c in score_cols if c in scored.columns], n=n_candidates))

    recommended = str(scored.iloc[0]["brand"]) if not scored.empty else ""
    print(f"\nRecommended brand: {recommended}")

    if not conv_rows.empty:
        conv_path = processed_dir / "candidate_conversations.csv"
        keep = [
            c
            for c in [
                "sample_brand",
                "conv_id",
                "tweet_id",
                "created_at",
                "author_id",
                "inbound",
                "text",
            ]
            if c in conv_rows.columns
        ]
        conv_rows[keep].to_csv(conv_path, index=False)
        print(f"Wrote {conv_path.relative_to(ROOT)} ({len(conv_rows)} tweets)")

    themes = {}
    for brand in candidates:
        themes[brand] = theme_table(customer_texts.get(brand, [])).to_dict(orient="records")

    payload = {
        "dataset": str(csv_path.relative_to(ROOT)),
        "n_slim_rows": int(len(slim)),
        "n_brands": int(len(stats)),
        "candidates": candidates,
        "recommended_brand": recommended,
        "score_weights": SCORE_WEIGHTS,
        "candidate_scores": scored.to_dict(orient="records"),
        "themes": themes,
        "seed": seed,
    }
    json_path = results_dir / "phase2_brand_selection.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)}")

    rec_row = scored.iloc[0].to_dict() if not scored.empty else {}
    rec_full = stats[stats["brand"] == recommended].iloc[0].to_dict() if recommended else {}
    _write_docs(
        reports_dir=reports_dir,
        stats=stats,
        scored=scored,
        top_n=top_n,
        candidates=candidates,
        recommended=recommended,
        rec_row=rec_row,
        rec_full=rec_full,
        themes=themes,
        seed=seed,
        csv_name=str(csv_path.relative_to(ROOT)),
        n_rows=len(slim),
    )
    print(f"Wrote { (reports_dir / 'brand_selection.md').relative_to(ROOT) }")
    return 0


def _md_table(df: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
    view = df[columns] if n is None else df[columns].head(n)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for _, row in view.iterrows():
        cells = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.3f}" if col.endswith("rate") or col.endswith("score") or "avg" in col or "median" in col or "diversity" in col else f"{val:.2f}")
            elif isinstance(val, (int, np_int := int)) or hasattr(val, "item"):
                try:
                    ival = int(val)
                    if abs(float(val) - ival) < 1e-9 and not str(col).endswith(("rate", "score", "length")):
                        cells.append(f"{ival:,}")
                    elif str(col).endswith(("length",)) and "avg" not in col and "median" not in col:
                        cells.append(f"{ival:,}")
                    else:
                        cells.append(str(val))
                except (TypeError, ValueError):
                    cells.append(str(val))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_docs(
    reports_dir: Path,
    stats: pd.DataFrame,
    scored: pd.DataFrame,
    top_n: int,
    candidates: list[str],
    recommended: str,
    rec_row: dict,
    rec_full: dict,
    themes: dict,
    seed: int,
    csv_name: str,
    n_rows: int,
) -> None:
    import textwrap

    top_cols = [
        "brand",
        "total_tweets",
        "customer_messages",
        "company_responses",
        "conversation_count",
        "resolved_conversations",
        "avg_conversation_length",
        "median_conversation_length",
        "response_rate",
    ]
    score_cols = [
        "brand",
        "volume_score",
        "resolution_score",
        "quality_score",
        "diversity_score",
        "response_usefulness_score",
        "overall_score",
    ]

    theme_blocks = []
    for brand in candidates:
        rows = themes.get(brand, [])
        if not rows:
            theme_blocks.append(f"### {brand}\n\nNo customer-text sample collected.\n")
            continue
        lines = [f"### {brand}", "", "| theme_bigram | count |", "| --- | --- |"]
        for row in rows[:12]:
            lines.append(f"| {row['theme_bigram']} | {row['count']} |")
        theme_blocks.append("\n".join(lines) + "\n")

    rec_resolved = int(rec_full.get("resolved_conversations", 0) or 0)
    rec_cust = int(rec_full.get("customer_messages", 0) or 0)
    rec_resp = int(rec_full.get("company_responses", 0) or 0)
    rec_rate = float(rec_full.get("response_rate", 0) or 0)
    golden_ok = rec_resolved >= 2000 and rec_cust >= 5000

    others = [b for b in candidates if b != recommended]

    selection = f"""# Brand Selection

## Dataset

Source file: `{csv_name}` ({n_rows:,} tweets in the slim pass).

Each row is a tweet. Company accounts are `author_id` values on outbound tweets
(`inbound=False`). Conversations are connected components using
`in_response_to_tweet_id` as a parent pointer (union-find). A conversation is
**resolved** when it contains at least one customer tweet and at least one reply
from that brand. **Response rate** is the share of customer tweets in the brand's
threads that are the parent of a reply from that same brand.

Numbers below are computed from the local CSV. They are not estimates.

## Selection Criteria

We did **not** pick the brand with the most tweets.

Structural metrics (full corpus, no text):

- resolved conversation count (historical examples for retrieval)
- direct response rate (coverage)
- median / multi-turn length (usable threads vs one-off pings)

Text metrics (fixed-seed sample of candidate tweets only):

- `useful_reply_rate`: company replies with length ≥ 40 that are not short DM-only templates
- `ngram_diversity`: share of sampled customer messages that contain a recurring content bigram

Overall score weights:

| Factor | Weight | Source |
| --- | --- | --- |
| Volume | {SCORE_WEIGHTS['volume']:.0%} | log1p(resolved conversations), min-max among candidates |
| Resolution | {SCORE_WEIGHTS['resolution']:.0%} | response_rate |
| Quality | {SCORE_WEIGHTS['quality']:.0%} | median length near 4 turns + multi-turn resolved share |
| Diversity | {SCORE_WEIGHTS['diversity']:.0%} | ngram_diversity, min-max among candidates |
| Response usefulness | {SCORE_WEIGHTS['response_usefulness']:.0%} | useful_reply_rate |

## Top Brands

Sorted by resolved conversations (more useful for retrieval than raw tweet count).

{_md_table(stats, top_cols, n=top_n)}

## Candidate Brands

Candidates are the top {len(candidates)} brands by `resolved_conversations * response_rate`
among brands with at least the configured minimum resolved conversations.

{', '.join(f'`{b}`' for b in candidates)}

Why they were considered: each has enough resolved threads to support intent
labels, retrieval, and a ~200-example golden set. Final ranking uses the score
table, not volume alone.

{_md_table(scored, score_cols)}

## Preliminary themes (customer-message bigrams)

These are **not** the final intent taxonomy. They are the most common content
bigrams in a seeded sample of inbound messages.

{''.join(theme_blocks)}

## Final Recommendation

**{recommended or 'NONE'}**

## Why This Brand

- Resolved conversations: **{rec_resolved:,}**
- Customer messages in its threads: **{rec_cust:,}**
- Company responses: **{rec_resp:,}**
- Direct response rate: **{rec_rate:.1%}**
- Overall score among candidates: **{float(rec_row.get('overall_score', 0) or 0):.3f}**
- Useful-reply rate (sample): **{float(rec_row.get('useful_reply_rate', 0) or 0):.1%}**
- DM-template rate (sample): **{float(rec_row.get('dm_template_rate', 0) or 0):.1%}**

It outranked {', '.join(f'`{b}`' for b in others) or 'no other candidates'} on the
weighted score, which balances volume with whether agents actually write grounded
replies (not only “please DM us”) and whether customer wording repeats enough
to learn intents.

Enough data: yes, if resolved conversations are in the tens of thousands (see
count above). Enough historical responses: company_responses is the retrieval
pool size. Recurring topics: see bigram table. Golden set of 150–250: feasible
when resolved conversations ≫ 200 (see feasibility below).

## Risks

- Twitter threads are public and often end with “DM us”; retrieval must not treat
  those as full resolutions.
- Mixed-brand conversations exist; later sampling should drop them.
- Anonymized handles and PII masks (`__email__`) appear in text.
- Language mix: if non-English volume is high, intents get noisier. Phase 3
  should filter or tag language if needed.
- CC BY-NC-SA 4.0 license: keep the CSV out of git.

## Golden-set feasibility ({recommended or 'n/a'})

- Available resolved conversations: {rec_resolved:,}
- Hand-label target: ~200
- Feasible from volume: {'yes' if golden_ok else 'borderline / check counts'}
- Common intents: take the highest-count bigrams as seeds, then label manually
- Rare / ambiguous / multi-intent: oversample short unclear tweets and tweets
  whose bigrams hit two themes

## Sampling Strategy

Unit = **conversation** (`conv_id`), never individual tweets. Seed = **{seed}**.

1. Keep conversations where `{recommended}` is the only company author.
2. Hash `conv_id` with seed {seed}.
3. Split conversations: **70% retrieval corpus**, **20% development**, **10% held-out**.
4. Draw the golden set (~200) **only** from held-out conversations so retrieval
   cannot see evaluation threads.
5. After Phase 5 intents exist, stratify the golden set by intent, plus buckets
   for ambiguous and multi-intent messages.
6. Do not put golden-set tweet ids into the FAISS index.

Development may include unlabeled retrieval threads for debugging, but evaluation
metrics for the report must use the golden set only.
"""

    (reports_dir / "brand_selection.md").write_text(selection, encoding="utf-8")

    decision_path = reports_dir / "decision_log.md"
    existing = decision_path.read_text(encoding="utf-8") if decision_path.exists() else "# Decision log\n"
    block = textwrap.dedent(
        f"""

        ## D7 — Selected `{recommended}` as the support brand (Phase 2)

        * Decision: Selected **{recommended}**
        * Reason: Highest overall score among candidates using the documented weights
          (volume 20%, resolution 25%, quality 20%, diversity 15%, response usefulness 20%).
        * Alternatives considered: {', '.join(candidates)}
        * Evidence: `data/processed/brand_statistics.csv`, `evaluation/results/phase2_brand_selection.json`
        * Trade-offs: another brand may have more raw tweets; we preferred resolved
          threads and less DM-only reply behaviour over maximum volume.
        """
    )
    if "D7 —" not in existing:
        decision_path.write_text(existing.rstrip() + "\n" + block, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
