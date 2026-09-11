# Data Preparation

## Dataset schema

The local file is `data/raw/twcs.csv`. Its verified columns are:
`tweet_id`, `author_id`, `inbound`, `created_at`, `text`,
`response_tweet_id`, and `in_response_to_tweet_id`.

The selected brand is `AmazonHelp`, read from `configs/config.yaml` after the
Phase 2 recommendation. `inbound=True` identifies a customer message directed
at a support account. Outbound rows have the support account handle in
`author_id`; rows with `author_id == AmazonHelp` are company messages.

## Brand filtering

The full CSV is first indexed in a temporary SQLite database using only tweet
IDs, parent IDs, inbound flags, and outbound author handles. Recursive parent
and child queries identify roots and all descendants of conversations containing
an AmazonHelp message. Text is read in a second chunked pass only for those
selected conversations.

## Conversation reconstruction

`conversation_id` is the root `tweet_id` reached by repeatedly following
`in_response_to_tweet_id`. Orphan tweets use their own ID. Messages are sorted
by parsed UTC timestamp, with tweet ID as a deterministic tie-breaker. The
parent pointer is authoritative; `response_tweet_id` is preserved but is not
used to invent additional relationships.

## Cleaning rules

- Numeric tweet IDs are required; invalid IDs are excluded from the processed data.
- Duplicate tweet IDs are ignored after the first occurrence in the relationship index.
- `raw_text` is preserved unchanged.
- `clean_text` removes control characters and collapses whitespace.
- Empty or shorter-than-eight-character cleaned messages receive `is_low_quality=True`.
- Borderline or potentially spammy messages are flagged rather than deleted.
- Missing or unparsable timestamps are retained and sorted after valid timestamps.

## Roles and resolved candidates

Messages are labelled `CUSTOMER`, `COMPANY`, or `OTHER_COMPANY`. A resolved
candidate requires at least one customer message and at least one AmazonHelp
company message, with the first company response occurring after the first
customer message. This is only an operational historical-corpus signal; it does
not prove the customer's issue was actually resolved.

## Leakage prevention

No golden set or Phase 4 split is created here. Every output retains a stable
conversation ID so later development, retrieval, and evaluation data can be
split by conversation rather than by individual tweet. This prevents messages
from one thread appearing in multiple datasets.

## Outputs

- `data/processed/selected_brand_messages.csv`: cleaned message-level data.
- `data/processed/conversations.csv`: one row per conversation with quality metadata.
- `data/processed/conversation_pairs.csv`: direct customer-message/company-response pairs.
- `data/processed/data_quality_report.csv`: machine-readable counts and summary statistics.
- `data/processed/phase3_metadata.json`: schema and reconstruction metadata.

The temporary SQLite relationship index is deleted after a successful run.

## Limitations

Twitter reply metadata does not guarantee an actual business resolution. Public
threads can contain sensitive details, automated replies, multiple brands, and
long gaps between messages. Phase 4 should exclude mixed-brand threads where
appropriate and retain these flags for later failure analysis.