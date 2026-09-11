# Data Splits

## Why splitting matters

Tweets from one support thread share context. Splitting individual tweets can
put a customer's question in one dataset and the brand's answer in another,
making evaluation unrealistically easy. Phase 4 therefore assigns each
conversation to exactly one split.

## Split strategy

Using seed `42`, the 82,556 AmazonHelp conversations are assigned by a stable
random permutation:

- 70% development: cleaned message-level conversations for intent discovery and later model development.
- 15% retrieval: direct customer/company response pairs for historical evidence.
- 15% golden pool: customer messages reserved for later manual selection and labeling.

The golden pool is intentionally larger than 200 examples. Later phases can
select approximately 200 examples while preserving coverage of difficult and
ambiguous cases.

## Golden pool

`data/splits/golden_pool.csv` contains customer messages only from golden-pool
conversations. It has no labels yet. It must remain locked and must not be used
for prompt tuning, threshold selection, intent discovery, retrieval, or model
selection.

## Leakage prevention

The script asserts that conversation IDs are disjoint. It also checks exact
normalized customer-message overlap across available outputs and reports cheap
character-signature overlap as a near-duplicate diagnostic. Repeated legitimate
support requests are measured, not aggressively deleted.

## Random seed and reproducibility

The seed is stored once in `configs/config.yaml` and recorded in
`data/splits/manifest.json`. Conversation IDs are sorted before the seeded
permutation, so rerunning the command produces the same assignment.

## Temporal analysis

`temporal_analysis.csv` compares a newest-15%-by-first-message-time robustness
holdout. It is not created as a model dataset because the main evaluation design
is the reproducible random conversation split. A temporal experiment can be
added later without changing the locked golden pool.

## Limitations

Text near-duplicates across unrelated conversations can remain. The current
diagnostic uses normalized exact text and character signatures rather than an
expensive all-pairs similarity search. Final intent labels do not yet exist, so
true intent-stratified sampling belongs to a later phase.