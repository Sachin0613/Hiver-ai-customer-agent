# Decision Log

Interview-facing record of the project's non-obvious technical decisions.

## D1 — Inspect the local schema before implementing behavior

**Decision:** Read the actual CSV header and sample rows before relying on documented Kaggle fields.

**Reason:** Dataset assumptions would make brand filtering and thread reconstruction unreliable.

**Alternative considered:** Hardcode the commonly documented schema.

**Trade-off:** A small discovery step adds code, but prevents silent schema errors.

## D2 — Select the brand empirically

**Decision:** Select `AmazonHelp` using full-corpus volume, response coverage, conversation quality, topic diversity, and sampled response usefulness.

**Reason:** The largest brand is not automatically the best support-agent dataset.

**Alternative considered:** Choose a brand from prior knowledge or raw tweet count.

**Trade-off:** A weighted score is less intuitive than volume alone, but is more aligned with usable support evidence.

## D3 — Keep raw text and create cleaned text

**Decision:** Preserve `raw_text` and add `clean_text` with whitespace/control-character normalization.

**Reason:** Original text is required for auditability and evidence.

**Alternative considered:** Overwrite text during preprocessing.

**Trade-off:** Outputs are slightly wider, but cleaning remains reversible.

## D4 — Use parent links to define conversations

**Decision:** Derive a deterministic conversation ID from the root reached through `in_response_to_tweet_id`.

**Reason:** Tweets are not independent support tickets; parent links are the available thread structure.

**Alternative considered:** Treat every tweet as an independent example.

**Trade-off:** Broken references and long threads remain imperfect, but context is preserved.

## D5 — Call threads resolved candidates, not resolved cases

**Decision:** Require a customer message followed by a company response for `is_resolved_candidate`.

**Reason:** A response does not prove that the customer's issue was actually solved.

**Alternative considered:** Label every company-response thread as resolved.

**Trade-off:** The definition is conservative and may understate success, but avoids an unsupported claim.

## D6 — Use a disk-backed relationship pass

**Decision:** Use chunked CSV processing and a temporary SQLite relationship index for the multi-million-row corpus.

**Reason:** Holding the complete relationship and text tables in memory was not reliable on a normal laptop.

**Alternative considered:** Load the full CSV and all relationships into pandas at once.

**Trade-off:** SQLite adds an intermediate step, but bounds memory use.

## D7 — Split by conversation, never by tweet

**Decision:** Assign each conversation to exactly one development, retrieval, or golden split.

**Reason:** Tweet-level splitting leaks context between customer messages and company replies.

**Alternative considered:** Randomly split individual messages.

**Trade-off:** Fewer independent units are available, but evaluation is more credible.

## D8 — Reserve the golden pool before development

**Decision:** Use a seeded 70/15/15 conversation split and keep the golden pool out of development and retrieval.

**Reason:** Evaluation examples must remain unseen during tuning and model selection.

**Alternative considered:** Create the evaluation set after model development.

**Trade-off:** Less data is available for development, but leakage risk is materially lower.

## D9 — Use a provisional AmazonHelp taxonomy

**Decision:** Derive nine provisional intents from recurring AmazonHelp development language and require manual review.

**Reason:** An unrelated taxonomy would not represent this brand's actual support problems.

**Alternative considered:** Import Banking77 or assign keyword labels automatically.

**Trade-off:** Manual annotation takes time, but labels are more defensible.

## D10 — Do not fabricate labels for baselines

**Decision:** Keep heuristic sampling buckets separate from blank manual `intent_label` fields, and block baseline metrics until labels exist.

**Reason:** Keyword buckets are useful for sampling but are not reliable supervised labels.

**Alternative considered:** Treat bucket names as gold labels.

**Trade-off:** Baseline execution is delayed, but every reported metric remains real.

## D11 — Use TF-IDF plus Logistic Regression as the classical baseline

**Decision:** Use word/bigram TF-IDF with Logistic Regression and a reproducible stratified validation split.

**Reason:** It is transparent, inexpensive, and gives a meaningful pre-LLM comparison.

**Alternative considered:** Start directly with embeddings or an LLM classifier.

**Trade-off:** It misses semantic similarity, but its errors and features are easy to explain.

## D12 — Use exact FAISS retrieval over normalized embeddings

**Decision:** Embed customer/company retrieval pairs with `all-MiniLM-L6-v2` and use FAISS `IndexFlatIP` on normalized vectors.

**Reason:** The retrieval corpus is small enough for exact search, and normalized inner product is cosine similarity.

**Alternative considered:** Approximate search or a database-backed vector service.

**Trade-off:** Exact search is simpler and more reproducible, with less infrastructure and potentially higher latency at larger scale.

## D13 — Make escalation deterministic for high-risk cases

**Decision:** Escalate legal, security, sensitive-information, severe-complaint, weak-evidence, and low-confidence cases before automatic handling.

**Reason:** These cases have asymmetric production risk and should not depend solely on an LLM.

**Alternative considered:** Let the LLM decide every escalation.

**Trade-off:** Rules may over-escalate some cases, but they are auditable and safer to tune on development data.

## D14 — Require structured, evidence-linked LLM outputs

**Decision:** Require JSON containing the reply, evidence indexes, uncertainty, and unsupported-claim notes.

**Reason:** Downstream escalation and evaluation need auditable links between a reply and its evidence.

**Alternative considered:** Accept free-form model text.

**Trade-off:** Strict parsing can reject otherwise readable responses, but malformed or unsupported replies fail visibly.

## D15 — Validate evaluation and reporting instead of inventing results

**Decision:** Build strict gates for golden labels, predictions, judge/human agreement, ablations, and failure analysis; prioritize labeling and evaluation before optimization.

**Reason:** The project's limiting factor is trustworthy evidence, not additional infrastructure.

**Alternative considered:** Fill missing results with heuristic, mock, or illustrative values.

**Trade-off:** Some later outputs remain blocked until humans provide labels and ratings, but the final report will be honest and reproducible.

## D16 — Use a reviewed development annotation pass for initial baselines

**Decision:** Label the 540-example development queue using the documented
taxonomy, primary-outcome precedence, and notes for ambiguity or risk.

**Reason:** The baselines need actual development labels, and the queue was
reviewed by reasoning over message text rather than copying sampling buckets.

**Alternative considered:** Keep all labels blank or treat keyword buckets as
gold labels.

**Trade-off:** The first pass enables real experiments, but it is not a
substitute for second-person adjudication or the locked golden evaluation set.
