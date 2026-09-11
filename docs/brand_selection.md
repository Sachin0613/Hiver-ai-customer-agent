# Brand Selection

## Dataset

Source file: `data\raw\twcs.csv` (2,811,774 tweets in the slim pass).

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
| Volume | 20% | log1p(resolved conversations), min-max among candidates |
| Resolution | 25% | response_rate |
| Quality | 20% | median length near 4 turns + multi-turn resolved share |
| Diversity | 15% | ngram_diversity, min-max among candidates |
| Response usefulness | 20% | useful_reply_rate |

## Top Brands

Sorted by resolved conversations (more useful for retrieval than raw tweet count).

| brand | total_tweets | customer_messages | company_responses | conversation_count | resolved_conversations | avg_conversation_length | median_conversation_length | response_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AmazonHelp | 374,042 | 203,830 | 169,840 | 82,556 | 82,556 | 4.531 | 3.000 | 0.760 |
| AppleSupport | 238,907 | 131,881 | 106,860 | 80,717 | 80,717 | 2.960 | 2.000 | 0.808 |
| Uber_Support | 128,550 | 72,204 | 56,270 | 41,923 | 41,923 | 3.066 | 2.000 | 0.764 |
| SpotifyCares | 91,889 | 48,589 | 43,265 | 28,280 | 28,277 | 3.249 | 2.000 | 0.856 |
| AmericanAir | 87,584 | 50,420 | 36,764 | 26,386 | 26,386 | 3.319 | 2.000 | 0.723 |
| Delta | 87,994 | 45,505 | 42,253 | 26,168 | 26,166 | 3.363 | 2.000 | 0.794 |
| comcastcares | 73,049 | 39,769 | 33,031 | 24,063 | 24,063 | 3.036 | 2.000 | 0.764 |
| TMobileHelp | 82,730 | 47,396 | 34,317 | 22,820 | 22,820 | 3.625 | 2.000 | 0.714 |
| SouthwestAir | 64,726 | 35,581 | 28,977 | 21,636 | 21,634 | 2.992 | 2.000 | 0.795 |
| Ask_Spectrum | 59,638 | 33,366 | 25,860 | 18,532 | 18,531 | 3.218 | 2.000 | 0.749 |
| Tesco | 73,159 | 34,376 | 38,573 | 16,722 | 16,722 | 4.375 | 4.000 | 0.735 |
| British_Airways | 60,716 | 31,275 | 29,361 | 16,452 | 16,452 | 3.690 | 3.000 | 0.770 |
| UPSHelp | 44,552 | 26,021 | 17,817 | 15,523 | 15,523 | 2.870 | 2.000 | 0.676 |
| hulu_support | 49,295 | 26,878 | 21,872 | 14,955 | 14,952 | 3.296 | 2.000 | 0.799 |
| VirginTrains | 65,810 | 37,878 | 27,817 | 14,853 | 14,851 | 4.431 | 3.000 | 0.694 |
| ChipotleTweets | 41,840 | 23,089 | 18,749 | 14,392 | 14,392 | 2.907 | 2.000 | 0.804 |
| sprintcare | 54,051 | 30,457 | 22,381 | 13,560 | 13,560 | 3.986 | 2.000 | 0.658 |
| XboxSupport | 57,657 | 32,455 | 24,557 | 13,455 | 13,435 | 4.285 | 3.000 | 0.623 |
| AskPlayStation | 43,136 | 23,963 | 19,098 | 12,533 | 12,533 | 3.442 | 2.000 | 0.778 |
| AskTarget | 30,645 | 17,407 | 13,218 | 11,151 | 11,151 | 2.748 | 2.000 | 0.755 |

## Candidate Brands

Candidates are the top 5 brands by `resolved_conversations * response_rate`
among brands with at least the configured minimum resolved conversations.

`AppleSupport`, `AmazonHelp`, `Uber_Support`, `SpotifyCares`, `Delta`

Why they were considered: each has enough resolved threads to support intent
labels, retrieval, and a ~200-example golden set. Final ranking uses the score
table, not volume alone.

| brand | volume_score | resolution_score | quality_score | diversity_score | response_usefulness_score | overall_score |
| --- | --- | --- | --- | --- | --- | --- |
| AmazonHelp | 1.000 | 0.760 | 0.685 | 0.000 | 0.978 | 0.723 |
| AppleSupport | 0.980 | 0.808 | 0.424 | 0.377 | 0.835 | 0.706 |
| Uber_Support | 0.410 | 0.764 | 0.430 | 1.000 | 0.739 | 0.657 |
| SpotifyCares | 0.068 | 0.856 | 0.436 | 0.112 | 0.911 | 0.513 |
| Delta | 0.000 | 0.794 | 0.463 | 0.211 | 0.857 | 0.494 |

## Preliminary themes (customer-message bigrams)

These are **not** the final intent taxonomy. They are the most common content
bigrams in a seeded sample of inbound messages.

### AppleSupport

| theme_bigram | count |
| --- | --- |
| applesupport_iphone | 133 |
| ios_update | 89 |
| iphone_plus | 72 |
| applesupport_fix | 69 |
| applesupport_yes | 66 |
| applesupport_ios | 64 |
| every_time | 59 |
| new_update | 53 |
| question_mark | 48 |
| battery_life | 48 |
| ever_since | 47 |
| applesupport_phone | 45 |
### AmazonHelp

| theme_bigram | count |
| --- | --- |
| customer_service | 87 |
| amazonhelp_yes | 64 |
| amazon_prime | 60 |
| amazonhelp_amazon | 48 |
| amazonhelp_already | 47 |
| customer_care | 46 |
| next_day | 41 |
| day_delivery | 37 |
| day_shipping | 36 |
| delivery_date | 31 |
| amazonhelp_order | 30 |
| amazonhelp_done | 28 |
### Uber_Support

| theme_bigram | count |
| --- | --- |
| uber_support | 2661 |
| support_uber | 91 |
| customer_service | 90 |
| uber_driver | 86 |
| support_sent | 79 |
| support_already | 61 |
| support_email | 59 |
| last_night | 57 |
| uber_eats | 55 |
| support_driver | 50 |
| support_need | 46 |
| phone_number | 42 |
### SpotifyCares

| theme_bigram | count |
| --- | --- |
| spotifycares_spotify | 57 |
| spotify_premium | 54 |
| premium_account | 49 |
| spotifycares_there | 48 |
| family_plan | 42 |
| web_player | 40 |
| spotify_account | 38 |
| spotifycares_tried | 34 |
| spotifycares_don | 34 |
| student_discount | 34 |
| there_way | 34 |
| need_help | 33 |
### Delta

| theme_bigram | count |
| --- | --- |
| delta_flight | 176 |
| customer_service | 112 |
| delta_delta | 93 |
| first_class | 52 |
| delta_there | 51 |
| flight_delayed | 43 |
| flight_attendant | 42 |
| gate_agent | 42 |
| fly_delta | 42 |
| delta_sent | 41 |
| delta_yes | 41 |
| delta_great | 40 |


## Final Recommendation

**AmazonHelp**

## Why This Brand

- Resolved conversations: **82,556**
- Customer messages in its threads: **203,830**
- Company responses: **169,840**
- Direct response rate: **76.0%**
- Overall score among candidates: **0.723**
- Useful-reply rate (sample): **97.8%**
- DM-template rate (sample): **0.5%**

It outranked `AppleSupport`, `Uber_Support`, `SpotifyCares`, `Delta` on the
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

## Golden-set feasibility (AmazonHelp)

- Available resolved conversations: 82,556
- Hand-label target: ~200
- Feasible from volume: yes
- Common intents: take the highest-count bigrams as seeds, then label manually
- Rare / ambiguous / multi-intent: oversample short unclear tweets and tweets
  whose bigrams hit two themes

## Sampling Strategy

Unit = **conversation** (`conv_id`), never individual tweets. Seed = **42**.

1. Keep conversations where `AmazonHelp` is the only company author.
2. Hash `conv_id` with seed 42.
3. Split conversations: **70% retrieval corpus**, **20% development**, **10% held-out**.
4. Draw the golden set (~200) **only** from held-out conversations so retrieval
   cannot see evaluation threads.
5. After Phase 5 intents exist, stratify the golden set by intent, plus buckets
   for ambiguous and multi-intent messages.
6. Do not put golden-set tweet ids into the FAISS index.

Development may include unlabeled retrieval threads for debugging, but evaluation
metrics for the report must use the golden set only.
