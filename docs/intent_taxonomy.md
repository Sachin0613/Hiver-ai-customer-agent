# AmazonHelp Intent Taxonomy

## Status and evidence

This is a **provisional Phase 5 taxonomy** derived from the AmazonHelp
development split only (`57,789` conversations and `143,133` customer
messages). The locked golden pool was not read or used. Recurring development
terms included `order`, `delivery`, `prime`, `delivered`, `package`, `refund`,
`product`, `account`, `email`, and `credit card`.

The taxonomy is intentionally small enough to explain and annotate. A first-pass
manual review has now labeled all 540 rows in
`data/processed/intent_annotation_queue.csv`. The labels are suitable for
development baselines, but a second-person adjudication pass is recommended for
ambiguous and multi-intent cases. Sampling buckets were not copied as labels.

## Intents

### `delivery_tracking`

**Description:** Tracking, shipping progress, delivery dates, delays, or a
package marked delivered.

**Include:** Where-is-my-order questions, late delivery, tracking updates,
delivery-date changes, missing packages, and false delivery confirmation.

**Exclude:** A refund/return request where delivery is only background, or a
generic order cancellation without a shipping question.

**Examples from development themes:** “delivery date”, “package”, “tracking”,
“delivered”, “late”.

**Ambiguous:** A missing package may also lead to a refund or replacement. Pick
the customer's primary requested outcome and note the secondary issue.

### `order_issue`

**Description:** Order status, cancellation, incorrect/missing item, or an
order-management problem not primarily about delivery tracking.

**Include:** Cancel an order, change an order, wrong item, missing item, and
questions about an order that do not ask about shipment progress.

**Exclude:** Pure delivery tracking, payment-only problems, and returns after
the customer has received the item.

**Examples from development themes:** “order”, “ordered”, “cancel”, “item”.

**Ambiguous:** If the central complaint is a late or missing shipment, use
`delivery_tracking`.

### `returns_refunds`

**Description:** Returns, replacements, refunds, reimbursements, or refund
timing and amount.

**Include:** How to return an item, refund not received, refund status, and
requesting money back.

**Exclude:** A payment authorization or unexpected charge with no refund
request.

**Examples from development themes:** “refund”, “return”, “money back”.

**Ambiguous:** A defective product belongs here only when the requested action
is return/refund; otherwise use `product_or_listing`.

### `payment_billing`

**Description:** Payment methods, charges, invoices, prices, or billing issues.

**Include:** Credit/debit card failures, duplicate or unexpected charges,
payment authorization, invoice questions, and price discrepancies.

**Exclude:** A refund whose main request is money back, which belongs to
`returns_refunds`.

**Examples from development themes:** “credit card”, “payment”, “charged”,
“price”.

**Ambiguous:** A price change may be an order issue or billing issue; label the
customer's requested resolution.

### `prime_membership`

**Description:** Prime membership, subscription benefits, renewal, trial, or
Prime delivery entitlement.

**Include:** Prime membership status, cancellation/renewal, trial questions,
and missing Prime benefits.

**Exclude:** A normal order problem with no Prime membership component.

**Examples from development themes:** “Prime”, “prime membership”, “trial”.

**Ambiguous:** Prime delivery delay remains `delivery_tracking` if membership
is only context.

### `account_access`

**Description:** Login, email, password, account security, or account
information problems.

**Include:** Cannot sign in, password reset, locked account, changed email, and
account access/security concerns.

**Exclude:** An email notification question that does not affect account access.

**Examples from development themes:** “account”, “email”, “login”.

**Ambiguous:** Possible account takeover or sensitive security concerns should
also be marked for escalation later; intent remains `account_access`.

### `product_or_listing`

**Description:** Product quality, compatibility, listing information, warranty,
or product-specific questions.

**Include:** Defective/broken product, product information, compatibility,
warranty, and listing details.

**Exclude:** A seller-service complaint without a product question, or a return
whose primary request is a refund.

**Examples from development themes:** “product”, “device”, “broken”, “warranty”.

**Ambiguous:** Marketplace seller complaints can use `seller_or_customer_service`
when the seller interaction is the main issue.

### `seller_or_customer_service`

**Description:** Seller interaction, marketplace support, customer-service
experience, or service complaint.

**Include:** Seller communication, seller behavior, poor customer service, and
complaints about support handling.

**Exclude:** A complaint with a clearly primary delivery, payment, refund, or
product issue; use that more specific intent.

**Examples from development themes:** “customer service”, “customer care”,
“seller”, “complaint”.

**Ambiguous:** Emotional language alone does not determine the intent; identify
the operational request first.

### `feedback_or_other`

**Description:** Praise, general feedback, broad questions, multilingual or
unclear messages, and support requests outside the other categories.

**Include:** Product praise, general comments, unclear short messages, and
messages with insufficient information for a more specific label.

**Exclude:** Any message with enough evidence for one of the specific intents.

**Examples from development themes:** “great”, “question”, and messages without
a recoverable support request.

**Ambiguous:** Use this conservatively; unclear intent is a likely escalation
case later, not permission to overuse `other`.

## Annotation rules

1. Label the customer's primary requested outcome, not every keyword present.
2. Record secondary intents and uncertainty in `annotator_notes`.
3. Use `feedback_or_other` only when no specific intent is supported.
4. Do not use model predictions or the golden pool during annotation.
5. Security, legal, and highly sensitive cases still receive an intent label;
   escalation is a separate later decision.