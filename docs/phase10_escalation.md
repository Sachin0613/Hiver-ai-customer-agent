# Phase 10 — Escalation Logic

Escalation is deterministic and runs before automatic reply handling. It
escalates messages containing legal threats, possible account takeover or
security incidents, sensitive personal/financial information, severe complaints,
or high-risk financial language.

It also escalates when:

- intent confidence is below `0.65`;
- no historical evidence is retrieved;
- the best evidence score is below `0.45`; or
- the top evidence matches are too close, indicating possible conflict.

The output is either `AUTO_HANDLE` or `ESCALATE`, with an explicit reason and
matched rule. Thresholds are configuration values and must be tuned only with
development data after intent labels exist. The golden pool is not used.