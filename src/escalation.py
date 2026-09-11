"""Deterministic escalation rules for customer-support messages."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class EscalationConfig:
    min_intent_confidence: float = 0.65
    min_evidence_score: float = 0.45
    max_evidence_score_spread: float = 0.12


@dataclass(frozen=True)
class EscalationDecision:
    decision: str
    escalation_reason: str | None
    matched_rule: str | None

    def to_dict(self) -> dict:
        return asdict(self)


RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("legal_threat", "Legal threat or regulatory complaint requires human review.", re.compile(r"\b(sue|lawsuit|lawyer|attorney|legal action|court|regulator|small claims)\b", re.I)),
    ("security_risk", "Possible account takeover or security incident requires human review.", re.compile(r"\b(hacked|hack|account takeover|stolen password|unauthorized access|identity theft|someone accessed)\b", re.I)),
    ("sensitive_information", "Sensitive personal or financial information requires human review.", re.compile(r"\b(full ssn|social security|bank account|routing number|credit card number|password is|one[- ]time code|otp)\b", re.I)),
    ("severe_complaint", "Severe unresolved complaint requires human review.", re.compile(r"\b(class action|fraud|scam|report you|never resolved|extremely dangerous|injured)\b", re.I)),
    ("high_risk_financial", "High-risk financial issue requires human review.", re.compile(r"\b(unknown charge|fraudulent charge|charged twice|financial loss|money stolen)\b", re.I)),
)


def _escalate(reason: str, rule: str) -> EscalationDecision:
    return EscalationDecision("ESCALATE", reason, rule)


def evaluate_escalation(
    customer_message: str,
    intent_confidence: float,
    evidence_scores: Iterable[float],
    config: EscalationConfig | None = None,
) -> EscalationDecision:
    """Apply high-risk rules before allowing automatic handling."""
    settings = config or EscalationConfig()
    message = customer_message.strip()
    for rule, reason, pattern in RULES:
        if pattern.search(message):
            return _escalate(reason, rule)

    if not 0 <= intent_confidence <= 1:
        raise ValueError("intent_confidence must be between 0 and 1")
    if intent_confidence < settings.min_intent_confidence:
        return _escalate("Intent confidence is below the automatic-handling threshold.", "low_intent_confidence")

    scores = [float(score) for score in evidence_scores]
    if not scores:
        return _escalate("No historical evidence was retrieved.", "no_evidence")
    if any(score < -1 or score > 1 for score in scores):
        raise ValueError("Evidence scores must be between -1 and 1.")
    best = max(scores)
    if best < settings.min_evidence_score:
        return _escalate("Historical evidence is below the evidence threshold.", "weak_evidence")
    if len(scores) > 1 and best - sorted(scores)[-2] < settings.max_evidence_score_spread:
        return _escalate("Historical evidence has conflicting top matches.", "conflicting_evidence")
    return EscalationDecision("AUTO_HANDLE", None, None)