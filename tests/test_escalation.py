import unittest

from src.escalation import evaluate_escalation


class EscalationTests(unittest.TestCase):
    def test_legal_threat_escalates(self) -> None:
        result = evaluate_escalation("I will sue if this is not fixed.", 0.99, [0.95])
        self.assertEqual(result.decision, "ESCALATE")
        self.assertEqual(result.matched_rule, "legal_threat")

    def test_security_issue_escalates_before_confidence(self) -> None:
        result = evaluate_escalation("My account was hacked.", 0.99, [0.99])
        self.assertEqual(result.matched_rule, "security_risk")

    def test_low_confidence_escalates(self) -> None:
        result = evaluate_escalation("Where is my order?", 0.40, [0.90])
        self.assertEqual(result.matched_rule, "low_intent_confidence")

    def test_missing_or_weak_evidence_escalates(self) -> None:
        self.assertEqual(evaluate_escalation("Need help", 0.9, []).matched_rule, "no_evidence")
        self.assertEqual(evaluate_escalation("Need help", 0.9, [0.2]).matched_rule, "weak_evidence")

    def test_good_low_risk_case_auto_handles(self) -> None:
        result = evaluate_escalation("Where is my delivery?", 0.90, [0.88, 0.60])
        self.assertEqual(result.decision, "AUTO_HANDLE")
        self.assertIsNone(result.escalation_reason)


if __name__ == "__main__":
    unittest.main()