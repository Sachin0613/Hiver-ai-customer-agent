import unittest

from src.response_generator import Evidence, build_user_prompt, parse_response


class ResponseGeneratorTests(unittest.TestCase):
    def test_prompt_contains_customer_and_evidence(self) -> None:
        prompt = build_user_prompt("Where is my order?", "delivery_tracking", [Evidence("Where is my order?", "Please check tracking.", 0.9)])
        self.assertIn("Where is my order?", prompt)
        self.assertIn("Please check tracking", prompt)

    def test_structured_response_is_parsed(self) -> None:
        result = parse_response('{"reply":"Please share the order details.","evidence_used":[0],"uncertainty":"medium","unsupported_claims":[]}')
        self.assertEqual(result.evidence_used, [0])
        self.assertEqual(result.uncertainty, "medium")

    def test_invalid_evidence_indexes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_response('{"reply":"Hello","evidence_used":[-1],"uncertainty":"low","unsupported_claims":[]}')


if __name__ == "__main__":
    unittest.main()