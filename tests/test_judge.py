import unittest

from evaluation.judge import Evidence, build_judge_prompt, parse_judge_response


class JudgeTests(unittest.TestCase):
    def test_prompt_contains_only_evaluation_inputs(self) -> None:
        prompt = build_judge_prompt(
            "Where is my order?",
            [Evidence("Where is my order?", "Please check tracking.")],
            "Please check the tracking link.",
        )
        self.assertIn("Generated reply to evaluate", prompt)
        self.assertNotIn("predicted_intent", prompt)

    def test_valid_scores_are_parsed(self) -> None:
        payload = '{"relevance":5,"groundedness":4,"helpfulness":4,"tone":5,"unsupported_claims":5,"reasoning":"Supported and concise."}'
        result = parse_judge_response(payload)
        self.assertEqual(result.groundedness, 4)

    def test_out_of_range_score_is_rejected(self) -> None:
        payload = '{"relevance":6,"groundedness":4,"helpfulness":4,"tone":5,"unsupported_claims":5,"reasoning":"Bad score."}'
        with self.assertRaises(ValueError):
            parse_judge_response(payload)


if __name__ == "__main__":
    unittest.main()