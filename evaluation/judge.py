"""LLM-as-judge for generated support replies."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable

from src.response_generator import Evidence, format_evidence


DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
CRITERIA = ("relevance", "groundedness", "helpfulness", "tone", "unsupported_claims")


@dataclass(frozen=True)
class JudgeResult:
    relevance: int
    groundedness: int
    helpfulness: int
    tone: int
    unsupported_claims: int
    reasoning: str

    def to_dict(self) -> dict:
        return asdict(self)


JUDGE_PROMPT = """You are evaluating one customer-support reply. Score each criterion from 1 to 5.
Return JSON with exactly these keys: relevance, groundedness, helpfulness, tone,
unsupported_claims, reasoning.

Scoring:
- relevance: 5 directly addresses the customer's request; 1 ignores it.
- groundedness: 5 is fully supported by the supplied evidence; 1 contradicts or invents facts.
- helpfulness: 5 gives a useful, safe next step; 1 gives no useful help.
- tone: 5 is concise, respectful, and appropriate for support; 1 is hostile or inappropriate.
- unsupported_claims: 5 makes no unsupported claims; 1 contains serious fabricated claims.

Use only the customer message and supplied historical evidence. Do not use any
system-generated scores or predictions. Explain the main reasons briefly.
"""


def build_judge_prompt(customer_message: str, evidence: Iterable[Evidence], generated_reply: str) -> str:
    return (
        f"Customer message:\n{customer_message}\n\n"
        f"Historical evidence:\n{format_evidence(evidence)}\n\n"
        f"Generated reply to evaluate:\n{generated_reply}"
    )


def parse_judge_response(payload: str) -> JudgeResult:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("Judge response was not valid JSON.") from error
    missing = (set(CRITERIA) | {"reasoning"}) - set(data)
    if missing:
        raise ValueError(f"Judge response is missing keys: {sorted(missing)}")
    scores = {}
    for criterion in CRITERIA:
        score = data[criterion]
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{criterion} must be an integer from 1 to 5")
        scores[criterion] = score
    if not isinstance(data["reasoning"], str) or not data["reasoning"].strip():
        raise ValueError("reasoning must be a non-empty string")
    return JudgeResult(**scores, reasoning=data["reasoning"].strip())


def judge_reply(
    customer_message: str,
    evidence: list[Evidence],
    generated_reply: str,
    model: str = DEFAULT_JUDGE_MODEL,
) -> JudgeResult:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the LLM judge.")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the openai package before using the LLM judge.") from error
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": build_judge_prompt(customer_message, evidence, generated_reply)},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Judge returned an empty response.")
    return parse_judge_response(content)