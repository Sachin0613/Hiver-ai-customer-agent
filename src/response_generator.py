"""Grounded support reply generation using an OpenAI-compatible API."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable


DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class Evidence:
    customer_message: str
    company_response: str
    similarity: float | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class GeneratedReply:
    reply: str
    evidence_used: list[int]
    uncertainty: str
    unsupported_claims: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = """You draft concise customer-support replies grounded only in the supplied historical evidence.
Return valid JSON with exactly these keys:
reply: string
evidence_used: array of integer evidence indexes
uncertainty: one of "low", "medium", "high"
unsupported_claims: array of strings

Rules:
- Address the customer's message directly and use a calm support tone.
- Do not invent policies, refund amounts, dates, tracking information, or actions.
- Do not claim that an action was completed unless the evidence supports that exact claim.
- Ask for information or state uncertainty when evidence is insufficient.
- Never repeat private information from evidence.
- If evidence conflicts or is insufficient, acknowledge that limitation in the reply.
"""


def format_evidence(evidence: Iterable[Evidence]) -> str:
    lines = []
    for index, item in enumerate(evidence):
        score = "" if item.similarity is None else f" similarity={item.similarity:.3f}"
        lines.append(f"[{index}]{score}\nCUSTOMER: {item.customer_message}\nBRAND: {item.company_response}")
    return "\n\n".join(lines) if lines else "No historical evidence was retrieved."


def build_user_prompt(customer_message: str, predicted_intent: str, evidence: Iterable[Evidence], escalation_context: str | None = None) -> str:
    return (
        f"Customer message:\n{customer_message}\n\n"
        f"Predicted intent:\n{predicted_intent}\n\n"
        f"Escalation context:\n{escalation_context or 'No escalation context was provided.'}\n\n"
        f"Historical evidence:\n{format_evidence(evidence)}"
    )


def parse_response(payload: str) -> GeneratedReply:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("LLM response was not valid JSON.") from error
    required = {"reply", "evidence_used", "uncertainty", "unsupported_claims"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"LLM response is missing keys: {sorted(missing)}")
    if data["uncertainty"] not in {"low", "medium", "high"}:
        raise ValueError("uncertainty must be low, medium, or high")
    if not isinstance(data["reply"], str) or not data["reply"].strip():
        raise ValueError("reply must be a non-empty string")
    if not isinstance(data["evidence_used"], list) or not all(isinstance(index, int) and index >= 0 for index in data["evidence_used"]):
        raise ValueError("evidence_used must contain non-negative integer indexes")
    if not isinstance(data["unsupported_claims"], list):
        raise ValueError("unsupported_claims must be an array")
    return GeneratedReply(
        reply=data["reply"].strip(),
        evidence_used=data["evidence_used"],
        uncertainty=data["uncertainty"],
        unsupported_claims=[str(item) for item in data["unsupported_claims"]],
    )


def generate_reply(customer_message: str, predicted_intent: str, evidence: list[Evidence], escalation_context: str | None = None, model: str = DEFAULT_MODEL) -> GeneratedReply:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate a grounded reply.")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the openai package before using response generation.") from error
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(customer_message, predicted_intent, evidence, escalation_context)},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned an empty response.")
    result = parse_response(content)
    if any(index >= len(evidence) for index in result.evidence_used):
        raise ValueError("LLM referenced an evidence index that was not provided.")
    return result