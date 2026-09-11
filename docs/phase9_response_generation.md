# Phase 9 — Grounded Response Generation

`src/response_generator.py` accepts a customer message, predicted intent,
retrieved historical examples, and optional escalation context. It asks an
OpenAI-compatible model for structured JSON containing:

- `reply`
- `evidence_used`
- `uncertainty`
- `unsupported_claims`

The prompt prohibits invented policies, refund amounts, dates, tracking details,
completed actions, and private-information repetition. The parser validates the
JSON shape and rejects evidence indexes that were not provided.

No API call is made by the test suite. Put `OPENAI_API_KEY=<your-key>` in a
local `.env` file, or set it in the shell environment, before calling
`generate_reply`. The runtime loads `.env` with `python-dotenv`. Never commit
the value. The repository's `.env.example` contains only an empty placeholder.

This phase does not add escalation decisions or a complete agent pipeline.