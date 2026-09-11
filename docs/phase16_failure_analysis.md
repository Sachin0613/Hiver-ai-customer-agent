# Phase 16 — Failure Analysis

Failure analysis is generated only from real evaluation failures in
`evaluation/failure_cases.csv`. Each row must include:

- A unique ID
- Customer message
- Expected output
- Actual output
- Failure category
- Why it failed
- Hypothesis
- Potential fix

The analyzer ranks the top five categories and writes:

- `evaluation/failure_summary.csv`
- `docs/failure_analysis.md`

Run:

```powershell
python scripts/analyze_failures.py
```

No failure examples are currently fabricated. The command blocks until actual
golden evaluation failures are recorded.