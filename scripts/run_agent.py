"""Interactive terminal smoke test for the Hiver support agent."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tfidf_baseline import fit_tfidf_baseline
from src.retrieval import HistoricalRetriever
from src.escalation import evaluate_escalation


def main() -> None:
    print("=" * 60)
    print(" Hiver AI Customer Support Agent - Terminal Test")
    print("=" * 60)
    print()

    # ---------------------------------------------------------
    # 1. Load the labeled development data
    # ---------------------------------------------------------
    development_path = ROOT / "data" / "processed" / "intent_annotation_queue.csv"
    development = pd.read_csv(development_path)

    print(f"Loading intent classifier from {len(development):,} examples...")

    classifier = fit_tfidf_baseline(
        development,
        random_seed=42,
    )

    # ---------------------------------------------------------
    # 2. Load the FAISS retrieval index
    # ---------------------------------------------------------
    import faiss

    index_path = ROOT / "data" / "indexes" / "retrieval.faiss"
    metadata_path = ROOT / "data" / "indexes" / "retrieval_metadata.csv"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Retrieval metadata not found: {metadata_path}")

    index = faiss.read_index(str(index_path))
    metadata = pd.read_csv(metadata_path)

    retriever = HistoricalRetriever(index, metadata)

    print(f"Loaded FAISS index with {index.ntotal:,} historical pairs.")
    print()

    # ---------------------------------------------------------
    # 3. Interactive customer input
    # ---------------------------------------------------------
    print("Type a customer message.")
    print("Type 'exit' to quit.")
    print()

    while True:
        customer_message = input("Customer message: ").strip()

        if customer_message.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not customer_message:
            print("Please enter a customer message.")
            continue

        print()
        print("-" * 60)
        print("PROCESSING...")
        print("-" * 60)

        # -----------------------------------------------------
        # 4. Intent prediction + confidence
        # -----------------------------------------------------
        clean_message = pd.Series([customer_message])

        predicted_intent = classifier.pipeline.predict(clean_message)[0]

        probabilities = classifier.pipeline.predict_proba(clean_message)[0]
        confidence = float(probabilities.max())

        # -----------------------------------------------------
        # 5. Historical retrieval
        # -----------------------------------------------------
        evidence = retriever.search(
            customer_message,
            top_k=5,
        )

        evidence_scores = [
            float(item["similarity"])
            for item in evidence
        ]

        # -----------------------------------------------------
        # 6. Escalation decision
        # -----------------------------------------------------
        decision = evaluate_escalation(
            customer_message=customer_message,
            intent_confidence=confidence,
            evidence_scores=evidence_scores,
        )

        # -----------------------------------------------------
        # 7. Display result
        # -----------------------------------------------------
        print()
        print(f"Intent:       {predicted_intent}")
        print(f"Confidence:   {confidence:.3f}")
        print()

        print("Historical Evidence:")
        if not evidence:
            print("  No evidence found.")
        else:
            for number, item in enumerate(evidence, start=1):
                print()
                print(f"  [{number}] Similarity: {item['similarity']:.3f}")
                print(f"      Customer: {item['customer_message']}")
                print(f"      Brand:    {item['company_response']}")

        print()
        print(f"Decision:     {decision.decision}")

        if decision.escalation_reason:
            print(f"Reason:       {decision.escalation_reason}")

        if decision.matched_rule:
            print(f"Rule:         {decision.matched_rule}")

        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()