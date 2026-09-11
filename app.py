from pathlib import Path
import sys
import threading
import webbrowser

import pandas as pd
import faiss
from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tfidf_baseline import fit_tfidf_baseline
from src.retrieval import HistoricalRetriever
from src.escalation import evaluate_escalation
from src.response_generator import Evidence, generate_reply


app = Flask(__name__)


# ============================================================
# LOAD INTENT CLASSIFIER
# ============================================================

development_path = (
    ROOT / "data" / "processed" / "intent_annotation_queue.csv"
)

if not development_path.exists():
    raise FileNotFoundError(
        f"Development data not found: {development_path}"
    )

development = pd.read_csv(development_path)

classifier = fit_tfidf_baseline(
    development,
    random_seed=42,
)


# ============================================================
# LOAD FAISS RETRIEVAL INDEX
# ============================================================

index_path = ROOT / "data" / "indexes" / "retrieval.faiss"
metadata_path = ROOT / "data" / "indexes" / "retrieval_metadata.csv"

if not index_path.exists():
    raise FileNotFoundError(
        f"FAISS index not found: {index_path}"
    )

if not metadata_path.exists():
    raise FileNotFoundError(
        f"Retrieval metadata not found: {metadata_path}"
    )

index = faiss.read_index(str(index_path))
metadata = pd.read_csv(metadata_path)

retriever = HistoricalRetriever(
    index,
    metadata,
)


# ============================================================
# WEB PAGE
# ============================================================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Hiver AI Customer Support Agent</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Arial,
                sans-serif;
            background: #f5f7fa;
            color: #172033;
        }

        .container {
            width: 100%;
            max-width: 950px;
            margin: 0 auto;
            padding: 32px 18px 60px;
        }

        .header {
            margin-bottom: 24px;
        }

        h1 {
            margin: 0 0 8px;
            font-size: 30px;
            line-height: 1.2;
        }

        h2 {
            margin-top: 0;
            font-size: 20px;
        }

        .subtitle {
            color: #667085;
            line-height: 1.5;
        }

        .card {
            background: #ffffff;
            border: 1px solid #e4e7ec;
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }

        textarea {
            display: block;
            width: 100%;
            min-height: 140px;
            padding: 14px;
            border: 1px solid #d0d5dd;
            border-radius: 10px;
            font-family: inherit;
            font-size: 16px;
            line-height: 1.5;
            resize: vertical;
            outline: none;
        }

        textarea:focus {
            border-color: #667085;
        }

        button {
            width: auto;
            margin-top: 12px;
            padding: 12px 20px;
            border: none;
            border-radius: 9px;
            background: #111827;
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
        }

        button:hover {
            opacity: 0.9;
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }

        .metric {
            background: #f8fafc;
            border-radius: 10px;
            padding: 14px;
            min-width: 0;
        }

        .label {
            color: #667085;
            font-size: 13px;
            margin-bottom: 6px;
        }

        .value {
            font-size: 18px;
            font-weight: 600;
            word-break: break-word;
        }

        .decision {
            font-weight: 700;
        }

        .reply {
            background: #f8fafc;
            border-radius: 10px;
            padding: 16px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .evidence {
            border-top: 1px solid #eaecf0;
            padding: 16px 0;
            line-height: 1.5;
        }

        .evidence:first-child {
            border-top: none;
        }

        .similarity {
            color: #667085;
            font-size: 13px;
            margin-top: 4px;
        }

        .reason {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 10px;
            padding: 14px;
            margin-top: 14px;
        }

        .warning {
            background: #fffaeb;
            border: 1px solid #fedf89;
            border-radius: 10px;
            padding: 14px;
            margin-top: 14px;
        }

        .success {
            background: #ecfdf3;
            border: 1px solid #abefc6;
            border-radius: 10px;
            padding: 14px;
            margin-top: 14px;
        }

        .error {
            color: #b42318;
            background: #fef3f2;
            border: 1px solid #fecdca;
            padding: 14px;
            border-radius: 10px;
        }

        .small {
            color: #667085;
            font-size: 13px;
        }

        ul {
            padding-left: 20px;
        }

        @media (max-width: 600px) {
            .container {
                padding: 20px 12px 40px;
            }

            h1 {
                font-size: 25px;
            }

            .card {
                padding: 16px;
                border-radius: 12px;
            }

            .grid {
                grid-template-columns: 1fr;
            }

            button {
                width: 100%;
            }

            textarea {
                min-height: 120px;
                font-size: 16px;
            }
        }
    </style>
</head>

<body>

<div class="container">

    <div class="header">
        <h1>Hiver AI Customer Support Agent</h1>

        <div class="subtitle">
            Test customer-support messages against historical AmazonHelp conversations.
        </div>
    </div>

    <div class="card">
        <h2>Customer Message</h2>

        <textarea
            id="message"
            placeholder="Example: My order was supposed to arrive yesterday."
        ></textarea>

        <button id="send" onclick="sendMessage()">
            Send Message
        </button>
    </div>

    <div id="result"></div>

</div>

<script>

async function sendMessage() {

    const message =
        document.getElementById("message").value.trim();

    const button =
        document.getElementById("send");

    const result =
        document.getElementById("result");

    if (!message) {
        result.innerHTML =
            '<div class="card error">Please enter a customer message.</div>';
        return;
    }

    button.disabled = true;
    button.textContent = "Analyzing...";

    result.innerHTML =
        '<div class="card">Analyzing customer message...</div>';

    try {

        const response = await fetch(
            "/api/chat",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    message: message
                })
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Request failed."
            );
        }

        let evidenceHTML = "";

        for (let i = 0; i < data.evidence.length; i++) {

            const item = data.evidence[i];

            evidenceHTML += `
                <div class="evidence">

                    <strong>
                        Evidence ${i + 1}
                    </strong>

                    <div class="similarity">
                        Similarity:
                        ${Number(item.similarity).toFixed(3)}
                    </div>

                    <p>
                        <strong>Customer:</strong>
                        ${escapeHtml(item.customer_message)}
                    </p>

                    <p>
                        <strong>AmazonHelp:</strong>
                        ${escapeHtml(item.company_response)}
                    </p>

                </div>
            `;
        }

        let replyHTML = "";

        if (data.reply) {

            replyHTML = `
                <div class="card">

                    <h2>AI Suggested Reply</h2>

                    <div class="reply">
                        ${escapeHtml(data.reply)}
                    </div>

                    <p class="small">
                        Uncertainty:
                        ${escapeHtml(data.uncertainty || "unknown")}
                    </p>

                </div>
            `;

        } else {

            replyHTML = `
                <div class="card">

                    <h2>AI Suggested Reply</h2>

                    <div class="warning">
                        Reply generation is unavailable.
                    </div>

                    ${
                        data.reply_error
                        ? `
                            <p class="small">
                                ${escapeHtml(data.reply_error)}
                            </p>
                          `
                        : ""
                    }

                </div>
            `;
        }

        let unsupportedHTML = "";

        if (
            data.unsupported_claims &&
            data.unsupported_claims.length > 0
        ) {

            unsupportedHTML = `
                <div class="warning">

                    <strong>Unsupported Claims</strong>

                    <ul>
                        ${data.unsupported_claims
                            .map(
                                claim =>
                                    `<li>${escapeHtml(claim)}</li>`
                            )
                            .join("")
                        }
                    </ul>

                </div>
            `;
        }

        result.innerHTML = `

            <div class="card">

                <h2>Agent Analysis</h2>

                <div class="grid">

                    <div class="metric">
                        <div class="label">Intent</div>

                        <div class="value">
                            ${escapeHtml(data.intent)}
                        </div>
                    </div>

                    <div class="metric">
                        <div class="label">Confidence</div>

                        <div class="value">
                            ${(Number(data.confidence) * 100).toFixed(1)}%
                        </div>
                    </div>

                    <div class="metric">
                        <div class="label">Decision</div>

                        <div class="value decision">
                            ${escapeHtml(data.decision)}
                        </div>
                    </div>

                    <div class="metric">
                        <div class="label">Escalation Rule</div>

                        <div class="value">
                            ${escapeHtml(data.matched_rule || "None")}
                        </div>
                    </div>

                </div>

                ${
                    data.escalation_reason
                    ? `
                        <div class="reason">

                            <strong>Reason</strong>

                            <p>
                                ${escapeHtml(data.escalation_reason)}
                            </p>

                        </div>
                      `
                    : ""
                }

            </div>

            ${replyHTML}

            ${unsupportedHTML}

            <div class="card">

                <h2>Historical Evidence</h2>

                ${
                    evidenceHTML ||
                    '<p class="small">No historical evidence found.</p>'
                }

            </div>
        `;

    } catch (error) {

        result.innerHTML = `
            <div class="card error">
                ${escapeHtml(error.message)}
            </div>
        `;

    } finally {

        button.disabled = false;
        button.textContent = "Send Message";

    }
}


function escapeHtml(text) {

    const div = document.createElement("div");

    div.textContent =
        text == null ? "" : String(text);

    return div.innerHTML;
}


document
    .getElementById("message")
    .addEventListener(
        "keydown",
        function(event) {

            if (
                (event.ctrlKey || event.metaKey) &&
                event.key === "Enter"
            ) {
                sendMessage();
            }

        }
    );

</script>

</body>
</html>
"""


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template_string(HTML)


# ============================================================
# CHAT API
# ============================================================

@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}

    message = str(
        data.get("message", "")
    ).strip()

    if not message:
        return jsonify({
            "error": "Customer message is required."
        }), 400

    try:

        # ----------------------------------------------------
        # 1. INTENT CLASSIFICATION
        # ----------------------------------------------------

        message_series = pd.Series([message])

        predicted_intent = classifier.pipeline.predict(
            message_series
        )[0]

        probabilities = classifier.pipeline.predict_proba(
            message_series
        )[0]

        confidence = float(probabilities.max())


        # ----------------------------------------------------
        # 2. HISTORICAL RETRIEVAL
        # ----------------------------------------------------

        evidence = retriever.search(
            message,
            top_k=5
        )

        evidence_scores = [
            float(item["similarity"])
            for item in evidence
        ]


        # ----------------------------------------------------
        # 3. ESCALATION DECISION
        # ----------------------------------------------------

        decision = evaluate_escalation(
            customer_message=message,
            intent_confidence=confidence,
            evidence_scores=evidence_scores,
        )


        # ----------------------------------------------------
        # 4. OPENAI GROUNDED REPLY
        # ----------------------------------------------------

        ai_reply = None
        uncertainty = "high"
        unsupported_claims = []
        reply_error = None

        try:

            evidence_objects = [
                Evidence(
                    customer_message=item["customer_message"],
                    company_response=item["company_response"],
                    similarity=float(item["similarity"]),
                    conversation_id=str(
                        item.get("conversation_id", "")
                    ),
                )
                for item in evidence
            ]

            generated = generate_reply(
                customer_message=message,
                predicted_intent=predicted_intent,
                evidence=evidence_objects,
                escalation_context=decision.escalation_reason,
            )

            ai_reply = generated.reply
            uncertainty = generated.uncertainty
            unsupported_claims = generated.unsupported_claims

        except Exception as error:

            reply_error = str(error)


        # ----------------------------------------------------
        # 5. RETURN COMPLETE RESULT
        # ----------------------------------------------------

        return jsonify({

            "message": message,

            "intent": predicted_intent,

            "confidence": confidence,

            "decision": decision.decision,

            "escalation_reason":
                decision.escalation_reason,

            "matched_rule":
                decision.matched_rule,

            "reply":
                ai_reply,

            "uncertainty":
                uncertainty,

            "unsupported_claims":
                unsupported_claims,

            "reply_error":
                reply_error,

            "evidence": [

                {
                    "customer_message":
                        item["customer_message"],

                    "company_response":
                        item["company_response"],

                    "similarity":
                        float(item["similarity"]),

                    "conversation_id":
                        str(
                            item.get(
                                "conversation_id",
                                ""
                            )
                        ),
                }

                for item in evidence

            ],

        })


    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    url = "http://127.0.0.1:5000"

    print("=" * 60)
    print("Hiver AI Customer Support Agent")
    print("=" * 60)
    print(f"Open: {url}")
    print("Press CTRL+C to stop.")
    print()

    threading.Timer(
        1.0,
        lambda: webbrowser.open(url)
    ).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
