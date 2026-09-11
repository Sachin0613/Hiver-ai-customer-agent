"""TF-IDF plus Logistic Regression intent baseline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.majority_baseline import require_labels


@dataclass
class TfidfBaseline:
    pipeline: Pipeline
    labels: list[str]

    def predict(self, messages: pd.Series) -> pd.Series:
        return pd.Series(self.pipeline.predict(messages.fillna("")), index=messages.index)


def split_labeled_data(
    frame: pd.DataFrame,
    random_seed: int,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = require_labels(frame)
    counts = labeled["intent_label"].value_counts()
    if len(counts) < 2:
        raise ValueError("At least two intent classes are required for TF-IDF evaluation.")
    if counts.min() < 2:
        raise ValueError("Every intent needs at least two labeled examples for a stratified split.")
    train, validation = train_test_split(
        labeled,
        test_size=validation_fraction,
        random_state=random_seed,
        stratify=labeled["intent_label"],
    )
    return train, validation


def fit_tfidf_baseline(train: pd.DataFrame, random_seed: int) -> TfidfBaseline:
    labels = sorted(train["intent_label"].astype(str).unique())
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_seed)),
        ]
    )
    pipeline.fit(train["clean_text"].fillna(""), train["intent_label"].astype(str))
    return TfidfBaseline(pipeline=pipeline, labels=labels)


def evaluate_tfidf(model: TfidfBaseline, validation: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    actual = validation["intent_label"].astype(str)
    predicted = model.predict(validation["clean_text"])
    labels = sorted(set(actual) | set(predicted))
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "validation_examples": int(len(validation)),
        "per_intent": report,
    }
    matrix = pd.DataFrame(
        confusion_matrix(actual, predicted, labels=labels),
        index=labels,
        columns=labels,
    )
    predictions = validation[["annotation_id", "conversation_id", "tweet_id", "clean_text", "intent_label"]].copy()
    predictions["predicted_intent"] = predicted.to_numpy()
    return metrics, matrix, predictions