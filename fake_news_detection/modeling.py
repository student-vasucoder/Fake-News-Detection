from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("Data/fake_or_real_news.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.txt"


@dataclass(frozen=True)
class TrainingResult:
    accuracy: float
    report: str
    confusion: list[list[int]]
    train_size: int
    test_size: int


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split())


def combine_text(title: str, text: str) -> str:
    title_part = _normalize_text(title)
    text_part = _normalize_text(text)
    return f"{title_part}. {text_part}".strip()


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.rename(columns={df.columns[0]: "row_id"})
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["combined_text"] = [
        combine_text(title, text) for title, text in zip(df["title"], df["text"])
    ]
    return df[["row_id", "title", "text", "combined_text", "label"]]


def build_pipeline(min_df: int = 2, max_df: float = 0.85) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=max_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(data_path: Path = DATA_PATH) -> tuple[Pipeline, TrainingResult]:
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["combined_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4)
    confusion = confusion_matrix(y_test, predictions, labels=["FAKE", "REAL"]).tolist()

    result = TrainingResult(
        accuracy=accuracy,
        report=report,
        confusion=confusion,
        train_size=len(X_train),
        test_size=len(X_test),
    )
    return pipeline, result


def save_artifacts(model: Pipeline, result: TrainingResult) -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics_text = "\n".join(
        [
            "Fake News Detection Metrics",
            "===========================",
            f"Accuracy: {result.accuracy:.4f}",
            f"Train size: {result.train_size}",
            f"Test size: {result.test_size}",
            "Confusion matrix [FAKE, REAL]:",
            str(result.confusion),
            "",
            "Classification report:",
            result.report,
        ]
    )
    METRICS_PATH.write_text(metrics_text, encoding="utf-8")


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(model_path)


def predict_news(
    model: Pipeline,
    *,
    title: str,
    text: str,
) -> dict[str, object]:
    payload = [combine_text(title, text)]
    prediction = model.predict(payload)[0]

    probability_map = None
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(payload)[0]
        labels: Iterable[str] = model.classes_
        probability_map = {
            label: round(float(score), 4) for label, score in zip(labels, probabilities)
        }
        confidence = probability_map[prediction]

    return {
        "label": prediction,
        "confidence": confidence,
        "probabilities": probability_map,
    }
