from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import make_target


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    task: str
    balanced_accuracy: float
    macro_f1: float
    labels: list[str]
    confusion_matrix: np.ndarray
    report: dict
    estimator: Pipeline


def build_pipeline(task: str, random_state: int = 42) -> Pipeline:
    """Build a model whose learned preprocessing is confined to training data."""
    preprocessing = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
    ]
    if task == "binary":
        return Pipeline(
            preprocessing
            + [
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2_000, random_state=random_state
                    ),
                )
            ]
        )
    if task == "multiclass":
        return Pipeline(
            preprocessing
            + [
                ("smote", SMOTE(random_state=random_state, k_neighbors=2)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError("task must be 'binary' or 'multiclass'.")


def train_and_evaluate(
    frame: pd.DataFrame,
    task: str = "binary",
    test_size: float = 0.25,
    random_state: int = 42,
) -> EvaluationResult:
    """Split first, fit a leakage-safe pipeline, and calculate imbalance-aware metrics."""
    if "Label" not in frame:
        raise ValueError("The frame must contain a 'Label' column.")
    features = frame.drop(columns="Label")
    target = make_target(frame["Label"], task)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    estimator = build_pipeline(task, random_state)
    estimator.fit(x_train, y_train)
    prediction = estimator.predict(x_test)
    labels = sorted(target.unique())
    return EvaluationResult(
        task=task,
        balanced_accuracy=float(balanced_accuracy_score(y_test, prediction)),
        macro_f1=float(f1_score(y_test, prediction, average="macro")),
        labels=labels,
        confusion_matrix=confusion_matrix(y_test, prediction, labels=labels),
        report=classification_report(y_test, prediction, labels=labels, output_dict=True),
        estimator=estimator,
    )
