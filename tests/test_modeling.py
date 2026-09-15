import numpy as np
import pandas as pd
import pytest

from cicids2017 import build_pipeline, train_and_evaluate


def example_frame(rows_per_class: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    labels = np.repeat(["BENIGN", "DDoS", "DoS Hulk", "PortScan"], rows_per_class)
    signal = (labels != "BENIGN").astype(float)
    return pd.DataFrame(
        {
            "duration": rng.normal(signal * 3, 1),
            "packets": rng.normal(signal * 4, 1),
            "bytes": rng.normal(signal * 2, 1),
            "missing": np.where(
                rng.random(len(labels)) < 0.1,
                np.nan,
                rng.normal(size=len(labels)),
            ),
            "Label": labels,
        }
    )


def test_binary_training_is_deterministic() -> None:
    frame = example_frame()
    first = train_and_evaluate(frame, "binary", random_state=9)
    second = train_and_evaluate(frame, "binary", random_state=9)
    assert first.balanced_accuracy == second.balanced_accuracy
    assert np.array_equal(first.confusion_matrix, second.confusion_matrix)
    assert first.macro_f1 > 0.8


def test_multiclass_smote_is_inside_pipeline() -> None:
    pipeline = build_pipeline("multiclass")
    assert list(pipeline.named_steps) == ["imputer", "scaler", "pca", "smote", "classifier"]
    result = train_and_evaluate(example_frame(), "multiclass")
    assert result.confusion_matrix.shape == (4, 4)
    assert set(result.labels) == {"Benign", "DDoS", "DoS", "Port Scan"}


def test_invalid_task_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_pipeline("invalid")
