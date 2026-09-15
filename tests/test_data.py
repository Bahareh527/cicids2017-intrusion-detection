from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cicids2017 import clean_frame, load_csv_directory, make_target


def test_clean_frame_normalizes_and_deduplicates() -> None:
    frame = pd.DataFrame(
        {
            " Feature ": [1.0, 1.0, np.inf],
            " Label ": ["BENIGN", "BENIGN", "Web Attack � XSS"],
        }
    )
    cleaned = clean_frame(frame)
    assert list(cleaned.columns) == ["Feature", "Label"]
    assert len(cleaned) == 2
    assert cleaned["Feature"].isna().sum() == 1
    assert cleaned["Label"].iloc[1] == "Web Attack - XSS"


def test_targets_are_grouped() -> None:
    labels = pd.Series(["BENIGN", "DoS Hulk", "FTP-Patator"])
    assert make_target(labels, "binary").tolist() == ["Benign", "Attack", "Attack"]
    assert make_target(labels, "multiclass").tolist() == ["Benign", "DoS", "Brute Force"]


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        make_target(pd.Series(["new attack"]), "binary")


def test_directory_loader_is_deterministic(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": range(20), "Label": ["BENIGN"] * 20})
    frame.to_csv(tmp_path / "one.csv", index=False)
    first = load_csv_directory(tmp_path, sample_fraction=0.5, random_state=3)
    second = load_csv_directory(tmp_path, sample_fraction=0.5, random_state=3)
    pd.testing.assert_frame_equal(first, second)

