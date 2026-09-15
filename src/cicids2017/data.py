from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ATTACK_GROUPS = {
    "BENIGN": "Benign",
    "DDoS": "DDoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "PortScan": "Port Scan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Bot": "Bot",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}


def _normalize_label(label: object) -> str:
    text = str(label).strip().replace("�", "-").replace("–", "-")
    return " ".join(text.split())


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and labels, remove duplicates, and convert infinities to missing."""
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    if "Label" not in cleaned:
        raise ValueError("The input must contain a 'Label' column.")
    cleaned["Label"] = cleaned["Label"].map(_normalize_label)
    numeric = cleaned.columns.drop("Label")
    cleaned[numeric] = cleaned[numeric].apply(pd.to_numeric, errors="coerce")
    cleaned[numeric] = cleaned[numeric].replace([np.inf, -np.inf], np.nan)
    return cleaned.drop_duplicates().reset_index(drop=True)


def load_csv_directory(
    directory: str | Path,
    sample_fraction: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load and clean every CSV in a directory, optionally sampling each file."""
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must lie in (0, 1].")
    paths = sorted(Path(directory).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {Path(directory).resolve()}")
    frames: list[pd.DataFrame] = []
    for index, path in enumerate(paths):
        frame = pd.read_csv(path, low_memory=False)
        if sample_fraction < 1:
            frame = frame.sample(frac=sample_fraction, random_state=random_state + index)
        frames.append(frame)
    return clean_frame(pd.concat(frames, ignore_index=True))


def make_target(labels: pd.Series, task: str) -> pd.Series:
    """Create binary or grouped multiclass labels."""
    normalized = labels.map(_normalize_label)
    unknown = sorted(set(normalized) - set(ATTACK_GROUPS))
    if unknown:
        raise ValueError(f"Unknown CIC-IDS2017 labels: {unknown}")
    grouped = normalized.map(ATTACK_GROUPS)
    if task == "binary":
        return grouped.map(lambda value: "Benign" if value == "Benign" else "Attack")
    if task == "multiclass":
        return grouped
    raise ValueError("task must be 'binary' or 'multiclass'.")

