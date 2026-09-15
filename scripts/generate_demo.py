from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cicids2017 import train_and_evaluate

ROOT = Path(__file__).resolve().parents[1]


def synthetic_flows(rows: int = 1_600, seed: int = 42) -> pd.DataFrame:
    """Generate CIC-like labeled features for software demonstration only."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(
        ["BENIGN", "DDoS", "DoS Hulk", "PortScan"],
        size=rows,
        p=[0.70, 0.12, 0.10, 0.08],
    )
    attack = (labels != "BENIGN").astype(float)
    port_scan = (labels == "PortScan").astype(float)
    return pd.DataFrame(
        {
            "Flow Duration": rng.lognormal(7.0 + 0.5 * attack, 0.8),
            "Total Fwd Packets": rng.poisson(8 + 16 * attack),
            "Total Backward Packets": rng.poisson(7 + 7 * attack),
            "Flow Bytes/s": rng.lognormal(5.5 + 1.2 * attack, 1.0),
            "Flow Packets/s": rng.lognormal(2.5 + 0.8 * attack, 0.7),
            "Destination Port": rng.integers(1, 65_535, rows) * (1 - port_scan)
            + rng.integers(1, 2_000, rows) * port_scan,
            "Label": labels,
        }
    )


def main() -> None:
    frame = synthetic_flows()
    result = train_and_evaluate(frame, task="multiclass")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    frame["Label"].value_counts().plot.bar(ax=axes[0], color="#4472C4")
    axes[0].set_title("Synthetic demonstration class balance")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("flows")
    sns.heatmap(
        result.confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=result.labels,
        yticklabels=result.labels,
        ax=axes[1],
    )
    axes[1].set_title(
        f"Leakage-safe pipeline\nmacro F1 = {result.macro_f1:.3f} (synthetic only)"
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    fig.tight_layout()
    output = ROOT / "docs" / "figures" / "synthetic_demo.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

