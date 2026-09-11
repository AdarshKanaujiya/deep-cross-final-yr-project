from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS_CSV = Path(
    "week4/metrics.csv"
)

OUTPUT_PLOT = Path(
    "week4/training_loss_by_ad.png"
)


def main():
    if not METRICS_CSV.exists():
        raise FileNotFoundError(
            f"Training log not found:\n{METRICS_CSV}"
        )

    df = pd.read_csv(METRICS_CSV)

    required = {"epoch", "train_loss", "val_loss"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Lightning may have NaN values in rows where only one
    # metric was logged. Keep each metric independently.
    train_df = (
        df[["epoch", "train_loss"]]
        .dropna()
        .drop_duplicates(subset=["epoch"])
        .sort_values("epoch")
    )

    val_df = (
        df[["epoch", "val_loss"]]
        .dropna()
        .drop_duplicates(subset=["epoch"])
        .sort_values("epoch")
    )

    if train_df.empty:
        raise ValueError("No train_loss values found.")

    if val_df.empty:
        raise ValueError("No val_loss values found.")

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        train_df["epoch"],
        train_df["train_loss"],
        marker="o",
        label="Train Loss",
    )

    plt.plot(
        val_df["epoch"],
        val_df["val_loss"],
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE Loss")
    plt.title("Cross-Modal Retrieval Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[OK] Training plot saved:")
    print(OUTPUT_PLOT)

    print("\nTraining epochs:")
    print(train_df.to_string(index=False))

    print("\nValidation epochs:")
    print(val_df.to_string(index=False))


if __name__ == "__main__":
    main()