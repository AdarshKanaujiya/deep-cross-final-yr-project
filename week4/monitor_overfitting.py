from pathlib import Path

import pandas as pd


METRICS_CSV = Path(
    "week4/metrics.csv"
)

OUTPUT_REPORT = Path(
    "outputs/reports/overfitting_analysis.txt"
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

    data = pd.merge(
        train_df,
        val_df,
        on="epoch",
        how="inner",
    )

    if len(data) < 2:
        raise ValueError(
            "Not enough complete epochs for overfitting analysis."
        )

    best_idx = data["val_loss"].idxmin()

    best_epoch = int(data.loc[best_idx, "epoch"])
    best_val_loss = float(data.loc[best_idx, "val_loss"])

    overfitting_detected = False
    overfit_epoch = None

    consecutive = 0

    for i in range(1, len(data)):

        previous_train = data.iloc[i - 1]["train_loss"]
        current_train = data.iloc[i]["train_loss"]

        previous_val = data.iloc[i - 1]["val_loss"]
        current_val = data.iloc[i]["val_loss"]

        train_decreasing = current_train < previous_train
        val_increasing = current_val > previous_val

        if train_decreasing and val_increasing:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= 2:
            overfitting_detected = True
            overfit_epoch = int(data.iloc[i]["epoch"])
            break

    lines = []

    lines.append("WEEK 4 — OVERFITTING ANALYSIS")
    lines.append("=" * 60)
    lines.append("")

    lines.append(
        f"Epochs analyzed: {len(data)}"
    )

    lines.append(
        f"Best validation loss: {best_val_loss:.6f}"
    )

    lines.append(
        f"Best validation epoch: {best_epoch}"
    )

    lines.append("")

    if overfitting_detected:

        lines.append("OVERFITTING DETECTED")
        lines.append(
            f"Detected at epoch: {overfit_epoch}"
        )
        lines.append(
            "Validation loss increased while training loss "
            "continued decreasing for at least 2 consecutive epochs."
        )

        lines.append(
            f"Best checkpoint epoch: {best_epoch}"
        )

        if best_epoch < 15:
            lines.append(
                "Overfitting occurred before epoch 15."
            )
            lines.append(
                "The best validation-loss checkpoint remains valid."
            )

    else:

        lines.append("NO CLEAR OVERFITTING DETECTED")
        lines.append(
            "No sequence of 2 consecutive epochs was found "
            "where validation loss increased while training loss decreased."
        )

    lines.append("")
    lines.append("EPOCH-BY-EPOCH RESULTS")
    lines.append("-" * 60)

    for _, row in data.iterrows():

        epoch = int(row["epoch"])
        train_loss = row["train_loss"]
        val_loss = row["val_loss"]

        status = ""

        if epoch == best_epoch:
            status = "BEST VAL"

        lines.append(
            f"Epoch {epoch:>2} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"{status}"
        )

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n".join(lines))
    print()
    print(f"[OK] Saved: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()