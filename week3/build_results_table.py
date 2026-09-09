import pandas as pd
import os

os.makedirs("outputs/reports", exist_ok=True)

data = {
    "Retrieval Mode": [
        "SAR → Multispectral (cross-modal)",
        "Multispectral → SAR (cross-modal)",
        "SAR → SAR (same-modal)",
        "Multispectral → Multispectral (same-modal)"
    ],
    "Recall@1": ["TBD", "TBD", "TBD", "TBD"],
    "Recall@5": ["TBD", "TBD", "TBD", "TBD"],
    "Recall@10": ["TBD", "TBD", "TBD", "TBD"],
    "mAP": ["TBD", "TBD", "TBD", "TBD"],
    "Avg Query Time (ms)": ["TBD", "TBD", "TBD", "TBD"],
    "Notes": [
        "Baseline — no contrastive training",
        "Baseline — no contrastive training",
        "Baseline — no contrastive training",
        "Baseline — no contrastive training"
    ]
}

df = pd.DataFrame(data)
output_path = "outputs/reports/baseline_results_table.csv"
df.to_csv(output_path, index=False)
print(f"Template saved: {output_path}")
print("\nFill in TBD values after running compute_map.py")
print(df.to_string(index=False))