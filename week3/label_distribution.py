import pandas as pd
import matplotlib.pyplot as plt
import os

METADATA = "metadata.parquet"
SUBSET_CSV = "csvs/subset_30k.csv"
OUTPUT = "outputs/visualizations/label_distribution.png"

os.makedirs("outputs/visualizations", exist_ok=True)

meta = pd.read_parquet(METADATA)
subset = pd.read_csv(SUBSET_CSV)

print(f"Metadata columns: {list(meta.columns)}")
print(f"Subset columns: {list(subset.columns)}")
print(f"Total subset patches: {len(subset)}")

id_col = "patch_id" if "patch_id" in meta.columns else meta.columns[0]
label_col = None
for col in meta.columns:
    if "label" in col.lower() or "lulc" in col.lower():
        label_col = col
        break
if label_col is None:
    label_col = [c for c in meta.columns if c != id_col][0]

subset_id_col = subset.columns[0]
subset_ids = set(subset[subset_id_col].astype(str).tolist())

if id_col in meta.columns:
    meta_filtered = meta[meta[id_col].astype(str).isin(subset_ids)]
else:
    meta_filtered = meta[meta.index.astype(str).isin(subset_ids)]

print(f"Matched {len(meta_filtered)} patches from subset in metadata")

label_counts = {}
for _, row in meta_filtered.iterrows():
    val = row[label_col]
    labels = val if isinstance(val, list) else [str(val)]
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

label_series = pd.Series(label_counts).sort_values(ascending=False)
print(f"\nTop 10 labels:\n{label_series.head(10)}")

fig, ax = plt.subplots(figsize=(14, 7))
colors = ["#2196F3" if i < 5 else "#78909C" for i in range(len(label_series))]
bars = ax.bar(range(len(label_series)), label_series.values, color=colors, edgecolor="white", linewidth=0.5)

ax.set_xticks(range(len(label_series)))
short_labels = [lbl[:18] + ".." if len(lbl) > 18 else lbl for lbl in label_series.index]
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Number of patches", fontsize=11)
ax.set_title("Land-Cover Label Distribution — BigEarthNet v2.0 Subset (30K pairs)\n"
             "Multi-label dataset: each patch can have multiple land-cover classes",
             fontsize=12, fontweight="bold")
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("white")

for bar, val in zip(bars, label_series.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(val), ha="center", va="bottom", fontsize=8, color="#333333")

ax.set_xlim(-0.5, len(label_series) - 0.5)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT}")
plt.close()