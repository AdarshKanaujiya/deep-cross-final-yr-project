import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import os
import random

MS_EMBEDDINGS = "outputs/embeddings/baseline_val_ms_embeddings.npy"
SAR_EMBEDDINGS = "outputs/embeddings/baseline_val_s1_embeddings.npy"
MS_IDS = "outputs/embeddings/baseline_val_ms_patch_ids.txt"
SAR_IDS = "outputs/embeddings/baseline_val_s1_patch_ids.txt"
METADATA = "metadata.parquet"
OUTPUT = "outputs/visualizations/baseline_tsne.png"
SAMPLES_PER_MODALITY = 1000
N_CLASSES = 5
RANDOM_SEED = 42

os.makedirs("outputs/visualizations", exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Loading embeddings...")
ms_emb = np.load(MS_EMBEDDINGS)
s1_emb = np.load(SAR_EMBEDDINGS)

with open(MS_IDS) as f:
    ms_ids = [line.strip() for line in f.readlines()]
with open(SAR_IDS) as f:
    s1_ids = [line.strip() for line in f.readlines()]

print(f"MS embeddings: {ms_emb.shape}, SAR embeddings: {s1_emb.shape}")

print("Loading metadata for labels...")
meta = pd.read_parquet(METADATA)
meta = meta.set_index("patch_id") if "patch_id" in meta.columns else meta
val_split = pd.read_csv("csvs/val_split.csv")
s1_to_patch_id = dict(zip(val_split["s1_name"], val_split["patch_id"]))

label_col = None
for col in meta.columns:
    if "label" in col.lower() or "class" in col.lower() or "lulc" in col.lower():
        label_col = col
        break

if label_col is None:
    print(f"Available columns: {list(meta.columns)}")
    label_col = meta.columns[0]
    print(f"Using column: {label_col}")

def resolve_metadata_id(patch_id):
    return s1_to_patch_id.get(patch_id, patch_id)

def get_primary_label(patch_id):
    try:
        row = meta.loc[resolve_metadata_id(patch_id)]
        val = row[label_col]
        if isinstance(val, list):
            return val[0] if len(val) > 0 else "unknown"
        if isinstance(val, np.ndarray):
            return str(val[0]) if len(val) > 0 else "unknown"
        if isinstance(val, str) and val.startswith("["):
            import ast
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                    return str(parsed[0])
            except Exception:
                pass
        return str(val)
    except:
        return "unknown"

ms_labels = [get_primary_label(pid) for pid in ms_ids]
s1_labels = [get_primary_label(pid) for pid in s1_ids]

from collections import Counter
ms_label_counts = Counter(ms_labels)
top_classes = [cls for cls, _ in ms_label_counts.most_common(N_CLASSES) if cls != "unknown"]
print(f"Top {N_CLASSES} classes selected: {top_classes}")

ms_filtered_idx = [i for i, lbl in enumerate(ms_labels) if lbl in top_classes]
s1_filtered_idx = [i for i, lbl in enumerate(s1_labels) if lbl in top_classes]

ms_sample_idx = random.sample(ms_filtered_idx, min(SAMPLES_PER_MODALITY, len(ms_filtered_idx)))
s1_sample_idx = random.sample(s1_filtered_idx, min(SAMPLES_PER_MODALITY, len(s1_filtered_idx)))

ms_sample_emb = ms_emb[ms_sample_idx]
s1_sample_emb = s1_emb[s1_sample_idx]
ms_sample_labels = [ms_labels[i] for i in ms_sample_idx]
s1_sample_labels = [s1_labels[i] for i in s1_sample_idx]

combined = np.vstack([ms_sample_emb, s1_sample_emb])
combined_labels = ms_sample_labels + s1_sample_labels
combined_modality = ["Multispectral"] * len(ms_sample_idx) + ["SAR"] * len(s1_sample_idx)

print(f"Running t-SNE on {len(combined)} embeddings (this takes 2-5 minutes)...")
# CHANGE 'n_iter' TO 'max_iter' BELOW:
tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, max_iter=1000, verbose=1)
reduced = tsne.fit_transform(combined)

print("Plotting...")
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("white")

class_markers = ["o", "s", "^", "D", "P"]
class_to_marker = {cls: class_markers[i] for i, cls in enumerate(top_classes)}

ms_color = "#2196F3"
sar_color = "#F44336"

n_ms = len(ms_sample_idx)
for i, cls in enumerate(top_classes):
    ms_mask = [j for j, lbl in enumerate(ms_sample_labels) if lbl == cls]
    s1_mask = [j for j, lbl in enumerate(s1_sample_labels) if lbl == cls]
    marker = class_to_marker[cls]
    short_cls = cls[:20] + "..." if len(cls) > 20 else cls
    if ms_mask:
        ax.scatter(reduced[ms_mask, 0], reduced[ms_mask, 1],
                   c=ms_color, marker=marker, alpha=0.6, s=25,
                   label=f"MS — {short_cls}")
    if s1_mask:
        s1_global = [j + n_ms for j in s1_mask]
        ax.scatter(reduced[s1_global, 0], reduced[s1_global, 1],
                   c=sar_color, marker=marker, alpha=0.6, s=25,
                   label=f"SAR — {short_cls}")

ms_patch = mpatches.Patch(color=ms_color, label="Multispectral (S2)")
sar_patch = mpatches.Patch(color=sar_color, label="SAR (S1)")
ax.legend(handles=[ms_patch, sar_patch] , loc="upper right", fontsize=10)

ax.set_title("t-SNE: Baseline Embeddings — Domain Gap Between SAR and Multispectral\n"
             "(No contrastive training — same-location patches appear in separate clusters)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
ax.tick_params(labelsize=9)

note = ("Each point = one image patch. Red = SAR (Sentinel-1). Blue = Multispectral (Sentinel-2).\n"
        "Separation between red and blue clusters demonstrates the domain gap this project addresses.")
fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#555555",
         style="italic", wrap=True)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
plt.close()