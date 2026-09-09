import numpy as np
import pandas as pd
import faiss
import time
import os
from collections import defaultdict

MS_EMBEDDINGS = "outputs/embeddings/baseline_val_ms_embeddings.npy"
SAR_EMBEDDINGS = "outputs/embeddings/baseline_val_s1_embeddings.npy"
MS_IDS = "outputs/embeddings/baseline_val_ms_patch_ids.txt"
SAR_IDS = "outputs/embeddings/baseline_val_s1_patch_ids.txt"
METADATA = "metadata.parquet"
OUTPUT_REPORT = "outputs/reports/map_results.txt"
TOP_K = [1, 5, 10]

os.makedirs("outputs/reports", exist_ok=True)

print("Loading embeddings...")
ms_emb = np.load(MS_EMBEDDINGS).astype("float32")
s1_emb = np.load(SAR_EMBEDDINGS).astype("float32")

with open(MS_IDS) as f:
    ms_ids = [line.strip() for line in f.readlines()]
with open(SAR_IDS) as f:
    s1_ids = [line.strip() for line in f.readlines()]

print(f"MS: {ms_emb.shape}, SAR: {s1_emb.shape}")

print("Loading metadata...")
meta = pd.read_parquet(METADATA)
val_split = pd.read_csv("csvs/val_split.csv")
s1_to_patch_id = dict(zip(val_split["s1_name"], val_split["patch_id"]))

label_col = None
for col in meta.columns:
    if "label" in col.lower() or "lulc" in col.lower() or "class" in col.lower():
        label_col = col
        break
if label_col is None:
    label_col = meta.columns[0]
print(f"Using label column: {label_col}")

id_col = "patch_id" if "patch_id" in meta.columns else meta.index.name or meta.columns[0]
if id_col != meta.index.name:
    meta = meta.set_index(id_col)

def parse_label_set(value):
    if isinstance(value, list):
        return set(value)
    if isinstance(value, np.ndarray):
        return set(value.tolist())
    if isinstance(value, str) and value.startswith("["):
        import ast
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                return set(parsed)
        except Exception:
            pass
    return {str(value)}

def resolve_metadata_id(patch_id):
    return s1_to_patch_id.get(patch_id, patch_id)

def get_labels(patch_id):
    try:
        resolved_id = resolve_metadata_id(patch_id)
        val = meta.loc[resolved_id, label_col]
        return parse_label_set(val)
    except:
        return set()

ms_id_to_idx = {pid: i for i, pid in enumerate(ms_ids)}
s1_id_to_idx = {pid: i for i, pid in enumerate(s1_ids)}

def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return x / norms

ms_emb_norm = l2_normalize(ms_emb)
s1_emb_norm = l2_normalize(s1_emb)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def compute_metrics(query_emb, query_ids, gallery_emb, gallery_ids,
                    gallery_index, mode="cross", paired_ids=None,
                    top_k_list=TOP_K):
    max_k = max(top_k_list) + 1
    D, I = gallery_index.search(query_emb, max_k)
    elapsed = (time.time() - start) / len(query_emb) * 1000
    print(f"Avg query time: {elapsed:.3f} ms")

    recall_at_k = {k: [] for k in top_k_list}
    ap_list = []

    for q_idx, q_id in enumerate(query_ids):
        retrieved_indices = I[q_idx]
        retrieved_ids = [gallery_ids[i] for i in retrieved_indices if i < len(gallery_ids)]

        if mode == "cross":
            if paired_ids is not None and q_idx < len(paired_ids):
                relevant_ids = {paired_ids[q_idx]}
            else:
                relevant_ids = set()
        else:
            q_labels = get_labels(q_id)
            relevant_ids = {gid for gid in gallery_ids
                            if gid != q_id and get_labels(gid) & q_labels}

        if not relevant_ids:
            continue

        if mode == "same":
            retrieved_ids = retrieved_ids[1:]

        for k in top_k_list:
            top_k_retrieved = set(retrieved_ids[:k])
            hit = 1 if top_k_retrieved & relevant_ids else 0
            recall_at_k[k].append(hit)

        precision_at_k = []
        hits = 0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                hits += 1
                precision_at_k.append(hits / rank)
        ap = np.mean(precision_at_k) if precision_at_k else 0.0
        ap_list.append(ap)

    results = {}
    for k in top_k_list:
        results[f"Recall@{k}"] = np.mean(recall_at_k[k]) if recall_at_k[k] else 0.0
    results["mAP"] = np.mean(ap_list) if ap_list else 0.0
    results["num_queries"] = len(ap_list)
    return results

print("\nBuilding FAISS indices...")
ms_index = build_faiss_index(ms_emb_norm)
s1_index = build_faiss_index(s1_emb_norm)

print("\nEvaluating SAR-to-Multispectral (cross-modal)...")
sar_to_ms = compute_metrics(s1_emb_norm, s1_ids, ms_emb_norm, ms_ids,
                             ms_index, mode="cross", paired_ids=ms_ids)

print("Evaluating Multispectral-to-SAR (cross-modal)...")
ms_to_sar = compute_metrics(ms_emb_norm, ms_ids, s1_emb_norm, s1_ids,
                             s1_index, mode="cross", paired_ids=s1_ids)

print("Evaluating SAR-to-SAR (same-modal)...")
sar_to_sar = compute_metrics(s1_emb_norm, s1_ids, s1_emb_norm, s1_ids,
                              s1_index, mode="same")

print("Evaluating Multispectral-to-Multispectral (same-modal)...")
ms_to_ms = compute_metrics(ms_emb_norm, ms_ids, ms_emb_norm, ms_ids,
                            ms_index, mode="same")

all_results = {
    "SAR → Multispectral (cross-modal)": sar_to_ms,
    "Multispectral → SAR (cross-modal)": ms_to_sar,
    "SAR → SAR (same-modal)": sar_to_sar,
    "Multispectral → Multispectral (same-modal)": ms_to_ms,
}

print("\n" + "="*70)
print("BASELINE EVALUATION RESULTS (No contrastive training)")
print("="*70)
lines = []
lines.append("BASELINE EVALUATION RESULTS")
lines.append("No contrastive training — pretrained ResNet50 features only")
lines.append("="*70)

for mode_name, metrics in all_results.items():
    line = (f"{mode_name:45s} | "
            f"R@1: {metrics['Recall@1']:.4f} | "
            f"R@5: {metrics['Recall@5']:.4f} | "
            f"R@10: {metrics['Recall@10']:.4f} | "
            f"mAP: {metrics['mAP']:.4f} | "
            f"Queries: {metrics['num_queries']}")
    print(line)
    lines.append(line)

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nSaved: {OUTPUT_REPORT}")