"""
generate_baseline_embeddings.py

Generates baseline ResNet-50 feature embeddings for the 4,500 validation patches.
Saved to outputs/embeddings/baseline_val_ms_embeddings.npy.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

from preprocessing.preprocess_s2 import preprocess_patch, DATASET_ROOT


# ============================================================
# RESNET-50 BASELINE EXTRACTOR
# ============================================================

def get_resnet50_extractor(device="cpu"):
    """
    Loads pretrained ResNet-50 with final FC layer removed.
    Output: 2048-dimensional feature vector.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()  # Remove classifier head -> 2048-dim output
    model = model.to(device)
    model.eval()
    return model


# ============================================================
# EMBEDDING GENERATION FUNCTION
# ============================================================

def generate_validation_embeddings(val_csv_path, dataset_root, output_dir, device="cpu"):
    """
    Generates 2048-dim embeddings for all validation patches.
    """
    val_csv_path = Path(val_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not val_csv_path.exists():
        print(f"Error: {val_csv_path} not found!")
        return

    df_val = pd.read_csv(val_csv_path)
    print(f"Loaded {len(df_val)} validation patches from {val_csv_path}")

    model = get_resnet50_extractor(device=device)
    print(f"Loaded pretrained ResNet-50 feature extractor on {device}")

    embeddings_list = []
    valid_patch_ids = []
    skipped_count = 0

    start_time = time.time()

    with torch.no_grad():
        for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Extracting embeddings"):
            patch_id = row["patch_id"]
            tile_name = "_".join(patch_id.split("_")[:-2])
            patch_folder = Path(dataset_root) / tile_name / patch_id

            if not patch_folder.exists():
                skipped_count += 1
                continue

            try:
                # Preprocess patch (3, 224, 224) standardized
                img_array = preprocess_patch(patch_folder)
                tensor = torch.from_numpy(img_array).unsqueeze(0).to(device)  # (1, 3, 224, 224)

                # Extract 2048-dim feature vector
                feat = model(tensor).squeeze(0).cpu().numpy()  # (2048,)
                embeddings_list.append(feat)
                valid_patch_ids.append(patch_id)
            except Exception as e:
                skipped_count += 1
                print(f"Error processing {patch_id}: {e}")

    elapsed = time.time() - start_time
    embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

    # Save output files
    emb_save_path = output_dir / "baseline_val_ms_embeddings.npy"
    ids_save_path = output_dir / "baseline_val_patch_ids.txt"

    np.save(emb_save_path, embeddings_matrix)
    with open(ids_save_path, "w") as f:
        f.write("\n".join(valid_patch_ids))

    print("\n==================================================")
    print("EMBEDDING GENERATION COMPLETE")
    print("==================================================")
    print(f"Processed      : {len(embeddings_matrix)} patches")
    print(f"Skipped        : {skipped_count} patches")
    print(f"Embedding shape: {embeddings_matrix.shape}")
    print(f"Saved array to : {emb_save_path}")
    print(f"Saved IDs to   : {ids_save_path}")
    print(f"Total time     : {elapsed:.2f} seconds ({elapsed/len(df_val)*1000:.2f} ms/patch)")
    print("==================================================")

    return embeddings_matrix


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_csv = "csvs/val_split.csv"
    output_dir = "outputs/embeddings"
    generate_validation_embeddings(val_csv, DATASET_ROOT, output_dir, device=device)
