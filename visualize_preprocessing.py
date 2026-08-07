from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess_s2 import (
    read_rgb_patch,
    preprocess_patch,
    MEAN,
    STD,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2"
)

METADATA_FILE = "metadata.parquet"

OUTPUT_FOLDER = Path("visualization_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

NUM_IMAGES = 10

# ============================================================
# LOAD TRAINING METADATA
# ============================================================

metadata = pd.read_parquet(METADATA_FILE)

train = metadata[
    metadata["split"] == "train"
].sample(NUM_IMAGES, random_state=42)

print(f"Visualizing {NUM_IMAGES} random training patches...\n")

# ============================================================
# PROCESS IMAGES
# ============================================================

for index, row in enumerate(train.itertuples(), start=1):

    patch_id = row.patch_id

    tile_name = "_".join(
        patch_id.split("_")[:-2]
    )

    patch_folder = (
        DATASET_ROOT /
        tile_name /
        patch_id
    )

    # -----------------------------
    # Original RGB
    # -----------------------------

    original = read_rgb_patch(patch_folder)

    original = original.transpose(1, 2, 0)

    original = original.astype(np.float32)

    # Min-Max scaling only for display
    original_display = (
        original - original.min()
    ) / (
        original.max() - original.min()
    )

    # -----------------------------
    # Preprocessed Image
    # -----------------------------

    processed = preprocess_patch(patch_folder)

    # Undo normalization ONLY for visualization
    processed = (
        processed * STD[:, None, None]
    ) + MEAN[:, None, None]

    processed = processed.transpose(1, 2, 0)

    processed = (
        processed - processed.min()
    ) / (
        processed.max() - processed.min()
    )

    # -----------------------------
    # Plot
    # -----------------------------

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_display)
    plt.title("Original RGB")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title("After Preprocessing")
    plt.axis("off")

    plt.tight_layout()

    save_path = OUTPUT_FOLDER / f"comparison_{index}.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {save_path}")

print("\nVisualization completed successfully!")