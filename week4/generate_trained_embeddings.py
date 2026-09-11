from pathlib import Path
import sys

import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = BASE_DIR / "models" / "best_model-v1.ckpt"

VAL_CSV = BASE_DIR / "csvs" / "val_split.csv"

S1_ROOT = Path(r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S1-Required")
S2_ROOT = Path(r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2")

OUTPUT_DIR = BASE_DIR / "outputs" / "embeddings"

SAR_OUTPUT = OUTPUT_DIR / "trained_val_sar_embeddings.npy"
MS_OUTPUT = OUTPUT_DIR / "trained_val_ms_embeddings.npy"

PATCH_IDS_OUTPUT = OUTPUT_DIR / "trained_val_patch_ids.txt"


# ============================================================
# PROJECT IMPORTS
# ============================================================

# Fix: Injected into sys.path at the absolute top of imports
# so that both 'config' (from BASE_DIR) and relative imports are discoverable.
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "week4"))

from week4.dataloader import create_paired_dataloaders
from week4.two_tower import TwoTowerNetwork



# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD MODEL FROM LIGHTNING CHECKPOINT
# ============================================================

def load_trained_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    print("=" * 60)
    print("LOADING TRAINED MODEL")
    print("=" * 60)

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device:     {DEVICE}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if "state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain a 'state_dict'."
        )

    checkpoint_state_dict = checkpoint["state_dict"]

    # --------------------------------------------------------
    # The Lightning model contains:
    #
    # self.network = TwoTowerNetwork(...)
    #
    # Therefore checkpoint keys look like:
    #
    # network.sar_encoder...
    # network.ms_encoder...
    # network.sar_projection...
    # network.ms_projection...
    #
    # We remove "network." before loading into
    # TwoTowerNetwork directly.
    # --------------------------------------------------------

    network_state_dict = {}

    for key, value in checkpoint_state_dict.items():

        if key.startswith("network."):
            new_key = key[len("network."):]
            network_state_dict[new_key] = value

    if not network_state_dict:
        raise RuntimeError(
            "Could not find 'network.' parameters in checkpoint."
        )

    # IMPORTANT:
    # pretrained=False is correct here because we are loading
    # the complete trained weights from the checkpoint.
    model = TwoTowerNetwork(pretrained=False)

    missing_keys, unexpected_keys = model.load_state_dict(
        network_state_dict,
        strict=False
    )

    if missing_keys:
        print("\nWARNING - Missing keys:")
        for key in missing_keys:
            print("  ", key)

    if unexpected_keys:
        print("\nWARNING - Unexpected keys:")
        for key in unexpected_keys:
            print("  ", key)

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model architecture mismatch. "
            "Do not generate embeddings until this is resolved."
        )

    model.to(DEVICE)
    model.eval()

    print("\n[OK] Trained TwoTowerNetwork loaded successfully.")

    return model


# ============================================================
# CREATE VALIDATION DATALOADER
# ============================================================

def create_validation_loader():

    print("\n" + "=" * 60)
    print("CREATING VALIDATION DATALOADER")
    print("=" * 60)

    if not VAL_CSV.exists():
        raise FileNotFoundError(
            f"Validation CSV not found:\n{VAL_CSV}"
        )

    if not S1_ROOT.exists():
        raise FileNotFoundError(
            f"S1 root not found:\n{S1_ROOT}"
        )

    if not S2_ROOT.exists():
        raise FileNotFoundError(
            f"S2 root not found:\n{S2_ROOT}"
        )

    _, val_loader = create_paired_dataloaders(
        train_csv=VAL_CSV,
        val_csv=VAL_CSV,
        s1_root=S1_ROOT,
        s2_root=S2_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        require_s2=True,
        drop_last_train=False,
    )

    print(f"[OK] Validation samples: {len(val_loader.dataset)}")
    print(f"[OK] Validation batches:  {len(val_loader)}")

    return val_loader


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

@torch.no_grad()
def generate_embeddings(model, val_loader):

    print("\n" + "=" * 60)
    print("GENERATING TRAINED VALIDATION EMBEDDINGS")
    print("=" * 60)

    sar_embeddings = []
    ms_embeddings = []
    patch_ids = []

    total_batches = len(val_loader)

    for batch_idx, batch in enumerate(val_loader):

        sar_batch, ms_batch, labels, batch_patch_ids = batch

        if ms_batch is None:
            raise RuntimeError(
                "MS batch is None. S2 data was not loaded."
            )

        sar_batch = sar_batch.to(
            DEVICE,
            non_blocking=True
        )

        ms_batch = ms_batch.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward through BOTH trained towers
        #
        # TwoTowerNetwork.forward() returns:
        #
        # sar_embeddings
        # ms_embeddings
        #
        # Each has dimension 512.
        # ----------------------------------------------------

        sar_batch_embeddings, ms_batch_embeddings = model(
            sar_batch,
            ms_batch
        )

        # Move results back to CPU / NumPy
        sar_embeddings.append(
            sar_batch_embeddings.cpu().numpy()
        )

        ms_embeddings.append(
            ms_batch_embeddings.cpu().numpy()
        )

        patch_ids.extend(batch_patch_ids)

        if (
            batch_idx == 0
            or (batch_idx + 1) % 10 == 0
            or batch_idx == total_batches - 1
        ):
            print(
                f"Batch {batch_idx + 1}/{total_batches} "
                f"| Samples processed: {len(patch_ids)}"
            )

    # Combine batches
    sar_embeddings = np.concatenate(
        sar_embeddings,
        axis=0
    )

    ms_embeddings = np.concatenate(
        ms_embeddings,
        axis=0
    )

    return sar_embeddings, ms_embeddings, patch_ids


# ============================================================
# SAVE RESULTS
# ============================================================

def save_outputs(
    sar_embeddings,
    ms_embeddings,
    patch_ids
):

    print("\n" + "=" * 60)
    print("SAVING TRAINED EMBEDDINGS")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    print(f"SAR embedding shape: {sar_embeddings.shape}")
    print(f"MS embedding shape:  {ms_embeddings.shape}")
    print(f"Patch ID count:      {len(patch_ids)}")

    if sar_embeddings.shape != (4500, 512):
        raise RuntimeError(
            f"Unexpected SAR embedding shape: "
            f"{sar_embeddings.shape}. "
            f"Expected (4500, 512)."
        )

    if ms_embeddings.shape != (4500, 512):
        raise RuntimeError(
            f"Unexpected MS embedding shape: "
            f"{ms_embeddings.shape}. "
            f"Expected (4500, 512)."
        )

    if len(patch_ids) != 4500:
        raise RuntimeError(
            f"Expected 4500 patch IDs, "
            f"got {len(patch_ids)}."
        )

    # --------------------------------------------------------
    # Save NumPy embeddings
    # --------------------------------------------------------

    np.save(
        SAR_OUTPUT,
        sar_embeddings
    )

    np.save(
        MS_OUTPUT,
        ms_embeddings
    )

    # --------------------------------------------------------
    # Save patch IDs
    # --------------------------------------------------------

    with open(
        PATCH_IDS_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        for patch_id in patch_ids:
            f.write(str(patch_id) + "\n")

    print("\n[OK] Files saved:")

    print(f"  SAR embeddings:")
    print(f"  {SAR_OUTPUT}")

    print(f"\n  MS embeddings:")
    print(f"  {MS_OUTPUT}")

    print(f"\n  Patch IDs:")
    print(f"  {PATCH_IDS_OUTPUT}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("WEEK 4 — TRAINED EMBEDDING GENERATION")
    print("=" * 60)

    # 1. Load trained checkpoint
    model = load_trained_model()

    # 2. Create exact paired validation loader
    val_loader = create_validation_loader()

    # 3. Generate embeddings
    sar_embeddings, ms_embeddings, patch_ids = (
        generate_embeddings(
            model,
            val_loader
        )
    )

    # 4. Save
    save_outputs(
        sar_embeddings,
        ms_embeddings,
        patch_ids
    )

    print("\n" + "=" * 60)
    print("TRAINED EMBEDDING GENERATION COMPLETE")
    print("=" * 60)

    print("\nFinal:")
    print(f"SAR: {sar_embeddings.shape}")
    print(f"MS:  {ms_embeddings.shape}")
    print(f"IDs: {len(patch_ids)}")


if __name__ == "__main__":
    main()