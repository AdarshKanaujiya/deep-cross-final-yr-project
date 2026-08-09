"""
zip_val_subset.py

Copies and zips only the 4,500 validation S2 patch folders listed in val_split.csv.
Output: outputs/s2_val_subset_4500.zip (~500 MB)
This allows team members (like Person 4) to work with validation patches without downloading 58GB!
"""

import os
import shutil
import zipfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm

DATASET_ROOT = Path(r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2")
VAL_CSV = Path("outputs/metadata/val_split.csv")
OUTPUT_ZIP = Path("outputs/s2_val_subset_4500.zip")


def create_val_subset_zip():
    if not VAL_CSV.exists():
        print(f"Error: {VAL_CSV} not found!")
        return

    df_val = pd.read_csv(VAL_CSV)
    val_patch_ids = set(df_val["patch_id"].tolist())
    print(f"Found {len(val_patch_ids)} validation patches in {VAL_CSV}")

    temp_dir = Path("outputs/s2_val_subset_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    print("Copying 4,500 validation patch folders to temporary directory...")
    for patch_id in tqdm(df_val["patch_id"], desc="Copying patches"):
        tile_name = "_".join(patch_id.split("_")[:-2])
        src_folder = DATASET_ROOT / tile_name / patch_id
        dst_folder = temp_dir / tile_name / patch_id

        if src_folder.exists():
            dst_folder.mkdir(parents=True, exist_ok=True)
            for f in src_folder.glob("*.tif"):
                shutil.copy2(f, dst_folder / f.name)
            copied += 1
        else:
            missing += 1

    print(f"\nCopied {copied} patch folders (missing: {missing}). Creating zip archive...")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, temp_dir)
                zipf.write(abs_path, rel_path)

    # Clean up temp folder
    shutil.rmtree(temp_dir)

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print("\n==================================================")
    print("ZIP CREATION COMPLETE ✅")
    print("==================================================")
    print(f"Output Zip File : {OUTPUT_ZIP.resolve()}")
    print(f"File Size       : {zip_size_mb:.2f} MB")
    print("==================================================")


if __name__ == "__main__":
    create_val_subset_zip()
