from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling


# ============================================================
# CONFIGURATION
# ============================================================

# Path to BigEarthNet-S2 dataset
DATASET_ROOT = Path(
    r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2"
)

# Metadata file location
METADATA_FILE = "metadata.parquet"


# Number of training images to process
#
# None  -> Process ALL training images (237,871)
# 100   -> Quick testing
# 1000  -> Medium testing
#
MAX_IMAGES = None



# ============================================================
# LOAD METADATA
# ============================================================

print("Loading metadata...")

metadata = pd.read_parquet(METADATA_FILE)


# Select only training patches from OUR project subset.
# IMPORTANT:
# Statistics must be calculated ONLY from the patches our model is trained on.
# Do NOT use the full BigEarthNet training set (237,871 patches) —
# that would include patches outside our 30K subset and cause subtle leakage.
#
# Our train split has 21000 patches (from train_split.csv).
train_split_df = pd.read_csv("csvs/train_split.csv")
train_patch_ids = set(train_split_df["patch_id"].tolist())

train_metadata = metadata[
    metadata["patch_id"].isin(train_patch_ids)
].reset_index(drop=True)


print(f"Total training patches in our subset: {len(train_metadata)}")



# ============================================================
# FUNCTION TO READ RGB PATCH
# ============================================================

def read_rgb_patch(patch_folder):
    """
    Reads Sentinel-2 RGB bands.

    Bands used:
        B04 -> Red
        B03 -> Green
        B02 -> Blue

    Returns:
        numpy array of shape:

        (3, height, width)

        Channel order:
        Red, Green, Blue
    """

    bands = ["B02", "B03", "B04"]

    images = []


    # --------------------------------------------------------
    # Use B02 (10m resolution) as reference
    # because RGB bands are 10m resolution.
    # --------------------------------------------------------

    reference_band = next(
        patch_folder.glob("*_B02.tif")
    )

    with rasterio.open(reference_band) as src:
        target_shape = (
            src.height,
            src.width
        )


    # --------------------------------------------------------
    # Read each RGB band
    # --------------------------------------------------------

    for band in bands:

        band_file = next(
            patch_folder.glob(f"*_{band}.tif")
        )


        with rasterio.open(band_file) as src:

            # Resize if required
            if (
                src.height,
                src.width
            ) != target_shape:

                data = src.read(
                    1,
                    out_shape=target_shape,
                    resampling=Resampling.bilinear
                )

            else:
                data = src.read(1)


        images.append(data)


    # Current order:
    # B02, B03, B04
    #
    # Convert to:
    # Red, Green, Blue

    image = np.stack(images)

    image = image[
        [2, 1, 0]
    ]


    return image.astype(np.float64)



# ============================================================
# SELECT NUMBER OF IMAGES
# ============================================================


if MAX_IMAGES is None:

    selected_metadata = train_metadata

else:

    selected_metadata = train_metadata.head(
        MAX_IMAGES
    )


print(
    f"\nProcessing {len(selected_metadata)} images..."
)



# ============================================================
# RUNNING STATISTICS VARIABLES
# ============================================================
#
# Instead of storing all pixels:
#
# 237871 images × 120 × 120 pixels
#
# which consumes huge RAM,
# we calculate statistics while reading.
#
# ============================================================


pixel_sum = np.zeros(3)

pixel_squared_sum = np.zeros(3)

pixel_count = 0



# ============================================================
# PROCESS TRAINING PATCHES
# ============================================================


processed = 0
skipped = 0


for index, patch_id in enumerate(
        selected_metadata["patch_id"],
        start=1
):


    # --------------------------------------------------------
    # Extract tile name from patch ID
    #
    # Example:
    #
    # Patch:
    # S2A_MSIL2A_....T33UUP_37_88
    #
    # Tile:
    # S2A_MSIL2A_....T33UUP
    #
    # --------------------------------------------------------

    tile_name = "_".join(
        patch_id.split("_")[:-2]
    )


    patch_folder = (
        DATASET_ROOT /
        tile_name /
        patch_id
    )


    # Skip missing folders

    if not patch_folder.exists():

        skipped += 1

        print(
            "Missing:",
            patch_folder
        )

        continue



    # Read RGB image

    rgb = read_rgb_patch(
        patch_folder
    )


    # Shape:
    # (3,120,120)

    pixels = rgb.reshape(
        3,
        -1
    )


    # Add channel sums

    pixel_sum += pixels.sum(
        axis=1
    )


    # Add squared sums

    pixel_squared_sum += (
        pixels ** 2
    ).sum(
        axis=1
    )


    # Count pixels

    pixel_count += pixels.shape[1]


    processed += 1



    if processed % 1000 == 0:

        print(
            f"{processed}/{len(selected_metadata)} images processed"
        )



# ============================================================
# CALCULATE FINAL MEAN AND STD
# ============================================================


mean = (
    pixel_sum /
    pixel_count
)


variance = (
    pixel_squared_sum /
    pixel_count
) - (
    mean ** 2
)


std = np.sqrt(
    variance
)



# ============================================================
# DISPLAY RESULTS
# ============================================================


print("\n==============================")
print("PROCESSING COMPLETE")
print("==============================")

print(
    f"Images processed: {processed}"
)

print(
    f"Images skipped: {skipped}"
)


print("\nChannel Mean:")
print(mean)


print("\nChannel Standard Deviation:")
print(std)



# ============================================================
# SAVE STATISTICS
# ============================================================


stats = {

    "bands": [
        "B04_Red",
        "B03_Green",
        "B02_Blue"
    ],

    "mean": mean.tolist(),

    "std": std.tolist(),

    "images_used": processed
}



with open(
    "s2_stats.json",
    "w"
) as file:

    json.dump(
        stats,
        file,
        indent=4
    )


print(
    "\nSaved statistics to s2_stats.json"
)