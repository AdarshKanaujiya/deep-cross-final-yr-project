from pathlib import Path
import rasterio
from rasterio.enums import Resampling
import numpy as np

# Verify the folder path exists
patch_folder = Path(r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_29_55")

if not patch_folder.exists():
    print(f"Error: The folder path does not exist:\n{patch_folder}")
    exit()

# B10 is removed because BigEarthNet only contains 12 bands
bands = [
    "B01", "B02", "B03", "B04", "B05", "B06", 
    "B07", "B08", "B8A", "B09", "B11", "B12"
]

images = []
target_shape = None

# 1. First, find a reference 10m band (like B02) to get the target resolution
ref_matches = list(patch_folder.glob("*_B02.[tT][iI][fF]"))
if ref_matches:
    with rasterio.open(ref_matches[0]) as ref_src:
        target_shape = (ref_src.height, ref_src.width)  # Usually 120x120 for BigEarthNet 10m
else:
    print("Error: Could not find reference band B02 to determine target size.")
    exit()

# 2. Process and resample each band
for band in bands:
    matches = list(patch_folder.glob(f"*_{band}.[tT][iI][fF]"))
    
    if not matches:
        print(f"Error: Could not find the file for band: {band}")
        exit()
        
    tif = matches[0]
    
    with rasterio.open(tif) as src:
        # If the band resolution doesn't match 10m, resample it on the fly
        if (src.height, src.width) != target_shape:
            data = src.read(
                1,
                out_shape=target_shape,
                resampling=Resampling.bilinear
            )
        else:
            data = src.read(1)
            
        images.append(data)

# 3. Stack into a 12-channel image
image = np.stack(images)
print("Successfully stacked image shape:", image.shape)  # Expected: (12, 120, 120)

# -----------------------------
# step 2 Extract RGB (B04, B03, B02)
# -----------------------------

rgb = image[[3, 2, 1]]

print("RGB shape:", rgb.shape)

# -----------------step 3----------------------
import torch
import torch.nn.functional as F

rgb_tensor = torch.from_numpy(rgb).float().unsqueeze(0)

rgb_resized = F.interpolate(
    rgb_tensor,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

print("Resized RGB shape:", rgb_resized.shape)


# step4 
import matplotlib.pyplot as plt

# Convert from (C,H,W) to (H,W,C)
rgb_image = rgb.transpose(1, 2, 0)

# Normalize only for display
rgb_display = rgb_image.astype(np.float32)
rgb_display = (rgb_display - rgb_display.min()) / (
    rgb_display.max() - rgb_display.min()
)

plt.imshow(rgb_display)
plt.title("Original RGB Image")
plt.axis("off")

plt.savefig("rgb_preview.png", dpi=300)
plt.show()