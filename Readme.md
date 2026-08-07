Here's a clean README-style summary of the work completed across **Week 1** and **Week 2**.

---

# BigEarthNet Cross-Modal Retrieval Project

## Person 2 – Metadata Analysis & Sentinel-2 Preprocessing

## Week 1 – Metadata Exploration and Analysis

### Objective

Understand the BigEarthNet v2.0 dataset structure and prepare clean metadata for downstream preprocessing.

### Tasks Completed

#### 1. Downloaded Required Files

* ✅ `metadata.parquet`
* ✅ `metadata_for_patches_with_snow_cloud_or_shadow.parquet`

#### 2. Explored Dataset Metadata

Loaded the metadata using **Pandas** and inspected the dataset.

Key findings:

* Total clean patches: **480,038**
* Metadata columns:

  * `patch_id`
  * `labels`
  * `split`
  * `country`
  * `s1_name`
  * `s2v1_name`
  * `contains_seasonal_snow`
  * `contains_cloud_or_shadow`

#### 3. Verified Dataset Splits

| Split      | Number of Patches |
| ---------- | ----------------: |
| Train      |           237,871 |
| Validation |           122,342 |
| Test       |           119,825 |

#### 4. Displayed Dataset Samples

* Printed the first 20 rows of metadata.
* Shared the dataset preview with the team.

#### 5. Snow/Cloud Metadata Analysis

Loaded the metadata containing snow, cloud, and cloud-shadow patches.

Verified:

* Clean metadata excludes corrupted patches.
* Dataset is ready for model training.

#### 6. Label Distribution Analysis

* Counted land-cover labels.
* Identified the most common classes.
* Checked for class imbalance.
* Shared findings with the team for future subset sampling.

---
# Week 2 – Sentinel-2 Multispectral Preprocessing Pipeline

### Objective

Build and validate the Sentinel-2 preprocessing pipeline required before feature extraction and deep learning model training for the cross-modal satellite image retrieval system.

---

## 1. Sentinel-2 Image Loading

Implemented code to read Sentinel-2 GeoTIFF patches from the BigEarthNet-S2 dataset using Rasterio.

For the RGB preprocessing pipeline, the following 10 m resolution bands were used:

* **B04** – Red
* **B03** – Green
* **B02** – Blue

The bands are loaded and reordered into standard RGB channel order:

```text
B04 → Red
B03 → Green
B02 → Blue
```

Each BigEarthNet-S2 RGB patch is initially loaded with the shape:

```text
(3, 120, 120)
```

---

## 2. Image Resizing

Implemented bilinear interpolation using PyTorch to resize each RGB patch from:

```text
120 × 120
```

to:

```text
224 × 224
```

The resulting image shape is:

```text
(3, 224, 224)
```

The 224 × 224 input size makes the processed images compatible with commonly used CNN architectures such as ResNet-50.

---

## 3. Training-Set Channel Statistics

Computed channel-wise mean and standard deviation using **only the 18,067 patches in the project's training split (`train_split.csv`)**.

This was corrected from the initial calculation that used the full BigEarthNet training set. Using only the project's training data ensures that the normalization statistics correspond to the actual data used for model training.

Final statistics:

| Channel |     Mean | Standard Deviation |
| ------- | -------: | -----------------: |
| Red     | 635.3302 |           682.1374 |
| Green   | 658.0161 |           598.0100 |
| Blue    | 453.9845 |           604.7489 |

The statistics were saved to:

```text
s2_stats.json
```

These statistics are used for channel-wise normalization during preprocessing.

---

## 4. Image Normalization

Implemented channel-wise standardization using:

```text
Normalized Pixel = (Pixel − Mean) / Standard Deviation
```

Normalization is performed independently for the Red, Green, and Blue channels.

The output is **standardized rather than min-max normalized**, so the resulting values are not expected to fall between 0 and 1.

For the tested patch, the normalized values were:

```text
Minimum: -0.809625
Maximum: 5.106034
```

This is expected behavior for standardization.

---

## 5. Complete Preprocessing Pipeline

Developed a reusable preprocessing function that performs the complete sequence:

```text
Sentinel-2 GeoTIFF
        ↓
Read B02, B03, B04
        ↓
Reorder to RGB
        ↓
Resize 120 × 120 → 224 × 224
        ↓
Normalize using training statistics
        ↓
Return processed image
```

Final output shape:

```text
(3, 224, 224)
```

This processed representation can be passed to the feature-extraction backbone in the later retrieval pipeline.

---

## 6. Efficient Statistics Computation

The statistics computation was implemented using **running sums and squared sums** rather than storing every pixel from every training image in memory.

This avoids loading the complete training dataset into RAM simultaneously.

The final computation successfully processed:

```text
Training patches: 18,067
Images processed: 18,067
Images skipped: 0
```

This approach significantly reduces memory consumption while still producing the required channel-wise mean and standard deviation.

---

## 7. Testing the Preprocessing Pipeline

The preprocessing pipeline was tested on **100 training patches**.

The tests verified:

* ✅ Correct output shape
* ✅ Successful RGB band loading
* ✅ Correct resizing to 224 × 224
* ✅ No NaN values
* ✅ Successful normalization
* ✅ Successful preprocessing across the test patches

Final preprocessing verification:

```text
Mean:
[635.33024992 658.01613194 453.98453217]

Std:
[682.1374482  598.00997309 604.74891859]

Final shape:
(3, 224, 224)

Minimum value:
-0.809625

Maximum value:
5.106034

Contains NaN:
False
```

---

## 8. Visualization

Created a visualization pipeline to inspect the effect of preprocessing on Sentinel-2 RGB patches.

The visualization compares the original RGB image with the processed representation to verify that the preprocessing pipeline is functioning correctly and that the spatial structure of the satellite image is preserved.

Generated comparison images are stored under:

```text
visualization_results/
```

---

# Tools & Libraries Used

* Python
* NumPy
* Pandas
* Rasterio
* PyTorch
* Matplotlib
* JSON

---

# Files Produced / Used

```text
metadata.parquet
metadata_for_patches_with_snow_cloud_or_shadow.parquet

compute_stats.py
preprocess_s2.py
test_preprocessing.py
visualize_preprocessing.py

s2_stats.json

visualization_results/
    comparison_1.png
    comparison_2.png
    ...
    comparison_10.png
```

---

# Outcome

By the end of Week 2, the project has a **verified Sentinel-2 RGB preprocessing pipeline** ready for integration with the retrieval system.

The pipeline successfully loads BigEarthNet-S2 RGB patches, converts them into standard RGB order, resizes them from 120 × 120 to 224 × 224, and applies channel-wise standardization using statistics calculated exclusively from the project's **18,067 training patches**.

The pipeline was successfully tested without missing values or NaNs, and the statistics computation processed all **18,067 training patches with zero skipped images**.

The resulting `(3, 224, 224)` normalized representation is now ready to be used by the feature-extraction and retrieval stages planned for the subsequent weeks.
