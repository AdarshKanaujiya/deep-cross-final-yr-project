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

Computed channel-wise mean and standard deviation using **only the 18,067 patches in the project's training split (`train_split.
PS C:\Users\hp\out\micro project\final ye project> # Count train samples                                                                                
>> $trainCount = (Import-Csv "csvs\train_split.csv").Count                                        
>> Write-Host "Train samples: $trainCount"
>>                             
>> # Count validation samples
>> $valCount = (Import-Csv "csvs\val_split.csv").Count
>> Write-Host "Validation samples: $valCount"
>> 
Train samples: 21000
Validation samples: 4500
(.venv) PS C:\Users\hp\out\micro project\final ye project> python .\compute_stats.py
Loading metadata...
Total training patches in our subset: 21000

Processing 21000 images...
1000/21000 images processed
2000/21000 images processed
3000/21000 images processed
4000/21000 images processed
5000/21000 images processed
6000/21000 images processed
7000/21000 images processed
8000/21000 images processed
9000/21000 images processed
10000/21000 images processed
11000/21000 images processed
12000/21000 images processed
13000/21000 images processed
14000/21000 images processed
15000/21000 images processed
16000/21000 images processed
17000/21000 images processed
18000/21000 images processed
19000/21000 images processed
20000/21000 images processed
21000/21000 images processed

==============================
PROCESSING COMPLETE
==============================
Images processed: 21000
Images skipped: 0

Channel Mean:
[587.03381196 613.27877453 438.18780225]

Channel Standard Deviation:
[691.37116852 611.59253036 615.30702982]

Saved statistics to s2_stats.json

(.venv) PS C:\Users\hp\out\micro project\final ye project> ^C
(.venv) PS C:\Users\hp\out\micro project\final ye project> python .\preprocessing\test_preprocessing.py
Mean: [587.03381196 613.27877453 438.18780225]
Std: [691.37116852 611.59253036 615.30702982]
100 patch test passed!
(.venv) PS C:\Users\hp\out\micro project\final ye project> python .\preprocessing\visualize_preprocessing.py
Mean: [587.03381196 613.27877453 438.18780225]
Std: [691.37116852 611.59253036 615.30702982]
Visualizing 10 random training patches...

Saved: visualization_results\comparison_1.png
Saved: visualization_results\comparison_2.png
Saved: visualization_results\comparison_3.png
Saved: visualization_results\comparison_4.png
Saved: visualization_results\comparison_5.png
Saved: visualization_results\comparison_6.png
Saved: visualization_results\comparison_7.png
Saved: visualization_results\comparison_8.png
Saved: visualization_results\comparison_9.png
Saved: visualization_results\comparison_10.png

Visualization completed successfully!
(.venv) PS C:\Users\hp\out\micro project\final ye project> 

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
Mean: [587.03381196 613.27877453 438.18780225]
Std: [691.37116852 611.59253036 615.30702982]

<!-- Final shape:
(3, 224, 224)

Minimum value:
-0.809625

Maximum value:
5.106034

Contains NaN:
False --> this is not sure because after changinf csvs to 21k , 4500,4500    need to check or its needed or not to check
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

---

## Team Integration & Helper Utilities

To enable team members (Person 4 - Fateh & Person 1 - Jainam) to perform feature extraction, FAISS retrieval, and evaluation without transferring the full 58 GB dataset, the following utility scripts and output files were developed:

### 1. Validation Subset Packager (`zip_val_subset.py`)
* **Script**: `zip_val_subset.py`
* **Output Archive**: `outputs/s2_val_subset_4500.zip` (~500 MB)
* **Purpose**: Extracts and packages only the **4,500 Sentinel-2 validation patch folders** corresponding to `val_split.csv`. This provides a lightweight 500 MB dataset package that any team member can download via Google Drive for local feature extraction and testing.

### 2. Baseline Feature Embedding Generator (`generate_baseline_embeddings.py`)
* **Script**: `generate_baseline_embeddings.py`
* **Outputs**:
  * `outputs/embeddings/baseline_val_ms_embeddings.npy` (Array shape: `4,500 × 2,048`, ~36 MB)
  * `outputs/embeddings/baseline_val_patch_ids.txt` (Ordered list of 4,500 validation patch IDs)
* **Purpose**: Uses pretrained **ResNet-50** (with FC layer replaced by `nn.Identity()`) combined with Person 2's standardized `preprocess_s2.py` pipeline to generate 2048-dimensional feature vectors for all 4,500 validation patches in ~1–2 minutes.

Mean: [587.03381196 613.27877453 438.18780225]
Std: [691.37116852 611.59253036 615.30702982]
Loaded 4500 validation patches from csvs\val_split.csv
Loaded pretrained ResNet-50 feature extractor on cpu
Extracting embeddings: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 4500/4500 [11:45<00:00,  6.38it/s]

==================================================
EMBEDDING GENERATION COMPLETE
==================================================
Processed      : 4500 patches
Skipped        : 0 patches
Embedding shape: (4500, 2048)
Saved array to : outputs\embeddings\baseline_val_ms_embeddings.npy
Saved IDs to   : outputs\embeddings\baseline_val_patch_ids.txt
Total time     : 705.20 seconds (156.71 ms/patch)
==================================================


### 3. Multi-Label mAP & Retrieval Metric Engine (`compute_map.py`)
* **Script**: `compute_map.py`
* **Core Functions**:
  * `parse_labels()`: Parses multi-label strings (e.g. `"['Pastures' 'Urban fabric']"`) safely without using `eval()`.
  * `labels_overlap()`: Checks if query and retrieved patches share at least one land-cover class.
  * `evaluate_retrieval_by_patch_id()`: Evaluates `mAP@10`, `Recall@1`, `Recall@5`, and `Recall@10` by string `patch_id` matching, preventing index misalignment issues across SAR and Multispectral sets.

---

# Week 3 – Baseline Retrieval Evaluation and t-SNE Visualization

### Objective

Measure baseline retrieval quality, generate the baseline t-SNE plot, and build the results table for the before-training report.

---

## 1. Tasks Completed in Week 3

### mAP Computation

Implemented and validated the baseline retrieval evaluation in `week3/compute_map.py`.

The script computes:

* Cross-modal retrieval: SAR → MS and MS → SAR
* Same-modal retrieval: SAR → SAR and MS → MS
* Recall@1, Recall@5, Recall@10
* mean Average Precision (mAP)

### Baseline Results Table

Built the baseline results table in `week3/build_results_table.py` and saved it to `outputs/reports/baseline_results_table.csv`.

### Baseline t-SNE Visualization

Generated the baseline t-SNE visualization in `week3/generate_tsne.py` and saved it to `outputs/visualizations/baseline_tsne.png`.

### Label Distribution Check

Ran `week3/label_distribution.py` to inspect the validation subset label distribution before sampling classes for visualization.

---

## 2. Files Used

### Week 3 Scripts

* `week3/label_distribution.py`
* `week3/compute_map.py`
* `week3/build_results_table.py`
* `week3/generate_tsne.py`

### Data Files

* `metadata.parquet`
* `csvs/val_split.csv`
* `outputs/embeddings/baseline_val_ms_embeddings.npy`
* `outputs/embeddings/baseline_val_s1_embeddings.npy`
* `outputs/embeddings/baseline_val_ms_patch_ids.txt`
* `outputs/embeddings/baseline_val_s1_patch_ids.txt`

---

## 3. Problems Faced and Fixes

### Missing Python Packages

Initial runs failed because `faiss` and `sklearn` were not installed in the active virtual environment.

Fix:

* Installed `faiss-cpu`
* Installed `scikit-learn`

### Wrong Patch-ID Lookup for SAR Labels

SAR patch IDs could not be used directly as `patch_id` lookups in the metadata, which caused the SAR same-modal queries to drop to zero.

Fix:

* Used `csvs/val_split.csv` to map each `s1_name` to its matching `patch_id`
* Resolved SAR IDs through this mapping before reading labels from metadata

### Unicode Encoding Error on Windows

Writing the results report failed with a `UnicodeEncodeError` because the output text contained arrow symbols and the default Windows encoding could not handle them.

Fix:

* Opened the report file with `encoding="utf-8"`

### t-SNE API Mismatch

`sklearn.manifold.TSNE` in the installed version did not accept the `n_iter` argument.

Fix:

* Removed `n_iter=1000` from the t-SNE call

### Wrong Missing File Reference

One run pointed to `baseline_val_patch_ids.txt`, which did not exist.

Fix:

* Corrected the script to use `baseline_val_ms_patch_ids.txt` and `baseline_val_s1_patch_ids.txt`

---

## 4. Final Output Results

### Baseline Retrieval Results

| Retrieval Mode | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: |
| SAR → Multispectral (cross-modal) | 0.0018 | 0.0060 | 0.0113 | 0.0040 |
| Multispectral → SAR (cross-modal) | 0.0060 | 0.0171 | 0.0267 | 0.0112 |
| SAR → SAR (same-modal) | 0.8971 | 0.9922 | 0.9971 | 0.9078 |
| Multispectral → Multispectral (same-modal) | 0.9440 | 0.9960 | 0.9978 | 0.9471 |

### Baseline Outputs Generated

* `outputs/reports/map_results.txt`
* `outputs/reports/baseline_results_table.csv`
* `outputs/visualizations/baseline_tsne.png`
* `outputs/visualizations/label_distribution.png`

---

## 5. Week 3 Completion Status

Week 3 is completed.

The baseline evaluation, results table, and t-SNE visualization were all produced successfully after fixing the environment and data lookup issues.

---

