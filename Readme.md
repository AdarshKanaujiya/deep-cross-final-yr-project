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


MS embeddings: (4500, 2048), SAR embeddings: (4500, 2048)
Loading metadata for labels...
Top 5 classes selected: ['Arable land', 'Coniferous forest', 'Broad-leaved forest', 'Marine waters', 'Agro-forestry areas']
Saved exact t-SNE patch IDs: outputs/visualizations/baseline_tsne_patch_ids.txt
MS t-SNE samples: 1000
SAR t-SNE samples: 1000
Running t-SNE on 2000 embeddings (this takes 2-5 minutes)...
[t-SNE] Computing 91 nearest neighbors...
[t-SNE] Indexed 2000 samples in 0.005s...
[t-SNE] Computed neighbors for 2000 samples in 0.718s...
[t-SNE] Computed conditional probabilities for sample 1000 / 2000
[t-SNE] Computed conditional probabilities for sample 2000 / 2000
[t-SNE] Mean sigma: 2.508326
[t-SNE] KL divergence after 250 iterations with early exaggeration: 67.057503
[t-SNE] KL divergence after 1000 iterations: 1.254656
Plotting...
Saved: outputs/visualizations/baseline_tsne.png

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


---

# Week 4 – Training Monitoring and Trained Embedding Generation

### Objective

Monitor the training run, analyze overfitting, and generate trained validation embeddings from the best checkpoint for the next retrieval stage.

---

## 1. Tasks Completed in Week 4

### Training Loss Plot

Created the training-loss plotting script and generated a combined train/validation loss curve from the training CSV log.

The plot was saved as:

* `week4/training_loss_by_ad.png`

### Overfitting Analysis

Built the overfitting monitoring script and checked for the point where validation loss stopped improving while training loss kept dropping.

The analysis found:

* Best validation loss: `0.141299`
* Best validation epoch: `19`
* Overfitting detected at epoch: `17`

The report was saved as:

* `outputs/reports/overfitting_analysis.txt`

### Trained Embedding Generation

Implemented the trained embedding generation flow using the best two-tower checkpoint.

The script processed all 4,500 validation samples and saved the trained embeddings for both towers.

---

## 2. Files Used

### Week 4 Scripts

* `week4/plot_training_loss.py`
* `week4/monitor_overfitting.py`
* `week4/generate_trained_embeddings.py`

### Week 4 Support Files

* `week4/dataloader.py`
* `week4/dataset.py`
* `week4/sar_preprocessing.py`
* `preprocessing/preprocess_s2.py`
* `config/settings.py`

### Data / Model Files

* `csvs/val_split.csv`
* `models/best_model-v1.ckpt`
* `s1_stats.json`
* `s2_stats.json`
* `dataset/BigEarthNet-S1-Required`
* `dataset/BigEarthNet-S2`

---

## 3. Problems Faced and Fixes

### Missing Python Packages

Initial runs failed because the active interpreter did not have the required packages loaded.

Fix:

* Used the project virtual environment
* Installed the missing runtime dependencies, including `numpy`, `faiss-cpu`, and `scikit-learn`

### Wrong Import / Function Boundary for Sentinel-2 Preprocessing

The week4 dataset expected the Sentinel-2 preprocessing path to be available through the existing preprocessing module.

Fix:

* Kept the preprocessing function aligned with the dataset import boundary

### Sentinel-1 Statistics Path

SAR preprocessing initially looked for `s1_stats.json` in the wrong folder.

Fix:

* Pointed the loader to the project-root `s1_stats.json`

### NumPy vs Torch Tensor Mismatch

The Sentinel-2 preprocessing output was a NumPy array, but the paired collate function expected Torch tensors.

Fix:

* Converted the Sentinel-2 output to a Torch tensor inside the dataset before batching

### Dataset and Package Resolution

Several import-path issues appeared while trying to run the script directly from the terminal.

Fix:

* Kept the final run inside the project environment so the week4 modules could resolve correctly

---

## 4. Notable Points

* The checkpoint used for generation was `models/best_model-v1.ckpt`.
* Validation was run on all 4,500 paired samples.
* The trained embeddings are 512-dimensional, not 2048-dimensional, because they come from the trained projection-head towers.
* The final run completed successfully after the dataset, preprocessing, and tensor types were aligned.

---

## 5. Final Outputs

### Training and Analysis Outputs

* `week4/training_loss_by_ad.png`
* `outputs/reports/overfitting_analysis.txt`

### Trained Embedding Outputs

* `outputs/embeddings/trained_val_sar_embeddings.npy`
* `outputs/embeddings/trained_val_ms_embeddings.npy`
* `outputs/embeddings/trained_val_patch_ids.txt`

### Final Trained Embedding Shapes

* SAR: `(4500, 512)`
* MS: `(4500, 512)`
* IDs: `4500`

---

## 6. Week 4 Completion Status

Yes, Week 4 is completed.

The training plot, overfitting analysis, and trained embedding generation all ran successfully, and the trained validation embeddings were saved for the next retrieval stage.


some snapshots from implementation:
python week4/plot_training_loss.py
[OK] Training plot saved:
week4\training_loss_by_ad.png

Training epochs:
 epoch  train_loss
     0    0.791149
     1    0.328534
     2    0.231112
     3    0.171134
     4    0.146115
     5    0.123228
     6    0.104864
     7    0.096999
     8    0.082918
     9    0.071743
    10    0.058400
    11    0.052742
    12    0.046903
    13    0.041027
    14    0.032551
    15    0.026218
    16    0.025290
    17    0.020151
    18    0.021698
    19    0.021086

Validation epochs:
 epoch  val_loss
     0  0.565804
     1  0.408351
     2  0.295164
     3  0.281123
     4  0.253096
     5  0.330051
     6  0.247094
     7  0.270023
     8  0.240193
     9  0.172643
    10  0.187142
    11  0.172032
    12  0.178261
    13  0.167865
    14  0.155063
    15  0.148981
    16  0.153984
    17  0.170534
    18  0.150742
    19  0.141299

============================================================
>>>python week4/monitor_overfitting.py
WEEK 4 — OVERFITTING ANALYSIS
============================================================

Epochs analyzed: 20
Best validation loss: 0.141299
Best validation epoch: 19

OVERFITTING DETECTED
Detected at epoch: 17
Validation loss increased while training loss continued decreasing for at least 2 consecutive epochs.
Best checkpoint epoch: 19

EPOCH-BY-EPOCH RESULTS
------------------------------------------------------------
Epoch  0 | Train: 0.791149 | Val: 0.565804 | 
Epoch  1 | Train: 0.328534 | Val: 0.408351 | 
Epoch  2 | Train: 0.231112 | Val: 0.295164 | 
Epoch  3 | Train: 0.171134 | Val: 0.281123 | 
Epoch  4 | Train: 0.146115 | Val: 0.253096 | 
Epoch  5 | Train: 0.123228 | Val: 0.330051 | 
Epoch  6 | Train: 0.104864 | Val: 0.247094 | 
Epoch  7 | Train: 0.096999 | Val: 0.270023 | 
Epoch  8 | Train: 0.082918 | Val: 0.240193 | 
Epoch  9 | Train: 0.071743 | Val: 0.172643 | 
Epoch 10 | Train: 0.058400 | Val: 0.187142 | 
Epoch 11 | Train: 0.052742 | Val: 0.172032 | 
Epoch 12 | Train: 0.046903 | Val: 0.178261 | 
Epoch 13 | Train: 0.041027 | Val: 0.167865 | 
Epoch 14 | Train: 0.032551 | Val: 0.155063 | 
Epoch 15 | Train: 0.026218 | Val: 0.148981 | 
Epoch 16 | Train: 0.025290 | Val: 0.153984 | 
Epoch 17 | Train: 0.020151 | Val: 0.170534 | 
Epoch 18 | Train: 0.021698 | Val: 0.150742 | 
Epoch 19 | Train: 0.021086 | Val: 0.141299 | BEST VAL

[OK] Saved: outputs\reports\overfitting_analysis.txt

============================================================

>>>python week4/generate_trained_embeddings.py

Mean: [587.03381196 613.27877453 438.18780225]
Std: [691.37116852 611.59253036 615.30702982]


============================================================
WEEK 4 — TRAINED EMBEDDING GENERATION
============================================================
============================================================
LOADING TRAINED MODEL
============================================================
Checkpoint: C:\Users\hp\out\micro project\final ye project\models\best_model-v1.ckpt
Device:     cpu

[OK] Trained TwoTowerNetwork loaded successfully.

============================================================
CREATING VALIDATION DATALOADER
============================================================
2026-09-12 00:57:30,774 - DataLoader - INFO - Initializing PairedBigEarthNetDataset instances for train and validation...
2026-09-12 00:57:30,774 - Dataset - INFO - Loading paired split metadata from: C:\Users\hp\out\micro project\final ye project\csvs\val_split.csv
2026-09-12 00:57:30,961 - Dataset - INFO - Initialized PairedBigEarthNetDataset with 4500 samples (S2 Available: True, Require S2: True).
2026-09-12 00:57:30,962 - Dataset - INFO - Loading paired split metadata from: C:\Users\hp\out\micro project\final ye project\csvs\val_split.csv
2026-09-12 00:57:31,066 - Dataset - INFO - Initialized PairedBigEarthNetDataset with 4500 samples (S2 Available: True, Require S2: True).
2026-09-12 00:57:31,067 - DataLoader - INFO - Creating paired DataLoaders (batch_size=16, num_workers=0, require_s2=True)...
[OK] Validation samples: 4500
[OK] Validation batches:  282

============================================================
GENERATING TRAINED VALIDATION EMBEDDINGS
============================================================
Batch 1/282 | Samples processed: 16
Batch 10/282 | Samples processed: 160
Batch 20/282 | Samples processed: 320
Batch 30/282 | Samples processed: 480
Batch 40/282 | Samples processed: 640
Batch 50/282 | Samples processed: 800
Batch 60/282 | Samples processed: 960
Batch 70/282 | Samples processed: 1120
Batch 80/282 | Samples processed: 1280
Batch 90/282 | Samples processed: 1440
Batch 100/282 | Samples processed: 1600
Batch 110/282 | Samples processed: 1760
Batch 120/282 | Samples processed: 1920
Batch 130/282 | Samples processed: 2080
Batch 140/282 | Samples processed: 2240
Batch 150/282 | Samples processed: 2400
Batch 160/282 | Samples processed: 2560
Batch 170/282 | Samples processed: 2720
Batch 180/282 | Samples processed: 2880
Batch 190/282 | Samples processed: 3040
Batch 200/282 | Samples processed: 3200
Batch 210/282 | Samples processed: 3360
Batch 220/282 | Samples processed: 3520
Batch 230/282 | Samples processed: 3680
Batch 240/282 | Samples processed: 3840
Batch 250/282 | Samples processed: 4000
Batch 260/282 | Samples processed: 4160
Batch 270/282 | Samples processed: 4320
Batch 280/282 | Samples processed: 4480
Batch 282/282 | Samples processed: 4500

============================================================
SAVING TRAINED EMBEDDINGS
============================================================
SAR embedding shape: (4500, 512)
MS embedding shape:  (4500, 512)
Patch ID count:      4500

[OK] Files saved:
  SAR embeddings:
  C:\Users\hp\out\micro project\final ye project\outputs\embeddings\trained_val_sar_embeddings.npy

  MS embeddings:
  C:\Users\hp\out\micro project\final ye project\outputs\embeddings\trained_val_ms_embeddings.npy

  Patch IDs:
  C:\Users\hp\out\micro project\final ye project\outputs\embeddings\trained_val_patch_ids.txt

============================================================
TRAINED EMBEDDING GENERATION COMPLETE
============================================================

Final:
SAR: (4500, 512)
MS:  (4500, 512)
IDs: 4500
(.venv) PS C:\Users\hp\out\micro project\final ye project> 

---

# Week 5 – Retrieval Evaluation, Visual Comparison, and Result Analysis

### Objective

Evaluate the trained cross-modal retrieval model on the validation/test embedding set, generate qualitative visual evidence, and summarize the retrieval performance in a paper-style discussion.

---

## 1. Tasks Completed in Week 5

### Retrieval Evaluation on the Trained Model

Implemented the retrieval evaluation using the trained validation/test embeddings produced in Week 4.

The script calculated cosine similarity between SAR and multispectral embeddings and reported:

* Successes: `1745`
* Failures: `2755`
* Top-1 Recall: `38.78%`

This result confirms that the trained model learns a shared representation space, but still has a large number of incorrect top-1 matches under exact paired-id evaluation.

### Success / Failure Case Generation

Created a case-generation script that identifies and saves:

* 3 successful retrieval cases
* 3 failure retrieval cases

The script saved the corresponding output images into:

* `week5/plot/success_case_01.png`
* `week5/plot/success_case_02.png`
* `week5/plot/success_case_03.png`
* `week5/plot/failure_case_01.png`
* `week5/plot/failure_case_02.png`
* `week5/plot/failure_case_03.png`

It also generated a report file:

* `week5/plot/cases_report.csv`

### Visual Retrieval Grid

Built a 5 × 11 comparison grid showing:

* 5 query SAR patches
* 5 baseline MS retrievals per query
* 5 trained MS retrievals per query

Final image saved as:

* `week5/plot/visual_retrieval_comparison.png`

---

## 2. Week 5 Scripts

* `week5/success_failure_cases.py`
* `week5/visual_retrieval_grid.py`
* `week5/results_analysis.md`

---

## 3. Data and Output Files Used

### Embedding Files

* `outputs/embeddings/trained_val_sar_embeddings.npy`
* `outputs/embeddings/trained_val_ms_embeddings.npy`
* `outputs/embeddings/trained_val_patch_ids.txt`

### Baseline Files

* `outputs/embeddings/baseline_val_s1_embeddings.npy`
* `outputs/embeddings/baseline_val_ms_embeddings.npy`
* `outputs/embeddings/baseline_val_s1_patch_ids.txt`
* `outputs/embeddings/baseline_val_ms_patch_ids.txt`

### Mapping Used for Correct Alignment

The SAR-side identifiers were mapped using:

* `csvs/val_split.csv`

This was required because SAR file IDs and canonical BigEarthNet `patch_id` values are not always directly interchangeable. The script resolved the correct pairings before performing retrieval visualization.

---

## 4. Problems Faced and Fixes

### Wrong ID Matching Across Modalities

The earlier visual script failed because the baseline SAR ID list and the trained patch-ID list were compared as raw strings without resolving the correct `s1_name → patch_id` mapping.

Fix:

* Used the validation CSV to map SAR IDs back to the canonical `patch_id` used in the retrieval evaluation.

### Repeated Full-Dataset Scanning

The first runs were extremely slow because the script scanned the entire BigEarthNet-S2 tree repeatedly during image loading.

Fix:

* Built a single file index once, then looked up patch files from the index instead of calling `rglob` repeatedly.

### Image Lookup for Large Dataset

The image-loading step initially caused long delays or interrupts when the script attempted to scan the entire large Sentinel-2 directory thousands of times.

Fix:

* Used the precomputed file index and direct root/stem lookup to reduce memory and runtime stress.

---

## 5. Verified Run Results

The following command was run successfully:

```powershell
python week5/visual_retrieval_grid.py
```

Output summary:

* Baseline SAR: `(4500, 2048)`
* Baseline MS: `(4500, 2048)`
* Trained SAR: `(4500, 512)`
* Trained MS: `(4500, 512)`
* Common SAR patches: `4500`
* Selected queries: `5`
* Saved output: `week5/plot/visual_retrieval_comparison.png`

The following command was also validated successfully:

```powershell
python week5/success_failure_cases.py
```

Output summary:

* Successes: `1745`
* Failures: `2755`
* Top-1 Recall: `38.78%`
* Saved images in: `week5/plot`
* Saved report: `week5/plot/cases_report.csv`

---

## 6. Week 5 Completion Status

Yes, Week 5 is completed.

The trained retrieval evaluation, success/failure case generation, and visual comparison grid were all produced successfully. The project now contains the retrieval performance evidence and visualization outputs needed for the final report and IEEE-style result discussion.

### Black-image note (important)

The black SAR panels seen in some success/failure examples are not an error in the retrieval model itself. They appear because the script loads a selected SAR patch image from the local dataset, and some of the patches chosen by the validation set are not present in the local `BigEarthNet-S1-Required` subset on this machine. In such cases, the image loader falls back to a dark/black panel instead of raising a retrieval failure. This means the black image is a data-availability rendering issue, not a model or evaluation bug.

### Why these specific images were selected

The example images are selected from the validation/query set using the actual retrieval results. The visualization and case-generation scripts identify candidate queries from the paired validation data, then rank the retrieved multispectral patches by similarity. Only the relevant matching and non-matching examples are displayed to illustrate model behavior. This makes the visual outputs evidence of retrieval quality, not random screenshots.

### If cleaner images are required

If we want all visual examples to be visually clear without black regions, the only change needed is to filter out query/sample IDs that do not have a valid local SAR image file before plotting. This is a presentation/data-selection issue, not a performance issue. In other words, we do not need to change the model or retrieval logic; we only need to ensure that the selected examples are restricted to locally available patches. This can be done in the plotting step or by excluding missing-file cases from the comparison set.

### terminal output
### python week5/success_failure_cases.py

Loading test embeddings...
SAR shape: (4500, 512)
MS shape:  (4500, 512)

Calculating SAR → MS similarity...
Similarity calculation finished.

Successes: 1745
Failures:  2755
Top-1 Recall: 38.78%

Building SAR file index...
Indexing files in: C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S1-Required
This is done only once...
Indexed 60000 file stems from C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S1-Required

Building MS file index...
Indexing files in: C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2
This is done only once...
Indexed 6593856 file stems from C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2

Generating success cases...
Success case 1/3: S2B_MSIL2A_20170930T095019_N9999_R079_T34UEG_53_44
Saved: C:\Users\hp\out\micro project\final ye project\week5\plot\success_case_01.png
Success case 2/3: S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_31_58
Saved: C:\Users\hp\out\micro project\final ye project\week5\plot\success_case_02.png
Success case 3/3: S2B_MSIL2A_20180515T094029_N9999_R036_T35VNJ_55_65
Saved: C:\Users\hp\out\micro project\final ye project\week5\plot\success_case_03.png

Generating failure cases...
Failure case 1/3: S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66
Saved: C:\Users\hp\out\micro project\final ye project\week5\plot\failure_case_01.png
Failure case 2/3: S2A_MSIL2A_20171002T094031_N9999_R036_T34TCR_59_41
Saved: C:\Users\hp\out\micro project\final ye project\week5\plot\failure_case_02.png
Failure case 3/3: S2B_MSIL2A_20180220T114339_N9999_R123_T29UPV_24_45
Saved: C:\Users\hp\out\micro project\final ye project\week5\plot\failure_case_03.png

========================================
DONE
========================================
Top-1 Recall: 38.78%
Successes:    1745
Failures:     2755
Report:       C:\Users\hp\out\micro project\final ye project\week5\plot\cases_report.csv
Images:       C:\Users\hp\out\micro project\final ye project\week5\plot


###  python week5/visual_retrieval_grid.py

========================================
Loading embeddings...
========================================
Baseline SAR: (4500, 2048)
Baseline MS:  (4500, 2048)
Trained SAR:  (4500, 512)
Trained MS:   (4500, 512)

Running safety checks...

Building embedding ID maps...
Baseline SAR patches: 4500
Trained SAR patches:  4500
Common SAR patches:   4500

Selected queries:
  0: S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_14_77
  900: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_70_62
  1800: S2B_MSIL2A_20180422T093029_N9999_R136_T34TEQ_74_67
  2700: S2B_MSIL2A_20180220T114339_N9999_R123_T29UPV_51_19
  3600: S2B_MSIL2A_20171016T101009_N9999_R022_T34VDM_21_22

========================================
Preparing image file indexes
========================================

Building file index...
Indexing files in: C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S1-Required
This is done only once...
Indexed 60000 files
Unique file stems: 60000

Building file index...
Indexing files in: C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2
This is done only once...
Indexed 6593856 files
Unique file stems: 6593856

Normalizing gallery embeddings...

========================================
Generating visual comparison
========================================

Query 1/5
SAR patch: S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_14_77
  Loading SAR image...
  Baseline #1: S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU_73_75
  Baseline #2: S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU_75_76
  Baseline #3: S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_14_77
  Baseline #4: S2B_MSIL2A_20180525T094029_N9999_R036_T35VNL_37_13
  Baseline #5: S2A_MSIL2A_20170701T093031_N9999_R136_T35VPK_13_66
  Trained #1: S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_14_77
  Trained #2: S2B_MSIL2A_20180220T114339_N9999_R123_T29UPV_20_28
  Trained #3: S2B_MSIL2A_20180522T093029_N9999_R136_T35VPJ_29_16
  Trained #4: S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_70_76
  Trained #5: S2B_MSIL2A_20170924T093019_N9999_R136_T35VPK_24_73

Query 2/5
SAR patch: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_70_62
  Loading SAR image...
  Baseline #1: S2A_MSIL2A_20180506T100031_N9999_R122_T33UWP_74_57
  Baseline #2: S2B_MSIL2A_20170802T092029_N9999_R093_T34TFN_16_56
  Baseline #3: S2B_MSIL2A_20180502T093039_N9999_R136_T34TEP_22_73
  Baseline #4: S2B_MSIL2A_20170802T092029_N9999_R093_T34TFN_15_53
  Baseline #5: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_13_71
  Trained #1: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_70_62
  Trained #2: S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_49_70
  Trained #3: S2B_MSIL2A_20180515T112109_N9999_R037_T29SND_20_69
  Trained #4: S2A_MSIL2A_20170818T103021_N9999_R108_T32TMT_59_72
  Trained #5: S2B_MSIL2A_20180515T112109_N9999_R037_T29SND_37_17

Query 3/5
SAR patch: S2B_MSIL2A_20180422T093029_N9999_R136_T34TEQ_74_67
  Loading SAR image...
  Baseline #1: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_28_74
  Baseline #2: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_22_67
  Baseline #3: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_13_71
  Baseline #4: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_19_67
  Baseline #5: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_17_66
  Trained #1: S2B_MSIL2A_20180422T093029_N9999_R136_T34TEQ_74_67
  Trained #2: S2B_MSIL2A_20180422T093029_N9999_R136_T34TEQ_77_77
  Trained #3: S2B_MSIL2A_20180422T093029_N9999_R136_T34TEQ_73_72
  Trained #4: S2B_MSIL2A_20180509T092029_N9999_R093_T34TFN_15_57
  Trained #5: S2B_MSIL2A_20170825T093029_N9999_R136_T34TEQ_45_76

Query 4/5
SAR patch: S2B_MSIL2A_20180220T114339_N9999_R123_T29UPV_51_19
  Loading SAR image...
  Baseline #1: S2A_MSIL2A_20170720T100031_N9999_R122_T34UDG_71_20
  Baseline #2: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_13_71
  Baseline #3: S2B_MSIL2A_20180502T093039_N9999_R136_T34TEP_76_48
  Baseline #4: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_22_67
  Baseline #5: S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU_57_18
  Trained #1: S2A_MSIL2A_20180225T114351_N9999_R123_T29UPU_40_13
  Trained #2: S2A_MSIL2A_20180225T114351_N9999_R123_T29UPU_25_19
  Trained #3: S2A_MSIL2A_20180225T114351_N9999_R123_T29UPU_19_18
  Trained #4: S2A_MSIL2A_20180225T114351_N9999_R123_T29UPU_26_15
  Trained #5: S2B_MSIL2A_20171112T114339_N9999_R123_T29UPU_39_15

Query 5/5
SAR patch: S2B_MSIL2A_20171016T101009_N9999_R022_T34VDM_21_22
  Loading SAR image...
  Baseline #1: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_13_71
  Baseline #2: S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_22_67
  Baseline #3: S2A_MSIL2A_20170701T093031_N9999_R136_T35VPK_13_66
  Baseline #4: S2A_MSIL2A_20170701T093031_N9999_R136_T35VPK_21_57
  Baseline #5: S2B_MSIL2A_20170924T093019_N9999_R136_T35VNH_15_30
  Trained #1: S2B_MSIL2A_20171016T101009_N9999_R022_T34VDM_21_22
  Trained #2: S2A_MSIL2A_20180413T095031_N9999_R079_T35VLG_41_19
  Trained #3: S2A_MSIL2A_20180413T095031_N9999_R079_T35VLG_14_31
  Trained #4: S2B_MSIL2A_20171016T101009_N9999_R022_T34VDM_39_13
  Trained #5: S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_16_14

========================================
DONE
========================================
Saved: C:\Users\hp\out\micro project\final ye project\week5\plot\visual_retrieval_comparison.png

Comparison:
  5 identical SAR query patches
  Baseline model → top-5 MS
  Trained model  → top-5 MS

Output directory: C:\Users\hp\out\micro project\final ye project\week5\plot

---
