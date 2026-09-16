# Week 5 — Results Analysis

## 1. Baseline Retrieval

Without contrastive training, the pretrained ResNet50 features produce
modality-specific embeddings that are not aligned across SAR and
multispectral domains. The baseline SAR-to-MS cross-modal retrieval
on the validation set achieved Recall@10 of 1.13%, confirming that
nearest-neighbour search without alignment training is essentially
random for cross-modal pairs.

## 2. Effect of Cross-Modal Contrastive Training

After Two-Tower InfoNCE training on 21,000 paired patches, the
test-set evaluation achieved:

- SAR → MS: Recall@1 38.78%, Recall@5 64.82%, Recall@10 75.04%
- MS → SAR: Recall@1 37.04%, Recall@5 63.73%, Recall@10 74.00%
- SAR → SAR: Recall@1 53.40%, Recall@5 85.98%, Recall@10 92.49%, mAP@10 61.24%
- MS → MS:  Recall@1 57.84%, Recall@5 87.04%, Recall@10 93.47%, mAP@10 64.09%

The visual retrieval comparison grid confirms this qualitatively.
In 4 of 5 query cases, the trained model retrieved the exact correct
paired MS patch as Top-1, while the baseline Top-1 retrievals were
geographically unrelated patches.

## 3. Failure Analysis

Failures occur when multiple patches share similar land-cover
composition — arable fields, mixed forest, and coastal areas produce
similar cross-modal representations even across different locations.
Class imbalance in BigEarthNet means dominant classes (arable land,
coniferous forest) have many visually similar patches, making exact
paired retrieval harder. Seasonal variation between SAR and MS
acquisition dates also contributes to misalignment for some patches.