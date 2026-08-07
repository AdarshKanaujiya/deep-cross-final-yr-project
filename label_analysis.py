# Day 5–7: BigEarthNet Label Distribution Analysis

import pandas as pd
from collections import Counter

# -----------------------------
# Step 1: Load clean metadata
# -----------------------------
print("Loading clean metadata...\n")

df = pd.read_parquet("metadata.parquet")

print("Dataset Shape:", df.shape)

# -----------------------------
# Step 2: Count all labels
# -----------------------------
label_counter = Counter()

for labels in df["labels"]:
    label_counter.update(labels)

# -----------------------------
# Step 3: Basic Statistics
# -----------------------------
print("\n==============================")
print("LABEL DISTRIBUTION ANALYSIS")
print("==============================")

print(f"Total image patches : {len(df)}")
print(f"Total unique labels : {len(label_counter)}")

# -----------------------------
# Step 4: Top 10 labels
# -----------------------------
print("\nTop 10 Most Common Labels")
print("-" * 40)

for label, count in label_counter.most_common(10):
    print(f"{label:<35} {count}")

# -----------------------------
# Step 5: Least Common Labels
# -----------------------------
print("\n10 Least Common Labels")
print("-" * 40)

for label, count in label_counter.most_common()[-10:]:
    print(f"{label:<35} {count}")

# -----------------------------
# Step 6: Underrepresented Labels
# -----------------------------
threshold = 100

underrepresented = {
    label: count
    for label, count in label_counter.items()
    if count < threshold
}

print(f"\nUnderrepresented Labels (< {threshold} samples)")
print("-" * 40)

if underrepresented:
    for label, count in sorted(underrepresented.items(), key=lambda x: x[1]):
        print(f"{label:<35} {count}")
else:
    print("No labels found below the threshold.")

# -----------------------------
# Step 7: Summary
# -----------------------------
print("\n==============================")
print("SUMMARY")
print("==============================")

print(f"Clean image patches : {len(df)}")
print(f"Unique land-cover labels : {len(label_counter)}")
print(f"Most common label : {label_counter.most_common(1)[0][0]}")
print(f"Least common label : {label_counter.most_common()[-1][0]}")

print("\nAnalysis completed successfully.")