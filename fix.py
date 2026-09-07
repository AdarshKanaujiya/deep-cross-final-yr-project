import pandas as pd

# 1. Load the original clean metadata (this has the split information already)
metadata = pd.read_parquet('metadata.parquet')

# 2. Filter to clean patches (no snow, no cloud, no shadow)
# This should give you clean metadata
# metadata['split'] column has values: 'train', 'validation', 'test'

# 3. Filter to ONLY train split
train_metadata = metadata[metadata['split'] == 'train']

# 4. Sample 30,000 pairs (you need to make sure they're paired)
# The patch_id format should be same for S1 and S2
# Check that train_metadata has s1_name and s2v1_name columns

# 5. Stratified sample across top 10 classes
# Adarsh identified the top 10 classes from his analysis
top_10_classes = ['Arable land', 'Mixed forest', 'Urban fabric', 'Inland waters', 
                  'Pastures', 'Complex cultivation patterns', 
                  'Land principally occupied by agriculture...', 
                  'Coniferous forest', 'Broad-leaved forest', 
                  'Transitional woodland, shrub']

# Filter to patches that have at least one of these classes
train_metadata['has_top_class'] = train_metadata['labels'].apply(
    lambda labels: any(cls in labels for cls in top_10_classes)
)
train_metadata = train_metadata[train_metadata['has_top_class']]

# 6. Sample 30,000 from this
train_metadata_sampled = train_metadata.sample(n=30000, random_state=42)

# 7. Split into train (21K), val (4.5K), test (4.5K)
train_split = train_metadata_sampled.iloc[:21000]
val_split = train_metadata_sampled.iloc[21000:25500]
test_split = train_metadata_sampled.iloc[25500:30000]

# 8. Save
train_split.to_csv('train_split.csv', index=False)
val_split.to_csv('val_split.csv', index=False)
test_split.to_csv('test_split.csv', index=False)

# 9. Verify
print(f"Train: {len(train_split)} rows")
print(f"Val: {len(val_split)} rows")
print(f"Test: {len(test_split)} rows")

# This should print:
# Train: 21000 rows
# Val: 4500 rows
# Test: 4500 rows