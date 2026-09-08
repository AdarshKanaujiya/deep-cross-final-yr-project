# day 3-4
import pandas as pd

# -----------------------------
# Step 1: Load datasets
# -----------------------------
print("Loading datasets...")

df_clean = pd.read_parquet("metadata.parquet")
df_bad = pd.read_parquet("metadata_for_patches_with_snow_cloud_or_shadow.parquet")

# -----------------------------
# Step 2: Basic information
# -----------------------------
print("Main dataset shape:", df_clean.shape)
print("Bad dataset shape:", df_bad.shape)
print(df_bad.head())
print(df_bad.columns)

# -----------------------------
# Step 3: Count excluded patches
# -----------------------------
excluded_count = len(df_bad)
print("Excluded patches:", excluded_count)

# -----------------------------
# Step 4: Filter function
# -----------------------------
def remove_bad_patches(main_df, bad_df):
    """
    Removes all patches whose patch_id appears
    in the bad metadata file.
    """
    return main_df[
        ~main_df["patch_id"].isin(bad_df["patch_id"])
    ].copy()

# -----------------------------
# Step 5: Apply filter
# -----------------------------
filtered_df = remove_bad_patches(df_clean, df_bad)

# -----------------------------
# Step 6: Results
# -----------------------------
print("\nResults")
print("--------------------")
print("Original patches :", len(df_clean))
print("Excluded patches :", len(df_bad))
print("Remaining clean  :", len(filtered_df))
print("Removed patches  :", len(df_clean) - len(filtered_df))