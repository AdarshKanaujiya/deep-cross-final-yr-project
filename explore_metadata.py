import pandas as pd

# 1. Load the main clean metadata
print("Loading clean dataset metadata...")
df_clean = pd.read_parquet("metadata.parquet")

# 2. Inspect the structure
print(f"Clean Dataset Shape: {df_clean.shape}")
print("\nAvailable Metadata Columns:")
print(df_clean.columns.tolist())

# 3. Print the first 20 rows
print("\n--- FIRST 20 ROWS FOR TEAM SHOWCASE ---")
pd.set_option('display.max_columns', None)
print(df_clean.head(20))

print(df_clean["split"].value_counts())      # How many train/validation/test samples
print(df_clean["country"].value_counts())    # Images per country
# print(df_clean.describe(include="all"))      # General summary