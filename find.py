import pandas as pd

df = pd.read_csv("csvs/train_split.csv")

print(df.shape)
print(df.columns)
print(df.head())


###
# # Count train samples
# $trainCount = (Import-Csv "csvs\train_split.csv").Count
# Write-Host "Train samples: $trainCount"

# # Count validation samples
# $valCount = (Import-Csv "csvs\val_split.csv").Count
# Write-Host "Validation samples: $valCount"

###