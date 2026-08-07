import pandas as pd

df = pd.read_csv("outputs/metadata/train_split.csv")

print(df.shape)
print(df.columns)
print(df.head())