import random
import numpy as np
import pandas as pd

from pathlib import Path

from preprocess_s2 import preprocess_patch


DATASET_ROOT = Path(
    r"C:\Users\hp\out\micro project\final ye project\dataset\BigEarthNet-S2"
)


metadata = pd.read_parquet(
    "metadata.parquet"
)


train_split = pd.read_csv("csvs/train_split.csv")

samples = train_split.sample(
    100,
    random_state=42
)



for _, row in samples.iterrows():


    patch_id = row["patch_id"]


    tile_name = "_".join(
        patch_id.split("_")[:-2]
    )


    folder = (
        DATASET_ROOT /
        tile_name /
        patch_id
    )


    image = preprocess_patch(
        folder
    )


    # Check shape

    assert image.shape == (
        3,
        224,
        224
    )


    # Check NaN

    assert not np.isnan(
        image
    ).any()


    # Check infinity

    assert np.isfinite(
        image
    ).all()



print(
    "100 patch test passed!"
)