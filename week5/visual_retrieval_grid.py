from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent
EMB = BASE / "outputs" / "embeddings"
WEEK5 = BASE / "week5"
PLOT = WEEK5 / "plot"
VAL_CSV = BASE / "csvs" / "val_split.csv"

PLOT.mkdir(parents=True, exist_ok=True)

S1 = BASE / "dataset" / "BigEarthNet-S1-Required"
S2 = BASE / "dataset" / "BigEarthNet-S2"


# ============================================================
# EMBEDDING FILES
# ============================================================

B_SAR_EMB = EMB / "baseline_val_s1_embeddings.npy"
B_MS_EMB = EMB / "baseline_val_ms_embeddings.npy"

B_SAR_IDS = EMB / "baseline_val_s1_patch_ids.txt"
B_MS_IDS = EMB / "baseline_val_ms_patch_ids.txt"

T_SAR_EMB = EMB / "trained_val_sar_embeddings.npy"
T_MS_EMB = EMB / "trained_val_ms_embeddings.npy"

T_IDS = EMB / "trained_val_patch_ids.txt"


# ============================================================
# HELPERS
# ============================================================

def ids(path):
    """
    Read patch IDs from a text file.
    """

    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def norm(x):
    """
    L2-normalize embedding rows.
    """

    x = x.astype("float32")

    n = np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    )

    return x / np.maximum(n, 1e-12)


# ============================================================
# BUILD FILE INDEX
# ============================================================

def build_file_index(root):
    """
    Scan the dataset once.

    Creates:

        filename_stem -> list of files

    This avoids repeatedly scanning the 6.5M+ MS files.
    """

    print()
    print("Building file index...")
    print(f"Indexing files in: {root}")
    print("This is done only once...")

    index = {}
    count = 0

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        count += 1

        stem = p.stem

        index.setdefault(
            stem,
            []
        ).append(p)

    print(
        f"Indexed {count} files"
    )

    print(
        f"Unique file stems: {len(index)}"
    )

    return index


def find_files(index, patch_id):
    """
    Find files belonging to a patch.

    Exact lookup first.
    Substring fallback second.
    """

    # Fast path
    if patch_id in index:

        return index[patch_id]

    # Fallback
    matches = []

    for stem, files in index.items():

        if patch_id in stem:

            matches.extend(files)

    return matches


# ============================================================
# IMAGE LOADERS
# ============================================================

def load_sar(patch_id, s1_index):
    """
    Load Sentinel-1 VV image.
    """

    files = find_files(
        s1_index,
        patch_id
    )

    vv = next(
        (
            f for f in files
            if "VV" in f.name.upper()
        ),
        None
    )

    if vv is None:

        print(
            f"WARNING: VV not found: {patch_id}"
        )

        return np.zeros(
            (120, 120)
        )

    with rasterio.open(vv) as src:

        img = src.read(
            1
        ).astype(
            np.float32
        )

    lo, hi = np.percentile(
        img,
        [2, 98]
    )

    return np.clip(
        (img - lo) /
        (hi - lo + 1e-8),
        0,
        1
    )


def load_ms(patch_id, s2_index):
    """
    Load Sentinel-2 RGB.

    B04 = Red
    B03 = Green
    B02 = Blue
    """

    files = find_files(
        s2_index,
        patch_id
    )

    def get_band(band):

        return next(
            (
                f for f in files
                if band in f.name.upper()
            ),
            None
        )

    r = get_band("B04")
    g = get_band("B03")
    b = get_band("B02")

    if not all([r, g, b]):

        print(
            f"WARNING: RGB not found: {patch_id}"
        )

        return np.zeros(
            (120, 120, 3)
        )

    def read_band(path):

        with rasterio.open(path) as src:

            x = src.read(
                1
            ).astype(
                np.float32
            )

        lo, hi = np.percentile(
            x,
            [2, 98]
        )

        return np.clip(
            (x - lo) /
            (hi - lo + 1e-8),
            0,
            1
        )

    red = read_band(r)
    green = read_band(g)
    blue = read_band(b)

    return np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )


# ============================================================
# TOP-5 RETRIEVAL
# ============================================================

def top5(query_embedding, gallery_embeddings):
    """
    Retrieve top-5 gallery embeddings using cosine similarity.
    """

    query = query_embedding.astype(
        np.float32
    )

    query_norm = np.linalg.norm(
        query
    )

    query = query / max(
        query_norm,
        1e-12
    )

    gallery = norm(
        gallery_embeddings
    )

    similarity = gallery @ query

    return np.argsort(
        -similarity
    )[:5]


# ============================================================
# VALIDATION ID MAPPING
# ============================================================

def load_patch_id_maps(csv_path):
    """
    Build canonical patch-id mapping for validation data.

    Some SAR ID files are stored as s1_name values, while the real
    paired metadata uses patch_id. Use the CSV to translate both forms
    back to the canonical patch_id used in the embedding files.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if {"s1_name", "patch_id"}.issubset(df.columns):
        s1_to_patch = dict(zip(df["s1_name"], df["patch_id"]))
        patch_to_s1 = dict(zip(df["patch_id"], df["s1_name"]))
        return s1_to_patch, patch_to_s1

    raise ValueError(f"Validation CSV is missing required columns: {csv_path}")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print()
print("========================================")
print("Loading embeddings...")
print("========================================")


b_sar = np.load(
    B_SAR_EMB
)

b_ms = np.load(
    B_MS_EMB
)

t_sar = np.load(
    T_SAR_EMB
)

t_ms = np.load(
    T_MS_EMB
)


b_sar_ids_raw = ids(
    B_SAR_IDS
)

b_ms_ids = ids(
    B_MS_IDS
)

t_ids = ids(
    T_IDS
)

s1_to_patch_id, patch_to_s1_name = load_patch_id_maps(VAL_CSV)

b_sar_ids = [s1_to_patch_id.get(pid, pid) for pid in b_sar_ids_raw]

print(
    f"Baseline SAR: {b_sar.shape}"
)

print(
    f"Baseline MS:  {b_ms.shape}"
)

print(
    f"Trained SAR:  {t_sar.shape}"
)

print(
    f"Trained MS:   {t_ms.shape}"
)


# ============================================================
# SAFETY CHECKS
# ============================================================

print()
print("Running safety checks...")


if len(b_sar) != len(b_sar_ids):

    raise ValueError(
        "Baseline SAR embedding count does not "
        "match baseline SAR ID count."
    )


if len(b_ms) != len(b_ms_ids):

    raise ValueError(
        "Baseline MS embedding count does not "
        "match baseline MS ID count."
    )


if len(t_sar) != len(t_ids):

    raise ValueError(
        "Trained SAR embedding count does not "
        "match trained SAR ID count."
    )


if len(t_ms) != len(t_ids):

    raise ValueError(
        "Trained MS embedding count does not "
        "match trained MS ID count."
    )


# ============================================================
# BUILD ID → INDEX MAPS
# ============================================================

print()
print("Building embedding ID maps...")


b_sar_map = {
    patch_id: i
    for i, patch_id in enumerate(
        b_sar_ids
    )
}


b_ms_map = {
    patch_id: i
    for i, patch_id in enumerate(
        b_ms_ids
    )
}


t_sar_map = {
    patch_id: i
    for i, patch_id in enumerate(
        t_ids
    )
}


t_ms_map = {
    patch_id: i
    for i, patch_id in enumerate(
        t_ids
    )
}


# ============================================================
# CHECK PATCH OVERLAP
# ============================================================

baseline_sar_set = set(
    b_sar_ids
)

trained_sar_set = set(
    t_ids
)


common_ids = (
    baseline_sar_set &
    trained_sar_set
)


print(
    f"Baseline SAR patches: {len(baseline_sar_set)}"
)

print(
    f"Trained SAR patches:  {len(trained_sar_set)}"
)

print(
    f"Common SAR patches:   {len(common_ids)}"
)


if len(common_ids) == 0:

    raise ValueError(
        "No common patch IDs found between "
        "baseline and trained datasets. "
        "Check that the SAR ID file is mapped through val_split.csv."
    )


# ============================================================
# SELECT QUERIES
# ============================================================

QUERIES = [
    0,
    900,
    1800,
    2700,
    3600
]


# ============================================================
# VERIFY QUERY PATCHES
# ============================================================

for qi in QUERIES:

    if qi < 0 or qi >= len(b_sar_ids):

        raise IndexError(
            f"Query index {qi} is invalid."
        )

    patch_id = b_sar_ids[qi]

    if patch_id not in t_sar_map:

        raise ValueError(
            f"Query patch is missing from trained "
            f"embeddings: {patch_id}"
        )


print()
print("Selected queries:")

for qi in QUERIES:

    print(
        f"  {qi}: {b_sar_ids[qi]}"
    )


# ============================================================
# BUILD DATASET FILE INDEX
# ============================================================

print()
print("========================================")
print("Preparing image file indexes")
print("========================================")


s1_index = build_file_index(
    S1
)

s2_index = build_file_index(
    S2
)


# ============================================================
# PRE-NORMALIZE GALLERIES
# ============================================================

print()
print("Normalizing gallery embeddings...")


b_ms_norm = norm(
    b_ms
)

t_ms_norm = norm(
    t_ms
)


# ============================================================
# CREATE FIGURE
# ============================================================

print()
print("========================================")
print("Generating visual comparison")
print("========================================")


fig, axes = plt.subplots(
    len(QUERIES),
    11,
    figsize=(24, 12)
)


fig.patch.set_facecolor(
    "white"
)


# ============================================================
# PROCESS QUERIES
# ============================================================

for row, qi in enumerate(QUERIES):

    sar_id = b_sar_ids[qi]

    print()
    print(
        f"Query {row + 1}/{len(QUERIES)}"
    )

    print(
        f"SAR patch: {sar_id}"
    )

    # The actual S1 folder name is s1_name, not patch_id.
    sar_lookup_id = patch_to_s1_name.get(sar_id, sar_id)

    # ========================================================
    # GET TRAINED EMBEDDING FOR SAME SAR PATCH
    # ========================================================

    if sar_id not in t_sar_map:
        raise ValueError(
            f"Query patch is missing from trained embeddings: {sar_id}"
        )

    trained_sar_idx = t_sar_map[
        sar_id
    ]

    trained_query = t_sar[
        trained_sar_idx
    ]


    # ========================================================
    # BASELINE TOP-5
    # ========================================================

    baseline_query = b_sar[qi]

    baseline_q = (
        baseline_query /
        max(
            np.linalg.norm(
                baseline_query
            ),
            1e-12
        )
    )

    baseline_scores = (
        b_ms_norm @ baseline_q
    )

    b_top = np.argsort(
        -baseline_scores
    )[:5]


    # ========================================================
    # TRAINED TOP-5
    # ========================================================

    trained_q = (
        trained_query /
        max(
            np.linalg.norm(
                trained_query
            ),
            1e-12
        )
    )

    trained_scores = (
        t_ms_norm @ trained_q
    )

    t_top = np.argsort(
        -trained_scores
    )[:5]


    # ========================================================
    # LOAD SAR IMAGE
    # ========================================================

    print(
        "  Loading SAR image..."
    )

    sar_img = load_sar(
        sar_lookup_id,
        s1_index
    )


    # ========================================================
    # DISPLAY SAR
    # ========================================================

    ax = axes[
        row,
        0
    ]

    ax.imshow(
        sar_img,
        cmap="gray"
    )

    ax.axis(
        "off"
    )

    if row == 0:

        ax.set_title(
            "Query\nSAR",
            fontsize=8,
            fontweight="bold"
        )


    # ========================================================
    # BASELINE RESULTS
    # ========================================================

    for k in range(5):

        ms_idx = b_top[k]

        retrieved_id = b_ms_ids[
            ms_idx
        ]

        score = baseline_scores[
            ms_idx
        ]

        print(
            f"  Baseline #{k + 1}: "
            f"{retrieved_id}"
        )

        ms_img = load_ms(
            retrieved_id,
            s2_index
        )

        ax = axes[
            row,
            1 + k
        ]

        ax.imshow(
            ms_img
        )

        ax.axis(
            "off"
        )

        if row == 0:

            ax.set_title(
                f"Base\n#{k + 1}",
                fontsize=7
            )


    # ========================================================
    # TRAINED RESULTS
    # ========================================================

    for k in range(5):

        ms_idx = t_top[k]

        retrieved_id = t_ids[
            ms_idx
        ]

        score = trained_scores[
            ms_idx
        ]

        print(
            f"  Trained #{k + 1}: "
            f"{retrieved_id}"
        )

        ms_img = load_ms(
            retrieved_id,
            s2_index
        )

        ax = axes[
            row,
            6 + k
        ]

        ax.imshow(
            ms_img
        )

        ax.axis(
            "off"
        )

        if row == 0:

            ax.set_title(
                f"Trained\n#{k + 1}",
                fontsize=7
            )


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "SAR → MS Retrieval: Baseline (val) vs Trained (val)\n"
    "Same SAR query patches | "
    "Baseline top-5 vs Trained top-5",
    fontsize=11,
    fontweight="bold"
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()


out = (
    PLOT /
    "visual_retrieval_comparison.png"
)


plt.savefig(
    out,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# DONE
# ============================================================

print()
print("========================================")
print("DONE")
print("========================================")

print(
    f"Saved: {out}"
)

print()
print(
    "Comparison:"
)

print(
    "  5 identical SAR query patches"
)

print(
    "  Baseline model → top-5 MS"
)

print(
    "  Trained model  → top-5 MS"
)

print()
print(
    f"Output directory: {PLOT}"
)