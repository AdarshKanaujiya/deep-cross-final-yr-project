from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent
WEEK5 = BASE / "week5"
PLOT = WEEK5 / "plot"

PLOT.mkdir(parents=True, exist_ok=True)

S1 = BASE / "dataset" / "BigEarthNet-S1-Required"
S2 = BASE / "dataset" / "BigEarthNet-S2"

SAR_EMB = WEEK5 / "trained_test_sar_embeddings.npy"
MS_EMB = WEEK5 / "trained_test_ms_embeddings.npy"

SAR_IDS = WEEK5 / "trained_test_sar_patch_ids.txt"
MS_IDS = WEEK5 / "trained_test_ms_patch_ids.txt"


# ============================================================
# HELPERS
# ============================================================

def ids(p):
    return [
        l.strip()
        for l in p.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def norm(x):
    x = x.astype("float32")

    n = np.linalg.norm(x, axis=1, keepdims=True)

    return x / np.maximum(n, 1e-12)


# ============================================================
# BUILD FILE INDEX ONCE
# ============================================================

def build_file_index(root):
    """
    Scan the dataset ONCE and create:

        patch_id -> list of files

    The original script searched the entire dataset every time
    an image was requested. This version avoids that repeated scan.
    """

    print(f"Indexing files in: {root}")
    print("This is done only once...")

    index = {}

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        stem = p.stem

        # Store the complete filename stem.
        # We later match patch IDs against it.
        index.setdefault(stem, []).append(p)

    print(f"Indexed {len(index)} file stems from {root}")

    return index


def find_files(index, patch_id):
    """
    Find files belonging to a patch.

    First try an exact stem match.
    If that doesn't exist, fall back to substring matching,
    matching the behavior of the original script.
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
    files = find_files(s1_index, patch_id)

    vv = next(
        (f for f in files if "VV" in f.name.upper()),
        None
    )

    if vv is None:
        return np.zeros((120, 120))

    with rasterio.open(vv) as s:
        img = s.read(1).astype(np.float32)

    lo, hi = np.percentile(img, [2, 98])

    return np.clip(
        (img - lo) / (hi - lo + 1e-8),
        0,
        1
    )


def load_ms(patch_id, s2_index):
    files = find_files(s2_index, patch_id)

    def get(b):
        return next(
            (f for f in files if b in f.name.upper()),
            None
        )

    r = get("B04")
    g = get("B03")
    bl = get("B02")

    if not all([r, g, bl]):
        return np.zeros((120, 120, 3))

    def rd(f):
        with rasterio.open(f) as s:
            x = s.read(1).astype(np.float32)

        lo, hi = np.percentile(x, [2, 98])

        return np.clip(
            (x - lo) / (hi - lo + 1e-8),
            0,
            1
        )

    return np.stack(
        [rd(r), rd(g), rd(bl)],
        axis=-1
    )


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading test embeddings...")

sar = norm(np.load(SAR_EMB))
ms = norm(np.load(MS_EMB))

sar_ids = ids(SAR_IDS)
ms_ids = ids(MS_IDS)

print(f"SAR shape: {sar.shape}")
print(f"MS shape:  {ms.shape}")


# ============================================================
# SAFETY CHECKS
# ============================================================

if len(sar) != len(sar_ids):
    raise ValueError(
        f"SAR embedding count ({len(sar)}) "
        f"does not match SAR ID count ({len(sar_ids)})"
    )

if len(ms) != len(ms_ids):
    raise ValueError(
        f"MS embedding count ({len(ms)}) "
        f"does not match MS ID count ({len(ms_ids)})"
    )


# ============================================================
# CALCULATE SIMILARITY
# ============================================================

print("\nCalculating SAR → MS similarity...")

sim = sar @ ms.T

print("Similarity calculation finished.")


# ============================================================
# TOP-1 RETRIEVAL
# ============================================================

top1 = np.argmax(sim, axis=1)

score = sim[
    np.arange(len(sar)),
    top1
]


# ============================================================
# SUCCESS / FAILURE
# ============================================================

successes = []
failures = []

for i in range(len(sar_ids)):

    rec = {
        "idx": i,
        "query": sar_ids[i],
        "retrieved": ms_ids[top1[i]],
        "correct": ms_ids[i],
        "score": float(score[i])
    }

    if ms_ids[top1[i]] == ms_ids[i]:
        successes.append(rec)
    else:
        failures.append(rec)


print()
print(f"Successes: {len(successes)}")
print(f"Failures:  {len(failures)}")

recall = len(successes) / len(sar_ids) * 100

print(f"Top-1 Recall: {recall:.2f}%")


# ============================================================
# BUILD FILE INDEX
# ============================================================

print("\nBuilding SAR file index...")
s1_index = build_file_index(S1)

print("\nBuilding MS file index...")
s2_index = build_file_index(S2)


# ============================================================
# GENERATE SUCCESS CASES
# ============================================================

rows = []

print("\nGenerating success cases...")

for n, case in enumerate(successes[:3], 1):

    print(
        f"Success case {n}/3: "
        f"{case['query']}"
    )

    sar_img = load_sar(
        case["query"],
        s1_index
    )

    ms_img = load_ms(
        case["retrieved"],
        s2_index
    )

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    ax[0].imshow(
        sar_img,
        cmap="gray"
    )

    ax[0].set_title(
        f"SAR Query\n{case['query']}",
        fontsize=8
    )

    ax[0].axis("off")

    ax[1].imshow(ms_img)

    ax[1].set_title(
        f"Top-1 MS (CORRECT)\n"
        f"{case['retrieved']}\n"
        f"score={case['score']:.4f}",
        fontsize=8
    )

    ax[1].axis("off")

    fig.suptitle(
        f"Success Case {n}",
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()

    out = PLOT / f"success_case_{n:02d}.png"

    plt.savefig(
        out,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {out}")

    rows.append({
        **case,
        "type": "success"
    })


# ============================================================
# GENERATE FAILURE CASES
# ============================================================

print("\nGenerating failure cases...")

for n, case in enumerate(failures[:3], 1):

    print(
        f"Failure case {n}/3: "
        f"{case['query']}"
    )

    sar_img = load_sar(
        case["query"],
        s1_index
    )

    wrong_img = load_ms(
        case["retrieved"],
        s2_index
    )

    correct_img = load_ms(
        case["correct"],
        s2_index
    )

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(14, 4)
    )

    ax[0].imshow(
        sar_img,
        cmap="gray"
    )

    ax[0].set_title(
        f"SAR Query\n{case['query']}",
        fontsize=8
    )

    ax[0].axis("off")

    ax[1].imshow(wrong_img)

    ax[1].set_title(
        f"Top-1 MS (WRONG)\n"
        f"{case['retrieved']}\n"
        f"score={case['score']:.4f}",
        fontsize=8
    )

    ax[1].axis("off")

    ax[2].imshow(correct_img)

    ax[2].set_title(
        f"Correct MS\n"
        f"{case['correct']}",
        fontsize=8
    )

    ax[2].axis("off")

    fig.suptitle(
        f"Failure Case {n}",
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()

    out = PLOT / f"failure_case_{n:02d}.png"

    plt.savefig(
        out,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {out}")

    rows.append({
        **case,
        "type": "failure"
    })


# ============================================================
# SAVE CSV
# ============================================================

report_path = PLOT / "cases_report.csv"

pd.DataFrame(rows).to_csv(
    report_path,
    index=False
)

print()
print("========================================")
print("DONE")
print("========================================")
print(f"Top-1 Recall: {recall:.2f}%")
print(f"Successes:    {len(successes)}")
print(f"Failures:     {len(failures)}")
print(f"Report:       {report_path}")
print(f"Images:       {PLOT}")

