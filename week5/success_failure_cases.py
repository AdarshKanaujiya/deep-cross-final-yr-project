from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

BASE  = Path(__file__).resolve().parent.parent
WEEK5 = BASE / "week5"
PLOT  = WEEK5 / "plot"
PLOT.mkdir(parents=True, exist_ok=True)

S1 = BASE / "dataset" / "BigEarthNet-S1-Required"
S2 = BASE / "dataset" / "BigEarthNet-S2"

SAR_EMB = WEEK5 / "trained_test_sar_embeddings.npy"
MS_EMB  = WEEK5 / "trained_test_ms_embeddings.npy"
SAR_IDS = WEEK5 / "trained_test_sar_patch_ids.txt"
MS_IDS  = WEEK5 / "trained_test_ms_patch_ids.txt"

def ids(p):
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def norm(x):
    x = x.astype("float32")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)

def find_files(root, patch_id):
    return [p for p in root.rglob("*") if p.is_file() and patch_id in p.stem]

def load_sar(patch_id):
    files = find_files(S1, patch_id)
    vv = next((f for f in files if "VV" in f.name.upper()), None)
    if vv is None:
        return np.zeros((120, 120))
    with rasterio.open(vv) as s:
        img = s.read(1).astype(np.float32)
    lo, hi = np.percentile(img, [2, 98])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)

def load_ms(patch_id):
    files = find_files(S2, patch_id)
    def get(b):
        return next((f for f in files if b in f.name.upper()), None)
    r, g, bl = get("B04"), get("B03"), get("B02")
    if not all([r, g, bl]):
        return np.zeros((120, 120, 3))
    def rd(f):
        with rasterio.open(f) as s:
            x = s.read(1).astype(np.float32)
        lo, hi = np.percentile(x, [2, 98])
        return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return np.stack([rd(r), rd(g), rd(bl)], axis=-1)

print("Loading test embeddings...")
sar = norm(np.load(SAR_EMB))
ms  = norm(np.load(MS_EMB))
sar_ids = ids(SAR_IDS)
ms_ids  = ids(MS_IDS)
print(f"SAR {sar.shape}  MS {ms.shape}")

sim   = sar @ ms.T
top1  = np.argmax(sim, axis=1)
score = sim[np.arange(len(sar)), top1]

successes, failures = [], []
for i in range(len(sar_ids)):
    rec = {"idx": i, "query": sar_ids[i], "retrieved": ms_ids[top1[i]],
           "correct": ms_ids[i], "score": float(score[i])}
    if ms_ids[top1[i]] == ms_ids[i]:
        successes.append(rec)
    else:
        failures.append(rec)

print(f"Successes: {len(successes)}  Failures: {len(failures)}")
print(f"Top-1 Recall: {len(successes)/len(sar_ids)*100:.2f}%")

rows = []

for n, case in enumerate(successes[:3], 1):
    sar_img = load_sar(case["query"])
    ms_img  = load_ms(case["retrieved"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(sar_img, cmap="gray"); ax[0].set_title(f"SAR Query\n{case['query']}", fontsize=8); ax[0].axis("off")
    ax[1].imshow(ms_img);               ax[1].set_title(f"Top-1 MS (CORRECT)\n{case['retrieved']}\nscore={case['score']:.4f}", fontsize=8); ax[1].axis("off")
    fig.suptitle(f"Success Case {n}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = PLOT / f"success_case_{n:02d}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")
    rows.append({**case, "type": "success"})

for n, case in enumerate(failures[:3], 1):
    sar_img     = load_sar(case["query"])
    wrong_img   = load_ms(case["retrieved"])
    correct_img = load_ms(case["correct"])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(sar_img, cmap="gray"); ax[0].set_title(f"SAR Query\n{case['query']}", fontsize=8); ax[0].axis("off")
    ax[1].imshow(wrong_img);            ax[1].set_title(f"Top-1 MS (WRONG)\n{case['retrieved']}\nscore={case['score']:.4f}", fontsize=8, color="#c0392b"); ax[1].axis("off")
    ax[2].imshow(correct_img);          ax[2].set_title(f"Correct MS\n{case['correct']}", fontsize=8, color="#27ae60"); ax[2].axis("off")
    fig.suptitle(f"Failure Case {n}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = PLOT / f"failure_case_{n:02d}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")
    rows.append({**case, "type": "failure"})

pd.DataFrame(rows).to_csv(PLOT / "cases_report.csv", index=False)
print("Done. All files saved to week5/plot/")