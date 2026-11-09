"""2.6 Error analysis: the 10 most confident mistakes of the MobileNetV2 model on the test set."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

KEY = "mobilenet"
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
lines = []


def say(t=""):
    print(t)
    lines.append(str(t))


def sharpness(path):
    """Variance of the Laplacian on a 224x224 grayscale copy: low value = blurry / smooth."""
    with Image.open(path) as im:
        g = im.convert("L").resize((224, 224))
    return float(ndimage.laplace(np.asarray(g, dtype=float)).var())


df = pd.read_csv(REPORT_DIR / f"test_predictions_{KEY}.csv")
audit = pd.read_csv(REPORT_DIR / "audit_table.csv")[
    ["path", "width", "height", "file_size_kb", "format", "mode"]]
df = df.merge(audit, on="path", how="left")
df["sharpness"] = df["path"].map(sharpness)
df["short_side"] = df[["width", "height"]].min(axis=1)

# confidence of the WRONG answer: p(AI) for real images called AI, 1 - p(AI) for AI called real
df["conf_wrong"] = np.where(df["label"] == 1, 1 - df["prob_ai"], df["prob_ai"])
wrong = df[~df["correct"]].copy()
wrong["kind"] = np.where(wrong["label"] == 0, "False positive (real called AI)",
                         "False negative (AI called real)")
fp = wrong[wrong["label"] == 0].sort_values("conf_wrong", ascending=False)
fn = wrong[wrong["label"] == 1].sort_values("conf_wrong", ascending=False)

sel = pd.concat([fp.head(5), fn.head(5)])
if len(sel) < 10:  # fill up if one error type has fewer than 5 cases
    rest = wrong.drop(sel.index).sort_values("conf_wrong", ascending=False)
    sel = pd.concat([sel, rest.head(10 - len(sel))])
sel = sel.reset_index(drop=True)
sel.index = sel.index + 1

say("=" * 70)
say(f"ERROR ANALYSIS ({KEY}) | test images: {len(df)} | mistakes: {len(wrong)} "
    f"(false positives: {len(fp)}, false negatives: {len(fn)})")
say("=" * 70)
cols = ["kind", "prob_ai", "width", "height", "file_size_kb", "format", "mode", "sharpness"]
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)
say(sel[cols].round(2).to_string())
say()
for i, r in sel.iterrows():
    say(f"#{i}: {r['path']}")
say()

say("--- Mistakes vs correct predictions (medians / shares) ---")
cmp = pd.DataFrame({
    "median sharpness": df.groupby("correct")["sharpness"].median(),
    "median file size KB": df.groupby("correct")["file_size_kb"].median(),
    "median short side px": df.groupby("correct")["short_side"].median(),
    "share PNG": df.groupby("correct")["format"].apply(lambda s: (s == "PNG").mean()),
    "share short side < 224": df.groupby("correct")["short_side"].apply(lambda s: (s < 224).mean()),
    "count": df.groupby("correct").size(),
})
cmp.index = cmp.index.map({True: "correct", False: "wrong"})
say(cmp.round(3).T.to_string())
say()
say("--- Mistakes per true class (class-level error rate) ---")
say(df.groupby("class")["correct"].agg(n="size", error_rate=lambda s: 1 - s.mean()).round(3).to_string())

sel.reset_index().rename(columns={"index": "id"}).to_csv(REPORT_DIR / "error_analysis.csv", index=False)
(REPORT_DIR / "error_analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")

# image grid
fig, axes = plt.subplots(2, 5, figsize=(21, 9.5))
for ax, (i, r) in zip(axes.ravel(), sel.iterrows()):
    with Image.open(r["path"]) as im:
        im = im.convert("RGB")
        im.thumbnail((500, 500))
        ax.imshow(im)
    true = "AI" if r["label"] == 1 else "Real"
    pred = "AI" if r["pred"] == 1 else "Real"
    ax.set_title(f"#{i}  true: {true}  |  predicted: {pred}\np(AI) = {r['prob_ai']:.2f}", fontsize=11)
    ax.set_xlabel(f"{int(r['width'])}x{int(r['height'])} px | {r['file_size_kb']:.0f} KB | "
                  f"{r['format']} | sharpness {r['sharpness']:.0f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Most confident mistakes of MobileNetV2 on the test set "
             "(top: real called AI, bottom: AI called real)", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "misclassified_grid.png", dpi=130)
print("Saved reports/figures/misclassified_grid.png")