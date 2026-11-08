from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("reports/audit_table.csv")
df = df[~df["corrupt"]].copy()
df["aspect"] = df["width"] / df["height"]
CLASSES = ["AI", "Real"]
COLORS = {"AI": "#d1495b", "Real": "#2e86ab"}
lines = []


def say(t=""):
    print(t)
    lines.append(str(t))


# ---- Figure 1: class distribution ----
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["class"].value_counts().reindex(CLASSES)
bars = ax.bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center")
ax.set_ylabel("Number of images")
ax.set_title("Class distribution")
fig.tight_layout()
fig.savefig(OUT / "class_distribution.png", dpi=150)
plt.close(fig)


# ---- Figures 2 and 3: sample grids ----
def grid(cls, fname, rows=3, cols=4):
    sub = df[df["class"] == cls].sample(rows * cols, random_state=SEED)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax, (_, r) in zip(axes.ravel(), sub.iterrows()):
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
            im.thumbnail((400, 400))
            ax.imshow(im)
        ax.set_title(f'{int(r["width"])}x{int(r["height"])} px', fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Random samples: {cls} images (seed={SEED})", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


grid("AI", "samples_ai.png")
grid("Real", "samples_real.png")


# ---- Figures 4 to 7: per-class histograms ----
def hist(col, xlabel, fname, log=True, clip=None):
    data = df[col]
    if clip:
        lo, hi = clip
    else:
        lo, hi = data.min(), data.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 30) if log else np.linspace(lo, hi, 40)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in CLASSES:
        d = df.loc[df["class"] == c, col].clip(lo, hi)
        ax.hist(d, bins=bins, alpha=0.55, color=COLORS[c], label=f"{c} (n={len(d)})")
        ax.axvline(d.median(), color=COLORS[c], linestyle="--", linewidth=1.5)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel + (" (log scale)" if log else ""))
    ax.set_ylabel("Number of images")
    ax.set_title(f"{xlabel} per class (dashed line = median)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


hist("width", "Width (px)", "hist_width.png")
hist("height", "Height (px)", "hist_height.png")
hist("file_size_kb", "File size (KB)", "hist_filesize.png")
hist("aspect", "Aspect ratio (width / height)", "hist_aspect.png", log=False, clip=(0.3, 3.0))

# ---- Numbers for the written interpretation ----
say("=" * 55)
say("EDA SUMMARY NUMBERS")
say("=" * 55)
for c in CLASSES:
    d = df[df["class"] == c]
    n = len(d)
    say(f"[{c}]  n = {n}")
    say(f"  median width x height     : {d['width'].median():.0f} x {d['height'].median():.0f}")
    say(f"  median file size (KB)     : {d['file_size_kb'].median():.1f}")
    say(f"  median aspect ratio       : {d['aspect'].median():.2f}")
    say(f"  square (0.95-1.05)        : {((d['aspect'] >= 0.95) & (d['aspect'] <= 1.05)).mean() * 100:.1f}%")
    say(f"  landscape (> 1.05)        : {(d['aspect'] > 1.05).mean() * 100:.1f}%")
    say(f"  portrait (< 0.95)         : {(d['aspect'] < 0.95).mean() * 100:.1f}%")
    say(f"  shorter side < 224 px     : {(d[['width', 'height']].min(axis=1) < 224).mean() * 100:.1f}%")
    say(f"  shorter side >= 1000 px   : {(d[['width', 'height']].min(axis=1) >= 1000).mean() * 100:.1f}%")
    say(f"  PNG share                 : {(d['format'] == 'PNG').mean() * 100:.1f}%")
    say()
Path("reports/eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")
say(f"Figures saved in {OUT}")