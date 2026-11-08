import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

DATA_DIR = Path("data/raw")
OUT_DIR = Path("reports")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
CLASS_NAMES = {"AiArtData": "AI", "RealArt": "Real"}

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def inspect(path):
    """Return (width, height, mode, format, error_message)."""
    try:
        with Image.open(path) as im:
            im.verify()  # structural integrity check
        with Image.open(path) as im:
            im.load()  # full decode: catches truncated files that verify() can miss
            return im.width, im.height, im.mode, im.format, ""
    except Exception as e:
        return None, None, None, None, f"{type(e).__name__}: {e}"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    files = sorted(p for p in DATA_DIR.rglob("*") if p.is_file())
    print(f"Scanning {len(files)} files under {DATA_DIR} ...")

    rows, skipped = [], []
    for i, p in enumerate(files, 1):
        top = p.relative_to(DATA_DIR).parts[0]
        cls = CLASS_NAMES.get(top, top)
        if p.suffix.lower() not in IMAGE_EXT:
            skipped.append(p.as_posix())
            continue
        w, h, mode, fmt, err = inspect(p)
        rows.append({
            "path": p.as_posix(),
            "class": cls,
            "file_size_bytes": p.stat().st_size,
            "width": w,
            "height": h,
            "mode": mode,
            "format": fmt,
            "sha256": sha256_of(p),
            "corrupt": bool(err),
            "error": err,
        })
        if i % 200 == 0:
            print(f"  processed {i}/{len(files)}")

    df = pd.DataFrame(rows)
    df["file_size_kb"] = (df["file_size_bytes"] / 1024).round(1)
    df.to_csv(OUT_DIR / "audit_table.csv", index=False)
    ok = df[~df["corrupt"]]

    pd.set_option("display.width", 130)
    pd.set_option("display.max_columns", 20)

    say("=" * 60)
    say("VISIONGUARD DATASET AUDIT")
    say("=" * 60)
    say(f"Files found            : {len(files)}")
    say(f"Image files audited    : {len(df)}")
    say(f"Non-image files skipped: {len(skipped)}")
    say()

    say("--- Images per class (total / readable / corrupt) ---")
    counts = pd.DataFrame({
        "total": df.groupby("class").size(),
        "readable": ok.groupby("class").size(),
        "corrupt": df[df["corrupt"]].groupby("class").size(),
    }).fillna(0).astype(int)
    say(counts.to_string())
    say()

    for col, label in [("width", "Width (px)"), ("height", "Height (px)"), ("file_size_kb", "File size (KB)")]:
        say(f"--- {label}: min / median / max per class ---")
        stats = ok.groupby("class")[col].agg(["min", "median", "max"]).round(1)
        say(stats.to_string())
        say()

    say("--- File format per class ---")
    say(pd.crosstab(ok["class"], ok["format"]).to_string())
    say()
    say("--- Colour mode per class ---")
    say(pd.crosstab(ok["class"], ok["mode"]).to_string())
    say()

    corrupt = df[df["corrupt"]]
    corrupt[["path", "class", "error"]].to_csv(OUT_DIR / "corrupt_files.csv", index=False)
    say(f"--- Corrupt / unreadable files: {len(corrupt)} ---")
    for _, r in corrupt.head(10).iterrows():
        say(f"  {r['path']}  ->  {r['error']}")
    say()

    dup = df[df.duplicated("sha256", keep=False)].sort_values("sha256")
    dup[["sha256", "class", "path"]].to_csv(OUT_DIR / "duplicates.csv", index=False)
    groups = dup["sha256"].nunique()
    redundant = len(dup) - groups
    cross = (dup.groupby("sha256")["class"].nunique() > 1).sum() if len(dup) else 0
    say("--- Exact duplicates (same SHA-256) ---")
    say(f"Duplicate groups                 : {groups}")
    say(f"Redundant files (could be removed): {redundant}")
    say(f"Groups appearing in BOTH classes : {cross}")
    for h_, grp in list(dup.groupby("sha256"))[:5]:
        say(f"  {h_[:12]}...  " + " | ".join(grp["path"].tolist()))
    say()

    (OUT_DIR / "audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {OUT_DIR / 'audit_table.csv'}, audit_summary.txt, corrupt_files.csv, duplicates.csv")


if __name__ == "__main__":
    main()