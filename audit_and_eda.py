import os
import hashlib
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def compute_sha256(filepath):
    """Compute SHA-256 hash of a file to identify exact byte duplicates."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def audit_dataset(dataset_dir, output_csv="dataset_audit.csv"):
    """
    Sub-task 1.2: Dataset Audit
    Walks the dataset and logs metadata for every image:
    class, file size (bytes), width, height, colour mode, format, sha256.
    Detects corrupt files and exact duplicates.
    """
    print(f"[*] Starting Dataset Audit on: {dataset_dir}")
    records = []
    seen_hashes = {}
    corrupt_files = []
    duplicate_count = 0

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_extensions:
                continue

            filepath = os.path.join(root, file)
            # Infer class from parent folder name
            parent_dir = os.path.basename(root).lower()
            if "ai" in parent_dir or "fake" in parent_dir:
                img_class = "AI"
            elif "real" in parent_dir:
                img_class = "Real"
            else:
                img_class = os.path.basename(root)

            file_size = os.path.getsize(filepath)

            # Compute SHA-256 Hash
            try:
                content_hash = compute_sha256(filepath)
            except Exception as e:
                corrupt_files.append((filepath, f"Hash error: {e}"))
                continue

            # Check duplicate
            is_duplicate = False
            if content_hash in seen_hashes:
                is_duplicate = True
                duplicate_count += 1
            else:
                seen_hashes[content_hash] = filepath

            # Read image metadata with PIL
            try:
                with Image.open(filepath) as img:
                    img.verify() # Verify file integrity
                # Re-open after verify() to read attributes safely
                with Image.open(filepath) as img:
                    width, height = img.size
                    mode = img.mode
                    fmt = img.format
            except Exception as e:
                corrupt_files.append((filepath, str(e)))
                continue

            records.append({
                "filepath": filepath,
                "filename": file,
                "class": img_class,
                "file_size_bytes": file_size,
                "file_size_kb": round(file_size / 1024, 2),
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 3),
                "colour_mode": mode,
                "format": fmt,
                "sha256": content_hash,
                "is_duplicate": is_duplicate
            })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[+] Audit complete. Metadata for {len(df)} images saved to '{output_csv}'.")
    print(f"[!] Total Corrupt/Unreadable Files Found: {len(corrupt_files)}")
    print(f"[!] Total Exact Duplicates Found: {duplicate_count}")

    # Generate Audit Summary Table
    summary = df.groupby("class").agg(
        image_count=("filename", "count"),
        width_min=("width", "min"),
        width_median=("width", "median"),
        width_max=("width", "max"),
        height_min=("height", "min"),
        height_median=("height", "median"),
        height_max=("height", "max"),
        size_kb_min=("file_size_kb", "min"),
        size_kb_median=("file_size_kb", "median"),
        size_kb_max=("file_size_kb", "max"),
        duplicate_images=("is_duplicate", "sum")
    ).reset_index()

    print("\n" + "="*80)
    print("SUB-TASK 1.2 EVIDENCE TABLE: DATASET AUDIT SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)

    summary.to_csv("audit_summary_table.csv", index=False)
    return df, corrupt_files

def generate_visual_eda(df, output_prefix="eda"):
    """
    Sub-task 1.3: Visual EDA
    Generates:
    1. Grid of sample images per class
    2. Histograms of width, height, and file size per class
    """
    print("\n[*] Generating Visual EDA figures for report...")
    sns.set_theme(style="whitegrid")

    # 1. Histograms of Width, Height, and File Size
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Width distribution
    sns.histplot(data=df, x="width", hue="class", kde=True, ax=axes[0], palette={"AI": "#e74c3c", "Real": "#2ecc71"})
    axes[0].set_title("Distribution of Image Width by Class", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Count")

    # Height distribution
    sns.histplot(data=df, x="height", hue="class", kde=True, ax=axes[1], palette={"AI": "#e74c3c", "Real": "#2ecc71"})
    axes[1].set_title("Distribution of Image Height by Class", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Height (pixels)")
    axes[1].set_ylabel("Count")

    # File size distribution
    sns.histplot(data=df, x="file_size_kb", hue="class", kde=True, ax=axes[2], palette={"AI": "#e74c3c", "Real": "#2ecc71"})
    axes[2].set_title("Distribution of File Size (KB) by Class", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("File Size (KB)")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    hist_path = f"{output_prefix}_histograms.png"
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"[+] Saved histogram figure to '{hist_path}'.")

    # 2. Grid of sample images (4 AI vs 4 Real)
    sample_ai = df[df["class"] == "AI"].head(4)
    sample_real = df[df["class"] == "Real"].head(4)
    samples = pd.concat([sample_ai, sample_real])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("VisionGuard: Sample Image Grid (Top: AI-Generated, Bottom: Real)", fontsize=14, fontweight="bold")

    for i, (_, row) in enumerate(sample_ai.iterrows()):
        ax = axes[0, i]
        img = Image.open(row["filepath"])
        ax.imshow(img)
        ax.set_title(f"AI: {row['width']}x{row['height']}\n{row['file_size_kb']} KB", fontsize=10)
        ax.axis("off")

    for i, (_, row) in enumerate(sample_real.iterrows()):
        ax = axes[1, i]
        img = Image.open(row["filepath"])
        ax.imshow(img)
        ax.set_title(f"Real: {row['width']}x{row['height']}\n{row['file_size_kb']} KB", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    grid_path = f"{output_prefix}_sample_grid.png"
    plt.savefig(grid_path, dpi=300)
    plt.close()
    print(f"[+] Saved sample grid figure to '{grid_path}'.")

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    df, corrupt = audit_dataset(data_dir)
    if len(df) > 0:
        generate_visual_eda(df)
