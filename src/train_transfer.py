import json
import shutil
from pathlib import Path

from input_pipeline import build_dataset
from models import compile_model
from train_utils import MODEL_DIR, REPORT_DIR, class_weights, fit_and_report, set_seed
from transfer import build_mobilenet, describe_trainable, unfreeze_last_block

set_seed()
train_ds = build_dataset("train", training=True)
val_ds = build_dataset("val", training=False)
cw = class_weights()

model, base = build_mobilenet()
report = [describe_trainable(model, base, "Stage 1: frozen backbone, train the head")]
print(report[-1])
compile_model(model, learning_rate=1e-3)
s1 = fit_and_report(model, "mobilenet_stage1", train_ds, val_ds, epochs=20, patience=5,
                    class_weight=cw, title="MobileNetV2 stage 1 (frozen)")

unfreeze_last_block(base)
report.append(describe_trainable(model, base, "Stage 2: last block unfrozen, much smaller learning rate"))
print(report[-1])
compile_model(model, learning_rate=1e-5)  # must recompile after changing trainable flags
s2 = fit_and_report(model, "mobilenet_stage2", train_ds, val_ds, epochs=15, patience=5,
                    class_weight=cw, title="MobileNetV2 stage 2 (last block unfrozen)")

(REPORT_DIR / "transfer_trainable_layers.txt").write_text("\n\n".join(report), encoding="utf-8")

# choose the better stage using VALIDATION loss only (the test set stays untouched)
final = s1 if s1["val_loss"] <= s2["val_loss"] else s2
shutil.copy(MODEL_DIR / f"{final['name']}_best.keras", MODEL_DIR / "mobilenet_best.keras")
(REPORT_DIR / "transfer_summary.json").write_text(
    json.dumps({"stage1": s1, "stage2": s2, "selected": final["name"]}, indent=2))
print(f"\nSelected for evaluation: {final['name']} -> models/mobilenet_best.keras")