from input_pipeline import build_dataset
from models import build_cnn, compile_model
from train_utils import class_weights, fit_and_report, set_seed

set_seed()
train_ds = build_dataset("train", training=True)
val_ds = build_dataset("val", training=False)

model = compile_model(build_cnn(), learning_rate=1e-3)
fit_and_report(model, "cnn", train_ds, val_ds, epochs=40, patience=8,
               class_weight=class_weights(), title="Baseline CNN")