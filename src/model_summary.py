from pathlib import Path

import pandas as pd

from input_pipeline import build_dataset
from models import build_cnn, compile_model

model = compile_model(build_cnn())
model.summary()

rows = [
    {
        "layer": l.name,
        "type": l.__class__.__name__,
        "output_shape": str(tuple(l.output.shape)),
        "params": l.count_params(),
    }
    for l in model.layers
]
df = pd.DataFrame(rows)
Path("reports").mkdir(exist_ok=True)
df.to_csv("reports/cnn_layers.csv", index=False)
print()
print(df.to_string(index=False))

trainable = sum(int(w.shape.num_elements()) for w in model.trainable_weights)
total = model.count_params()
print(f"\nTotal params: {total:,} | trainable: {trainable:,} | non-trainable: {total - trainable:,}")

# sanity check on one real validation batch
xb, yb = next(iter(build_dataset("val", training=False)))
out = model(xb, training=False)
print("\nbatch in :", tuple(xb.shape), "| batch out:", tuple(out.shape),
      "| output range:", round(float(out.numpy().min()), 3), "to", round(float(out.numpy().max()), 3))