import getpass
import importlib.metadata as md
import platform
import sys
from datetime import datetime

import tensorflow as tf

print("User     :", getpass.getuser())
print("Date     :", datetime.now().strftime("%Y-%m-%d %H:%M"))
print("Python   :", sys.version.split()[0])
print("Platform :", platform.platform())
for pkg in ["tensorflow", "keras", "numpy", "pandas", "pillow", "matplotlib", "scikit-learn"]:
    try:
        print(f"{pkg:13}:", md.version(pkg))
    except md.PackageNotFoundError:
        print(f"{pkg:13}: not installed")

gpus = tf.config.list_physical_devices("GPU")
print("GPU      :", gpus if gpus else "none visible (CPU only)")