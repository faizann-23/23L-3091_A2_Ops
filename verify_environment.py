import sys
import platform

print("="*60)
print("VISIONGUARD ENVIRONMENT VERIFICATION (TASK 1.1)")
print("="*60)
print(f"Python Version      : {platform.python_version()}")
print(f"Python Path         : {sys.executable}")
print(f"Operating System    : {platform.system()} {platform.release()} ({platform.machine()})")

try:
    import tensorflow as tf
    print(f"TensorFlow Version  : {tf.__version__}")
    try:
        import keras
        print(f"Keras Version       : {keras.__version__}")
    except Exception:
        pass
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Available       : YES ({len(gpus)} GPU(s) detected)")
        for i, gpu in enumerate(gpus):
            print(f"  -> GPU {i}: {gpu.name}")
    else:
        print("GPU Available       : NO (Running in CPU mode)")
except ImportError:
    print("TensorFlow          : NOT INSTALLED YET")

try:
    import pandas as pd
    print(f"Pandas Version      : {pd.__version__}")
except ImportError:
    pass

try:
    import PIL
    print(f"Pillow Version      : {PIL.__version__}")
except ImportError:
    pass

try:
    import matplotlib
    print(f"Matplotlib Version  : {matplotlib.__version__}")
except ImportError:
    pass

print("="*60)
