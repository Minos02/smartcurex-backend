import tensorflow as tf
from tensorflow import keras
import numpy as np

# CHANGE THIS TO YOUR EXACT MODEL PATH
model_path = r'../Piyush Gupta/SmartCureX/samplez/BrainTumor/brain_tumor_model_cpu.h5'

print(f"Loading model from: {model_path}")
model = keras.models.load_model(model_path)

print("=" * 60)
print("BRAIN TUMOR MODEL ARCHITECTURE")
print("=" * 60)

# 1. INPUT SHAPE
print(f"\n✅ Model Input Shape: {model.input_shape}")

# 2. FULL SUMMARY
print("\n📋 FULL MODEL SUMMARY:")
print("-" * 60)
model.summary()

# 3. TEST DIFFERENT SIZES
print("\n🧪 TESTING INPUT SIZES:")
test_sizes = [(150, 150), (180, 180), (192, 192), (224, 224), (128, 128), (111, 111), (112, 112), (96, 96)]

for size in test_sizes:
    try:
        test_img = np.random.rand(1, size[0], size[1], 3).astype('float32')
        pred = model.predict(test_img, verbose=0)
        print(f"   ✅ {size} WORKS! → Output: {pred.shape}")
        print(f"      USE THIS SIZE: img.resize({size})")
        break
    except Exception as e:
        error_msg = str(e)
        if "expected" in error_msg:
            print(f"   ❌ {size} - {error_msg[:80]}")
        else:
            print(f"   ❌ {size} failed")

print("\n" + "=" * 60)
