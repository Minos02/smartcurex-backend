from tensorflow.keras.models import load_model
import numpy as np

m = load_model('../Piyush Gupta/SmartCureX/models/New Datasets/pneumonia_binary_best_model.h5', compile=False)
print("Input shape:", m.input_shape)
H, W, C = m.input_shape[1:]

dummy = np.zeros((1, H, W, C), dtype='float32')
pred = m.predict(dummy)
print("Output shape:", pred.shape)
print(pred)
