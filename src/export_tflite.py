import keras
import tensorflow as tf

from config import ROOT

# the free cloud server only has 512 MB of memory, full tensorflow doesn't fit,
# so the champion is converted to tensorflow lite for the api
model = keras.models.load_model(ROOT / "models" / "champion.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

out = ROOT / "api" / "champion.tflite"
out.write_bytes(tflite_model)
print(f"saved {out} ({len(tflite_model) / 1e6:.1f} MB)")