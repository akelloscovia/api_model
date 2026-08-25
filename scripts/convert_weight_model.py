import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SRC = BASE_DIR / "cattle_weight_cnn_best.keras"
DST = BASE_DIR / "cattle_weight_cnn_best.tflite"

model = tf.keras.models.load_model(SRC, compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# No converter.optimizations set — keep float32, no quantization,
# to preserve numerical parity with the original Keras model.
tflite_model = converter.convert()

DST.write_bytes(tflite_model)

print(f"Wrote {DST} ({DST.stat().st_size} bytes)")
