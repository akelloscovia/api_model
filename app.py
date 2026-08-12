import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI(title="Cattle Weight Prediction API")

# -------------------------------------------------
# Model configuration
# -------------------------------------------------
MODEL_PATH = "my_model.keras"      # Change to your actual model filename
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Training / preprocessing statistics
TARGET_MEAN = 160.60294951786727
TARGET_STD = 36.74140080538368

# Model evaluation statistics
MAE = 25.003055572509766
RMSE = 34.75696696965498
TRAIN_MEAN = 162.221923828125
TRAIN_STD = 36.01840591430664

print("Loading Cattle Weight Model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -------------------------------------------------
# Image preprocessing
# -------------------------------------------------
def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# -------------------------------------------------
# Confidence estimation
# -------------------------------------------------
def estimate_confidence(weight):
    """
    Heuristic confidence score based on distance from
    the training mean.
    """
    z = abs(weight - TRAIN_MEAN) / TRAIN_STD
    confidence = max(5.0, 100.0 * np.exp(-0.5 * z**2))
    return float(confidence)

# -------------------------------------------------
# Estimated accuracy
# -------------------------------------------------
def estimate_accuracy(weight):
    """
    Approximate prediction reliability using MAE.
    """
    if weight <= 0:
        return 0.0

    error_pct = (MAE / weight) * 100.0
    accuracy = max(0.0, min(100.0, 100.0 - error_pct))
    return float(accuracy)

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    processed_image = preprocess_image(contents)

    prediction = model.predict(processed_image, verbose=0)

    # Model outputs standardized weight
    predicted_weight = float(prediction[0][0] * TARGET_STD + TARGET_MEAN)

    confidence = estimate_confidence(predicted_weight)
    accuracy = estimate_accuracy(predicted_weight)

    lower = predicted_weight - RMSE
    upper = predicted_weight + RMSE

    return {
        "filename": file.filename,
        "predicted_weight_kg": round(predicted_weight, 2),
        "confidence_level_percent": round(confidence, 1),
        "estimated_accuracy_percent": round(accuracy, 1),
        "confidence_interval_kg": {
            "lower": round(lower, 1),
            "upper": round(upper, 1)
        },
        "expected_error_kg": round(MAE, 1),
        "model_rmse_kg": round(RMSE, 1)
    }

# -------------------------------------------------
# Run locally
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)