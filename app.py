import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from ultralytics import YOLO


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Cattle Weight Prediction API",
    description="Detects cattle first, then predicts cattle weight.",
    version="2.0.0"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WEIGHT_MODEL_PATH = BASE_DIR / "cattle_weight_cnn_best.keras"
YOLO_MODEL_PATH = BASE_DIR / "yolo11n.pt"


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224


# ============================================================
# CATTLE DETECTION CONFIGURATION
# ============================================================

# COCO class ID for cow
COW_CLASS_ID = 19

# Minimum confidence required for YOLO to accept a cow
CATTLE_DETECTION_THRESHOLD = 0.40


# ============================================================
# TRAINING / PREPROCESSING STATISTICS
# ============================================================

TARGET_MEAN = 160.60294951786727
TARGET_STD = 36.74140080538368


# ============================================================
# MODEL EVALUATION STATISTICS
# ============================================================

MAE = 25.003055572509766
RMSE = 34.75696696965498

TRAIN_MEAN = 162.221923828125
TRAIN_STD = 36.01840591430664


# ============================================================
# LOAD CATTLE WEIGHT MODEL
# ============================================================

print("=" * 60)
print("LOADING CATTLE WEIGHT MODEL")
print("=" * 60)

print("Weight model path:")
print(WEIGHT_MODEL_PATH)

if not WEIGHT_MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Cattle weight model not found: {WEIGHT_MODEL_PATH}"
    )

weight_model = tf.keras.models.load_model(
    WEIGHT_MODEL_PATH,
    compile=False
)

print("Cattle weight model loaded successfully!")

print("Input shape:", weight_model.input_shape)
print("Output shape:", weight_model.output_shape)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("=" * 60)
print("LOADING YOLO CATTLE DETECTOR")
print("=" * 60)

print("YOLO model path:")
print(YOLO_MODEL_PATH)

if not YOLO_MODEL_PATH.exists():

    raise FileNotFoundError(
        f"YOLO model not found: {YOLO_MODEL_PATH}"
    )

cattle_detector = YOLO(
    str(YOLO_MODEL_PATH)
)

print("YOLO model loaded successfully!")

print("Cow class ID:", COW_CLASS_ID)
print("Cow class name:", cattle_detector.names[COW_CLASS_ID])


# ============================================================
# IMAGE PREPROCESSING FOR WEIGHT MODEL
# ============================================================

def preprocess_image(image_bytes: bytes):

    """
    Convert uploaded image into the format
    expected by the cattle weight model.
    """

    img = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    img = img.resize(
        (IMG_WIDTH, IMG_HEIGHT)
    )

    img_array = np.array(
        img,
        dtype=np.float32
    ) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# CATTLE DETECTION
# ============================================================

def detect_cattle(image_bytes: bytes):

    """
    Run YOLO before the weight model.

    Returns:

        {
            "cattle_detected": True/False,
            "confidence": highest cow confidence,
            "detections": [...]
        }
    """

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Run YOLO
    results = cattle_detector.predict(
        source=image,
        conf=CATTLE_DETECTION_THRESHOLD,
        verbose=False
    )

    detections = []

    highest_cow_confidence = 0.0

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            class_name = cattle_detector.names[
                class_id
            ]

            detections.append(
                {
                    "class": class_name,
                    "confidence": round(
                        confidence,
                        3
                    )
                }
            )

            # Check specifically for cow
            if class_id == COW_CLASS_ID:

                highest_cow_confidence = max(
                    highest_cow_confidence,
                    confidence
                )

    cattle_detected = (
        highest_cow_confidence
        >= CATTLE_DETECTION_THRESHOLD
    )

    return {
        "cattle_detected": cattle_detected,

        "confidence": round(
            highest_cow_confidence,
            3
        ),

        "detections": detections
    }


# ============================================================
# CONFIDENCE ESTIMATION
# ============================================================

def estimate_confidence(weight):

    z = abs(
        weight - TRAIN_MEAN
    ) / TRAIN_STD

    confidence = max(
        5.0,
        100.0 * np.exp(
            -0.5 * z ** 2
        )
    )

    return float(confidence)


# ============================================================
# ESTIMATED ACCURACY
# ============================================================

def estimate_accuracy(weight):

    if weight <= 0:

        return 0.0

    error_pct = (
        MAE / weight
    ) * 100.0

    accuracy = max(
        0.0,
        min(
            100.0,
            100.0 - error_pct
        )
    )

    return float(accuracy)


# ============================================================
# MAIN PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # STEP 1 — READ IMAGE
    # --------------------------------------------------------

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    # --------------------------------------------------------
    # STEP 2 — CHECK THAT FILE IS AN IMAGE
    # --------------------------------------------------------

    try:

        Image.open(
            io.BytesIO(contents)
        ).verify()

    except Exception:

        return {
            "valid": False,
            "cattle_detected": False,
            "message": (
                "Invalid file. "
                "Please upload a valid image."
            )
        }

    # --------------------------------------------------------
    # STEP 3 — RUN CATTLE DETECTION
    # --------------------------------------------------------

    detection = await run_in_threadpool(
        detect_cattle,
        contents
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # If there is NO cattle:
    #
    # STOP HERE.
    #
    # The TensorFlow weight model will NOT run.
    # --------------------------------------------------------

    if not detection["cattle_detected"]:

        return {

            "valid": False,

            "cattle_detected": False,

            "message": (
                "Invalid image. "
                "No cattle detected. "
                "Please upload an image containing cattle."
            ),

            "detections": detection[
                "detections"
            ]
        }

    # --------------------------------------------------------
    # STEP 4 — CATTLE WAS DETECTED
    # --------------------------------------------------------

    processed_image = preprocess_image(
        contents
    )

    # --------------------------------------------------------
    # STEP 5 — PREDICT WEIGHT
    # --------------------------------------------------------

    prediction = await run_in_threadpool(
        weight_model.predict,
        processed_image,
        verbose=0
    )

    predicted_weight = float(
        prediction[0][0]
        * TARGET_STD
        + TARGET_MEAN
    )

    # Prevent negative weight
    predicted_weight = max(
        0.0,
        predicted_weight
    )

    # --------------------------------------------------------
    # STEP 6 — CALCULATE METRICS
    # --------------------------------------------------------

    confidence = estimate_confidence(
        predicted_weight
    )

    accuracy = estimate_accuracy(
        predicted_weight
    )

    lower = max(
        0.0,
        predicted_weight - RMSE
    )

    upper = (
        predicted_weight + RMSE
    )

    # --------------------------------------------------------
    # STEP 7 — RETURN RESULT
    # --------------------------------------------------------

    return {

        "valid": True,

        "cattle_detected": True,

        "animal": "cow",

        "detection_confidence_percent": round(
            detection["confidence"] * 100,
            1
        ),

        "predicted_weight_kg": round(
            predicted_weight,
            2
        ),

        "confidence_level_percent": round(
            confidence,
            1
        ),

        "estimated_accuracy_percent": round(
            accuracy,
            1
        ),

        "confidence_interval_kg": {

            "lower": round(
                lower,
                1
            ),

            "upper": round(
                upper,
                1
            )
        },

        "expected_error_kg": round(
            MAE,
            1
        ),

        "model_rmse_kg": round(
            RMSE,
            1
        ),

        "filename": file.filename,

        "detections": detection[
            "detections"
        ]
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {

        "status": "online",

        "message": (
            "Cattle Weight Prediction API"
        ),

        "cattle_detector_loaded": (
            cattle_detector is not None
        ),

        "weight_model_loaded": (
            weight_model is not None
        ),

        "cattle_detection_threshold": (
            CATTLE_DETECTION_THRESHOLD
        )
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )