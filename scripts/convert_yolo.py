from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = YOLO(str(BASE_DIR / "yolo11n.pt"))

model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    dynamic=False,
    simplify=True,
    nms=False,
)
