from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Image to test
image_path = "human&cows.jpg"

print("=" * 50)
print("RUNNING CATTLE DETECTION")
print("=" * 50)

results = model(image_path, verbose=False)

cattle_found = False

for result in results:

    if result.boxes is None:
        print("No objects detected.")
        continue

    for box in result.boxes:

        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        class_name = model.names[class_id]

        print(
            f"Detected: {class_name} "
            f"| Confidence: {confidence:.2f}"
        )

        # Class 19 = cow
        if class_id == 19:
            cattle_found = True

print("=" * 50)

if cattle_found:
    print("RESULT: CATTLE DETECTED")
else:
    print("RESULT: NO CATTLE DETECTED")

print("=" * 50)