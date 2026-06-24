import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import urllib.request
import os
print("=" * 55)
print(" YOLOv8 Object Detection — Starting...")
print("=" * 55)

model = YOLO("yolov8n.pt") # auto-downloads ~6MB on first run

print(f"\n✅ Model loaded! Can detect {len(model.names)} classes")
print(f" Sample classes: {list(model.names.values())[:8]}...\n")
sample_images = {
    "street_scene.jpg": "https://ultralytics.com/images/zidane.jpg",
    "bus_scene.jpg": "https://ultralytics.com/images/bus.jpg",
}

print("📥 Downloading sample images...")
for filename, url in sample_images.items():
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
        print(f" ✔ Downloaded: {filename}")
    else:
        print(f" ✔ Already exists: {filename}")

image_files = list(sample_images.keys())
np.random.seed(42)
colors = np.random.randint(100, 255, size=(80, 3), dtype=np.uint8)

CONFIDENCE = 0.4 # only show detections above 40% confidence

def detect_and_draw(image_path):
    """Run YOLO on image, draw bounding boxes, return annotated image."""

    # Run detection
    results = model(image_path, conf=CONFIDENCE, verbose=False)
    boxes = results[0].boxes

    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"\n📸 Processing: {image_path}")
    print(f" Objects detected: {len(boxes)}")
    print(f" {'Class':<15} {'Confidence':>12} Bounding Box")
    print(f" {'-'*55}")

    for box in boxes:
        conf = float(box.conf[0])
        if conf < CONFIDENCE:
            continue

        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        color = colors[cls_id % 80].tolist()

        # ── Draw bounding box rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)

        # ── Draw label background bar
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img,
                      (x1, y1 - th - 12),
                      (x1 + tw + 8, y1),
                      color, -1)

        # ── Draw label text
        cv2.putText(img, text,
                    (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

        print(f" {label:<15} {conf:>11.1%} "
              f"[{x1},{y1}] → [{x2},{y2}]")

    return img

print("\n🔍 Running detection...\n")

fig, axes = plt.subplots(1, len(image_files), figsize=(18, 8))
if len(image_files) == 1:
    axes = [axes]

for ax, image_path in zip(axes, image_files):
    annotated_img = detect_and_draw(image_path)
    ax.imshow(annotated_img)
    ax.set_title(image_path, fontsize=12, fontweight='bold', pad=10)
    ax.axis('off')

plt.suptitle(
    "YOLOv8 — Real Bounding Box Object Detection",
    fontsize=15, fontweight='bold', y=1.02
)
plt.tight_layout()


output_path = "yolo_output.png"
plt.savefig(output_path, bbox_inches='tight', dpi=150)
print(f"\n✅ Output saved as: {output_path}")


print("\n" + "=" * 55)
print(" FINAL DETECTION SUMMARY")
print("=" * 55)

for image_path in image_files:
    results = model(image_path, conf=CONFIDENCE, verbose=False)
    print(f"\n {image_path}")
    for box in results[0].boxes:
        if float(box.conf[0]) >= CONFIDENCE:
            cls = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f" ✔ {cls:<15} {conf:.1%}")

print("\n" + "=" * 55)
print(" ✅ DONE — Check yolo_output.png in your folder!")
print("=" * 55)

plt.show() # opens the visual window
