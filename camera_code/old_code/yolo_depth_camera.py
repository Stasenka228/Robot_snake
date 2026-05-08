from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import time
import numpy as np
import torch
import sys

# Add Depth Anything repo to Python path
sys.path.append("/home/joste/robot_vision/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2


# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Load Depth Anything V2 Small
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384]
    }
}

depth_model = DepthAnythingV2(**model_configs["vits"])
depth_model.load_state_dict(
    torch.load(
        "/home/joste/robot_vision/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth",
        map_location="cpu"
    )
)
depth_model = depth_model.to(DEVICE).eval()


# -----------------------------
# Set up camera
# -----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

time.sleep(2)


def get_median_depth(depth_map, x1, y1, x2, y2):
    h, w = depth_map.shape

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return float("nan")

    roi = depth_map[y1:y2, x1:x2]
    return float(np.median(roi))


def normalize_depth_for_display(depth):
    dmin = np.min(depth)
    dmax = np.max(depth)
    if dmax - dmin < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)
    depth_norm = (depth - dmin) / (dmax - dmin)
    return (depth_norm * 255).astype(np.uint8)


while True:
    # Picamera2 gives RGB frame
    frame_rgb = picam2.capture_array()

    # YOLO can use RGB directly
    results = model(frame_rgb, imgsz=320, conf=0.4, verbose=False)

    # Depth Anything usually works well with BGR image input
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    depth_map = depth_model.infer_image(frame_bgr, 224)

    # For display
    annotated = frame_bgr.copy()
    h, w, _ = annotated.shape
    image_center_x = w / 2.0

    cv2.line(annotated, (w // 2, 0), (w // 2, h), (0, 255, 255), 2)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls_id]

            x1, y1, x2, y2 = xyxy.tolist()

            bbox_center_x = (x1 + x2) / 2.0
            bbox_center_y = (y1 + y2) / 2.0
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            x_offset = bbox_center_x - image_center_x
            median_depth = get_median_depth(depth_map, x1, y1, x2, y2)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = (
                f"{class_name} {conf:.2f} "
                f"off={x_offset:.1f} "
                f"depth={median_depth:.3f}"
            )
            cv2.putText(
                annotated,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Print to terminal too
            print(
                f"class={class_name}, conf={conf:.2f}, "
                f"center=({bbox_center_x:.1f},{bbox_center_y:.1f}), "
                f"size=({bbox_width},{bbox_height}), "
                f"x_offset={x_offset:.1f}, median_depth={median_depth:.3f}"
            )

    # Depth visualization
    depth_vis = normalize_depth_for_display(depth_map)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    depth_vis = cv2.resize(depth_vis, (w, h))

    combined = np.hstack((annotated, depth_vis))
    cv2.imshow("YOLO + Depth", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()