import cv2
from ultralytics import YOLO

# -----------------------------
# Settings
# -----------------------------
MODEL_PATH = "yolo11n.pt"  
CONF_THRES = 0.3
IMG_W = 640
IMG_H = 480

TARGET_CLASSES = {
    "person",
    "chair",
    "bottle",
    "backpack",
    "box",
    "cup",
    "book",
    "potted plant",
    "tv",
    "laptop",
    "keyboard",
    "mouse"
}

USE_METRIC_DISTANCE = True

# Replace after calibration
FX = 529.0

# Approximate real widths in meters
CLASS_WIDTHS_M = {
    "bottle": 0.07,
    "chair": 0.45,
    "backpack": 0.30,
    "box": 0.20,
    "person": 0.45
}

# -----------------------------
# Load model
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# OpenCV Camera
# -----------------------------
cap = cv2.VideoCapture(0)

# optional: try to set size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)


def estimate_distance_m(class_name, box_width_px):
    if box_width_px <= 1:
        return None
    real_width = CLASS_WIDTHS_M.get(class_name)
    if real_width is None:
        return None
    return (FX * real_width) / box_width_px


def choose_primary_obstacle(detections):
    """
    Pick the obstacle that is both central and visually large
    """
    if not detections:
        return None

    best = None
    best_score = -1.0

    for d in detections:
        centeredness = 1.0 - min(1.0, abs(d["offset"]))
        closeness = min(1.0, d["area_ratio"] * 8.0)
        score = 0.65 * centeredness + 0.35 * closeness

        if score > best_score:
            best_score = score
            best = d

    return best


while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame from camera")
        break

    # if actual camera size differs, use real frame shape
    frame_h, frame_w = frame.shape[:2]

    # keep track() if it worked better for you
    # results = model.track(frame, persist=True, conf=CONF_THRES, verbose=False)
    results = model(frame, conf=CONF_THRES, verbose=False)
    r = results[0]

    detections = []

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names

        for box, conf, cls_id in zip(boxes, confs, clss):
            class_name = names[cls_id]

            if class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # steering offset: -1 left, +1 right
            offset = (cx - frame_w / 2.0) / (frame_w / 2.0)

            # relative closeness from box area
            area_ratio = (bw * bh) / (frame_w * frame_h)

            distance_m = None
            if USE_METRIC_DISTANCE:
                distance_m = estimate_distance_m(class_name, bw)

            detections.append({
                "class_name": class_name,
                "conf": float(conf),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "cx": float(cx),
                "cy": float(cy),
                "bw": float(bw),
                "bh": float(bh),
                "offset": float(offset),
                "area_ratio": float(area_ratio),
                "distance_m": None if distance_m is None else float(distance_m),
            })

    primary = choose_primary_obstacle(detections)

    # Defaults when nothing important detected
    steering_offset = 0.0
    danger = 0.0
    distance_m = None

    if primary is not None:
        steering_offset = primary["offset"]
        danger = min(1.0, primary["area_ratio"] * 8.0)
        distance_m = primary["distance_m"]

        print(
            f'offset={steering_offset:.3f}, '
            f'danger={danger:.3f}, '
            f'distance={distance_m}, '
            f'class={primary["class_name"]}'
        )

    # Draw detections
    for d in detections:
        x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])

        label = f'{d["class_name"]} {d["conf"]:.2f} off={d["offset"]:.2f} area={d["area_ratio"]:.3f}'
        if d["distance_m"] is not None:
            label += f' dist={d["distance_m"]:.2f}m'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Highlight chosen obstacle
    if primary is not None:
        x1, y1, x2, y2 = map(int, [primary["x1"], primary["y1"], primary["x2"], primary["y2"]])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        info = f'offset={steering_offset:.2f} danger={danger:.2f}'
        if distance_m is not None:
            info += f' dist={distance_m:.2f}m'

        cv2.putText(
            frame,
            info,
            (20, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # Center line
    cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (255, 0, 0), 1)

    cv2.imshow("Snake Robot Vision", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()