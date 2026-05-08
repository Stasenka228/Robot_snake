import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# -----------------------------
# Settings
# -----------------------------
MODEL_PATH = "yolo11n.pt"   # change if you want another model
CONF_THRES = 0.35
IMG_W = 640
IMG_H = 480

TARGET_CLASSES = {
    "person",
    "chair",
    "bottle",
    "backpack",
    "box"
}

USE_METRIC_DISTANCE = False   # keep False for now until calibration is solid
FX = 529.0

CLASS_WIDTHS_M = {
    "bottle": 0.07,
    "chair": 0.45,
    "backpack": 0.30,
    "box": 0.20,
    "person": 0.45
}

CMD_FORWARD = "FORWARD"
CMD_LEFT = "LEFT"
CMD_RIGHT = "RIGHT"
CMD_HARD_LEFT = "HARD_LEFT"
CMD_HARD_RIGHT = "HARD_RIGHT"
CMD_STOP = "STOP"

# -----------------------------
# Load model
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# Pi Camera
# -----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (IMG_W, IMG_H), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

last_cmd = None

def estimate_distance_m(class_name, box_width_px):
    if box_width_px <= 1:
        return None
    known_width = CLASS_WIDTHS_M.get(class_name)
    if known_width is None:
        return None
    return (FX * known_width) / box_width_px

def choose_command(detections):
    if not detections:
        return CMD_FORWARD, None

    best = None
    best_score = -1

    for d in detections:
        centeredness = 1.0 - min(1.0, abs(d["offset"]))
        closeness = min(1.0, d["area_ratio"] * 8.0)
        score = 0.65 * centeredness + 0.35 * closeness

        if score > best_score:
            best_score = score
            best = d

    offset = best["offset"]
    area_ratio = best["area_ratio"]
    dist_m = best["dist_m"]

    # Keep the old behavior style because it was actually steering
    if USE_METRIC_DISTANCE and dist_m is not None:
        if dist_m < 0.25:
            return (CMD_HARD_RIGHT if offset < 0 else CMD_HARD_LEFT), best
        elif dist_m < 0.45:
            return (CMD_RIGHT if offset < 0 else CMD_LEFT), best
        else:
            return CMD_FORWARD, best

    # old area-based logic that was more responsive
    if area_ratio > 0.22 and abs(offset) < 0.20:
        return CMD_STOP, best
    elif area_ratio > 0.12:
        return (CMD_RIGHT if offset < 0 else CMD_LEFT), best
    else:
        return CMD_FORWARD, best

while True:
    frame = picam2.capture_array()

    # go back to tracking since this behaved better for you
    results = model.track(frame, persist=True, conf=CONF_THRES, verbose=False)
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

            offset = (cx - IMG_W / 2) / (IMG_W / 2)
            area_ratio = (bw * bh) / (IMG_W * IMG_H)

            dist_m = None
            if USE_METRIC_DISTANCE:
                dist_m = estimate_distance_m(class_name, bw)

            detections.append({
                "class_name": class_name,
                "conf": float(conf),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "offset": float(offset),
                "area_ratio": float(area_ratio),
                "dist_m": None if dist_m is None else float(dist_m),
            })

    cmd, best = choose_command(detections)

    if cmd != last_cmd:
        print(f"COMMAND: {cmd}")
        last_cmd = cmd

    for d in detections:
        x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])

        label = f'{d["class_name"]} {d["conf"]:.2f} off={d["offset"]:.2f} area={d["area_ratio"]:.3f}'
        if d["dist_m"] is not None:
            label += f' dist={d["dist_m"]:.2f}m'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # command text lower, no background box
    cv2.putText(
        frame, f"CMD: {cmd}", (20, IMG_H - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )

    # center guide line
    cv2.line(frame, (IMG_W // 2, 0), (IMG_W // 2, IMG_H), (255, 0, 0), 1)

    cv2.imshow("Snake Robot Obstacle Avoidance", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()