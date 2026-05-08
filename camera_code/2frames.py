import cv2
import time
import math
import serial
from ultralytics import YOLO

# =============================
# USER SETTINGS
# =============================
MODEL_PATH = "yolo11n.pt"
CAMERA_INDEX = 0
IMG_W = 640
IMG_H = 480
CONF_THRES = 0.35

# Calibrated focal length in pixels
FX = 229.0

ROBOT_SPEED_MPS = 0.05

# Minimum time between depth update
# Larger dt = more reliable parallax
MIN_DT_FOR_DEPTH = 0.20

# Only trust depth when bbox center shift is not tiny
MIN_PIXEL_SHIFT = 2.0

# Optional UART locomotion. Set False if you only want camera/testing
USE_SERIAL = True
SERIAL_PORT = "/dev/serial0"
BAUDRATE = 9600

# Navigation thresholds
DANGER_DISTANCE_M = 0.25
WARNING_DISTANCE_M = 0.50
OFFSET_DEADBAND = 0.20

# Limit classes to keep speed sane. Empty set means use all YOLO classes
TARGET_CLASSES = {
    "person", "chair", "bottle", "backpack", "cup", "book",
    "potted plant", "tv", "laptop", "keyboard", "mouse", "cell phone"
}

# =============================
# SETUP
# =============================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

ser = None
if USE_SERIAL:
    try:
        ser = serial.Serial(SERIAL_PORT, baudrate=BAUDRATE, timeout=0.5)
    except Exception as e:
        print("Serial disabled because:", e)
        ser = None

omega = 1.0
phase_delta = math.pi / 4
wave_t = 0.0

# track_id -> previous measurement
track_history = {}


def estimate_depth_from_two_frames(prev, curr, robot_speed_mps, fx):
    """
    Estimate object depth using two time-separated frames and known forward robot speed.

    Assumption: the camera moved approximately sideways/parallel relative to the image
    coordinate being used, so pixel displacement is usable as parallax.

    Z = fx * baseline / pixel_shift
    baseline = robot_speed * dt

    This is NOT reliable when the camera moves perfectly straight toward the object
    and the object center barely shifts. In that case use scale-change fallback below.
    """
    dt = curr["time"] - prev["time"]
    if dt < MIN_DT_FOR_DEPTH:
        return None, "dt too small"

    baseline_m = robot_speed_mps * dt
    du = curr["cx"] - prev["cx"]
    dv = curr["cy"] - prev["cy"]
    pixel_shift = math.sqrt(du * du + dv * dv)

    if pixel_shift < MIN_PIXEL_SHIFT:
        return None, "pixel shift too small"

    z_m = (fx * baseline_m) / pixel_shift
    return z_m, f"parallax dp={pixel_shift:.1f}px base={baseline_m:.3f}m dt={dt:.2f}s"


def estimate_depth_from_scale_change(prev, curr, robot_speed_mps):
    """
    Forward-motion fallback using bbox width growth.

    Pinhole model: w_px = fx * W_real / Z.
    For two frames, W_real and fx cancel:
        Z1 / Z2 = w2 / w1
    If the robot moved forward by baseline B, then Z2 = Z1 - B.
    Solving gives:
        Z1 = B * w2 / (w2 - w1)
        Z2 = Z1 - B
    Return current depth Z2.

    This works only if the same object is tracked and its bbox grows clearly.
    """
    dt = curr["time"] - prev["time"]
    if dt < MIN_DT_FOR_DEPTH:
        return None, "dt too small"

    baseline_m = robot_speed_mps * dt
    w1 = prev["bw"]
    w2 = curr["bw"]
    dw = w2 - w1

    if dw <= 1.0:
        return None, "bbox not growing"

    z_prev = baseline_m * w2 / dw
    z_curr = z_prev - baseline_m

    if z_curr <= 0 or z_curr > 10:
        return None, "scale result unrealistic"

    return z_curr, f"scale w1={w1:.1f}px w2={w2:.1f}px base={baseline_m:.3f}m dt={dt:.2f}s"


def send_locomotion(command, t):
    if command == "FORWARD":
        alpha, beta = 40, 90
    elif command == "LEFT":
        alpha, beta = 40, 120
    elif command == "RIGHT":
        alpha, beta = 40, 60
    else:  # STOP or fallback
        alpha, beta = 0, 90

    values = []
    for i in range(4):
        v = int(alpha * math.sin(omega * t + i * phase_delta) + beta)
        values.append(max(0, min(255, v)))

    if ser is not None:
        try:
            for v in values:
                ser.write(bytearray([v]))
        except Exception as e:
            print("UART error:", e)

    print("CMD:", command, "Sent:", values)


def choose_primary_obstacle(detections):
    if not detections:
        return None

    best = None
    best_score = -1.0
    for d in detections:
        centeredness = 1.0 - min(1.0, abs(d["offset"]))
        closeness = 0.0
        if d["distance_m"] is not None:
            closeness = max(0.0, min(1.0, (WARNING_DISTANCE_M - d["distance_m"]) / WARNING_DISTANCE_M))
        else:
            closeness = min(1.0, d["area_ratio"] * 8.0)
        score = 0.60 * centeredness + 0.40 * closeness
        if score > best_score:
            best_score = score
            best = d
    return best


def command_from_detection(d):
    if d is None:
        return "FORWARD"

    dist = d["distance_m"]
    offset = d["offset"]

    if dist is not None and dist > WARNING_DISTANCE_M:
        return "FORWARD"

    # If no trusted distance yet, use visual area as backup.
    if dist is None and d["area_ratio"] < 0.08:
        return "FORWARD"

    if offset < -OFFSET_DEADBAND:
        return "RIGHT"   # object is left, steer right
    if offset > OFFSET_DEADBAND:
        return "LEFT"    # object is right, steer left

    # Object is centered and close. Pick a default avoidance direction.
    return "LEFT"


while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read camera frame")
        break

    now = time.time()
    frame_h, frame_w = frame.shape[:2]

    results = model.track(frame, persist=True, conf=CONF_THRES, verbose=False)
    r = results[0]
    detections = []

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(boxes)
        names = r.names

        for box, conf, cls_id, track_id in zip(boxes, confs, clss, ids):
            class_name = names[cls_id]
            if TARGET_CLASSES and class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            offset = (cx - frame_w / 2.0) / (frame_w / 2.0)
            area_ratio = (bw * bh) / (frame_w * frame_h)

            curr = {"time": now, "cx": cx, "cy": cy, "bw": bw, "bh": bh}
            distance_m = None
            method = "no previous track"

            if track_id is not None and track_id in track_history:
                prev = track_history[track_id]

                # Try parallax first, then bbox scale-change fallback.
                distance_m, method = estimate_depth_from_two_frames(prev, curr, ROBOT_SPEED_MPS, FX)
                if distance_m is None:
                    distance_m, method = estimate_depth_from_scale_change(prev, curr, ROBOT_SPEED_MPS)

            if track_id is not None:
                track_history[track_id] = curr

            detections.append({
                "class_name": class_name,
                "conf": float(conf),
                "track_id": None if track_id is None else int(track_id),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "bw": float(bw), "bh": float(bh), "cx": float(cx), "cy": float(cy),
                "offset": float(offset),
                "area_ratio": float(area_ratio),
                "distance_m": None if distance_m is None else float(distance_m),
                "method": method,
            })

    primary = choose_primary_obstacle(detections)
    command = command_from_detection(primary)
    send_locomotion(command, wave_t)
    wave_t += 0.1

    if primary is not None:
        dist_txt = "None" if primary["distance_m"] is None else f'{primary["distance_m"]:.2f}m'
        print(
            f'primary id={primary["track_id"]} class={primary["class_name"]} '
            f'off={primary["offset"]:.2f} area={primary["area_ratio"]:.3f} '
            f'dist={dist_txt} method={primary["method"]}'
        )

    # Draw detections
    for d in detections:
        x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])
        label = f'{d["class_name"]} id={d["track_id"]} off={d["offset"]:.2f}'
        if d["distance_m"] is not None:
            label += f' Z={d["distance_m"]:.2f}m'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    if primary is not None:
        x1, y1, x2, y2 = map(int, [primary["x1"], primary["y1"], primary["x2"], primary["y2"]])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (255, 0, 0), 1)
    cv2.putText(frame, f"CMD={command}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Two-frame YOLO depth", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
if ser is not None:
    ser.close()
cv2.destroyAllWindows()
