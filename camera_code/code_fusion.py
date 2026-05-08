import cv2
import time
import math
import serial
from ultralytics import YOLO

ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=.5)

MODEL_PATH = "yolo11n.pt"
CONF_THRES = 0.35
IMG_W = 320
IMG_H = 240

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

omega = 1
delta = math.pi / 4
t = 0

# Robot behavior thresholds
DANGER_AREA_RATIO = 0.18       # object fills 8% of frame = probably close
WARNING_AREA_RATIO = 0.035
EXPANSION_DANGER = 0.20        # bbox area grew 20% quickly = approaching
CENTER_WEIGHT = 0.55
SIZE_WEIGHT = 0.45

track_history = {}


def send_locomotion(command, t):
    if command == "FORWARD":
        alpha = 40
        beta = 90

    elif command == "LEFT":
        alpha = 40
        beta = 120

    elif command == "RIGHT":
        alpha = 40
        beta = 60

    elif command == "STOP":
        alpha = 0
        beta = 90

    else:
        alpha = 40
        beta = 90

    values = []

    for i in range(4):
        v = int(alpha * math.sin(omega * t + i * delta) + beta)
        v = max(0, min(255, v))
        values.append(v)

    try:
        for v in values:
            ser.write(bytearray([v]))
        print("CMD:", command, "Sent:", values)
    except Exception as e:
        print("UART error:", e)

# def get_command(offset, danger):


    """
    offset < 0 means object is left of center, so steer right.
    offset > 0 means object is right of center, so steer left.
    """
    # if danger < 0.78:
    #     return "FORWARD"

    # if offset < 0:
    #     return "RIGHT"
    # else:
    #     return "LEFT"

def get_command(offset, danger):
    if danger < 0.70:
        return "FORWARD"

    if offset < -0.20:
        return "RIGHT"
    elif offset > 0.20:
        return "LEFT"
    else:
        return "LEFT"


def choose_primary_obstacle(detections):
    if not detections:
        return None

    best = None
    best_score = -1

    for d in detections:
        centeredness = 1.0 - min(1.0, abs(d["offset"]))
        visual_closeness = min(1.0, d["area_ratio"] / DANGER_AREA_RATIO)

        score = CENTER_WEIGHT * centeredness + SIZE_WEIGHT * visual_closeness

        if score > best_score:
            best_score = score
            best = d

    return best


while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read camera frame")
        break

    now = time.time()
    frame_h, frame_w = frame.shape[:2]

    # Use track() so same object can be compared between frames
    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRES,
        verbose=False
    )

    r = results[0]
    detections = []

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id

        if ids is not None:
            ids = ids.cpu().numpy().astype(int)
        else:
            ids = [None] * len(boxes)

        names = r.names

        for box, conf, cls_id, track_id in zip(boxes, confs, clss, ids):
            class_name = names[cls_id]

            x1, y1, x2, y2 = box
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            offset = (cx - frame_w / 2.0) / (frame_w / 2.0)
            area_ratio = (bw * bh) / (frame_w * frame_h)

            # distance_m = estimate_distance_m(class_name, bw)

            expansion_rate = 0.0

            if track_id is not None:
                prev = track_history.get(track_id)

                if prev is not None:
                    dt = max(1e-6, now - prev["time"])
                    prev_area = max(1e-6, prev["area_ratio"])

                    # Positive means object is getting bigger / closer
                    expansion_rate = ((area_ratio - prev_area) / prev_area) / dt

                track_history[track_id] = {
                    "time": now,
                    "area_ratio": area_ratio
                }

            # Danger combines size and approach speed
            size_danger = min(1.0, area_ratio / DANGER_AREA_RATIO)
            approach_danger = min(1.0, max(0.0, expansion_rate) / EXPANSION_DANGER)

            danger = 0.75 * size_danger + 0.25 * approach_danger
            danger = min(1.0, danger)

            detections.append({
                "class_name": class_name,
                "conf": float(conf),
                "track_id": track_id,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "bw": float(bw),
                "bh": float(bh),
                "cx": float(cx),
                "cy": float(cy),
                "offset": float(offset),
                "area_ratio": float(area_ratio),
                # "distance_m": distance_m,
                "expansion_rate": float(expansion_rate),
                "danger": float(danger),
            })

    primary = choose_primary_obstacle(detections)

    steering_offset = 0.0
    danger = 0.0
    command = "FORWARD"

    if primary is not None:
        steering_offset = primary["offset"]
        danger = primary["danger"]
        command = get_command(steering_offset, danger)

        send_locomotion(command, t)
        t += 0.1
        time.sleep(0.1)

        print(
            f'CMD={command} '
            f'class={primary["class_name"]} '
            f'offset={steering_offset:.2f} '
            f'area={primary["area_ratio"]:.3f} '
            f'exp={primary["expansion_rate"]:.2f} '
            f'danger={danger:.2f} '
            # f'dist={primary["distance_m"]}'
        )

    # Draw all detections
    for d in detections:
        x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])

        label = (
            f'{d["class_name"]} {d["conf"]:.2f} '
            f'off={d["offset"]:.2f} '
            f'area={d["area_ratio"]:.3f} '
            f'danger={d["danger"]:.2f}'
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            2
        )

    # Highlight chosen obstacle
    if primary is not None:
        x1, y1, x2, y2 = map(int, [primary["x1"], primary["y1"], primary["x2"], primary["y2"]])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (255, 0, 0), 1)

    hud = f"CMD={command} offset={steering_offset:.2f} danger={danger:.2f}"
    cv2.putText(
        frame,
        hud,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Snake Robot Vision", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()