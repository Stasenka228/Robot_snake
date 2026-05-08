import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # or small pretrained model
cap = cv2.VideoCapture(0)

IMG_W = 640
IMG_H = 480

# example camera-based constants
FX = 700.0           # replace with calibrated fx in pixels
KNOWN_WIDTH_M = 0.10 # width of target object in meters

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.track(frame, persist=True, conf=0.35, verbose=False)
    r = results[0]

    best_cmd = ("forward", 0.0)

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        best_score = -1

        for box, cls_id, conf in zip(boxes, clss, confs):
            x1, y1, x2, y2 = box
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0

            offset = (cx - IMG_W / 2) / (IMG_W / 2)      # -1 left, +1 right
            area_ratio = (bw * bh) / (IMG_W * IMG_H)

            # approximate distance only for known-size targets
            dist_m = (FX * KNOWN_WIDTH_M) / bw

            # combine closeness + center danger
            centeredness = 1.0 - min(1.0, abs(offset))
            score = 0.6 * centeredness + 0.4 * min(1.0, area_ratio * 8)

            if score > best_score:
                best_score = score

                if dist_m < 0.25:
                    if offset < 0:
                        best_cmd = ("hard_right", dist_m)
                    else:
                        best_cmd = ("hard_left", dist_m)
                elif dist_m < 0.45:
                    if offset < 0:
                        best_cmd = ("right", dist_m)
                    else:
                        best_cmd = ("left", dist_m)
                else:
                    best_cmd = ("forward", dist_m)

    cmd, dist = best_cmd
    print(f"cmd={cmd}, dist={dist:.2f} m")