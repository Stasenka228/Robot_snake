import cv2
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from ultralytics import YOLO
import serial
import time
import math

# ---------------------------------------------------
# SERIAL / SNAKE SETTINGS
# ---------------------------------------------------

ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=0.5)

NUM_SERVOS = 6

CENTER = 118
ALPHA = 80
OMEGA = 2.0
DELTA = math.pi / 4

DT = 0.05

# ---------------------------------------------------
# YOLO SETTINGS
# ---------------------------------------------------

MODEL_PATH = "yolo11n.pt"

CONF_THRES = 0.3

IMG_W = 640
IMG_H = 480

STREAM_PORT = 5800

TARGET_CLASSES = {
    "person",
    "chair",
    "bottle",
    "backpack",
    "box",
    "cup",
    "book"
}

FX = 529.0

CLASS_WIDTHS_M = {
    "bottle": 0.07,
    "chair": 0.45,
    "backpack": 0.30,
    "box": 0.20,
    "person": 0.45,
    "cup": 0.08,
    "book": 0.15
}

# ---------------------------------------------------
# MJPEG STREAM SERVER
# ---------------------------------------------------

output_frame = None
lock = threading.Lock()

class StreamHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        self.send_response(200)

        self.send_header(
            'Content-type',
            'multipart/x-mixed-replace; boundary=frame'
        )

        self.end_headers()

        try:

            while True:

                with lock:

                    if output_frame is None:
                        continue

                    _, encoded = cv2.imencode(
                        '.jpg',
                        output_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )

                    data = encoded.tobytes()

                self.wfile.write(b'--frame\r\n')

                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', len(data))

                self.end_headers()

                self.wfile.write(data)
                self.wfile.write(b'\r\n')

        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass


def start_stream_server():

    server = HTTPServer(('0.0.0.0', STREAM_PORT), StreamHandler)

    server.serve_forever()


threading.Thread(
    target=start_stream_server,
    daemon=True
).start()

print("STREAM STARTED")

# CHANGE THIS TO YOUR PI IP
print(f"http://172.31.102.137:{STREAM_PORT}")

# ---------------------------------------------------
# LOAD YOLO
# ---------------------------------------------------

model = YOLO(MODEL_PATH)

# ---------------------------------------------------
# CAMERA
# ---------------------------------------------------

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

# ---------------------------------------------------
# DISTANCE ESTIMATION
# ---------------------------------------------------

def estimate_distance_m(class_name, box_width_px):

    if box_width_px <= 1:
        return None

    real_width = CLASS_WIDTHS_M.get(class_name)

    if real_width is None:
        return None

    return (FX * real_width) / box_width_px

# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------

t = 0

try:

    while True:

        # ---------------------------------------------
        # SNAKE LOCOMOTION
        # ---------------------------------------------

        values = []

        for i in range(NUM_SERVOS):

            v = int(
                CENTER
                - ALPHA * math.sin(OMEGA * t - i * DELTA)
            )

            v = max(0, min(255, v))

            values.append(v)

        for v in values:
            ser.write(bytearray([v]))

        t += DT

        # ---------------------------------------------
        # CAMERA
        # ---------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("Camera failed")
            break

        frame_h, frame_w = frame.shape[:2]

        # ---------------------------------------------
        # YOLO
        # ---------------------------------------------

        results = model(frame, conf=CONF_THRES, verbose=False)

        r = results[0]

        primary = None
        biggest_area = 0

        if r.boxes is not None and len(r.boxes) > 0:

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy().astype(int)

            names = r.names

            for box, conf, cls_id in zip(boxes, confs, clss):

                class_name = names[cls_id]

                if class_name not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = box

                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)

                area = bw * bh

                # choose biggest object
                if area > biggest_area:

                    biggest_area = area

                    cx = (x1 + x2) / 2.0

                    offset = (
                        (cx - frame_w / 2.0)
                        / (frame_w / 2.0)
                    )

                    distance_m = estimate_distance_m(
                        class_name,
                        bw
                    )

                    primary = {
                        "class": class_name,
                        "distance": distance_m,
                        "offset": offset,
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }

        # ---------------------------------------------
        # DISPLAY
        # ---------------------------------------------

        if primary is not None:

            x1 = primary["x1"]
            y1 = primary["y1"]
            x2 = primary["x2"]
            y2 = primary["y2"]

            offset = primary["offset"]

            if offset < -0.15:
                direction = "LEFT"

            elif offset > 0.15:
                direction = "RIGHT"

            else:
                direction = "CENTER"

            distance_text = "???"

            if primary["distance"] is not None:
                distance_text = f'{primary["distance"]:.2f} m'

            label1 = primary["class"].upper()
            label2 = direction
            label3 = distance_text

            print(
                label1,
                label2,
                distance_text
            )

            # red box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            # simple text
            cv2.putText(
                frame,
                label1,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label2,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label3,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # center line
        cv2.line(
            frame,
            (frame_w // 2, 0),
            (frame_w // 2, frame_h),
            (255, 0, 0),
            1
        )

        # ---------------------------------------------
        # SEND FRAME TO BROWSER
        # ---------------------------------------------

        with lock:
            output_frame = frame.copy()

        time.sleep(DT)

except KeyboardInterrupt:

    print("STOPPED")

    # center snake

    for _ in range(20):

        for i in range(NUM_SERVOS):
            ser.write(bytearray([CENTER]))

        time.sleep(0.05)

finally:

    cap.release()

    ser.close()