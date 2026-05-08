from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import time

model = YOLO("yolov8n.pt")

# setting up cam
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

time.sleep(2)

while True:
    frame = picam2.capture_array()

    results = model(frame, imgsz=640, conf=0.4, verbose=False)

    annotated = results[0].plot()

    cv2.imshow("YOLO Camera", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()