import cv2
from ultralytics import YOLO
import serial
import time
import math

ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=.5)

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

alpha = 40          # how strong the snake bends
omega = 1           # speed of the wave
delta = math.pi/4   # phase shift between segments
t = 0               # time variable for sine wave

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]   # get image size for normalization

    results = model(frame, conf=0.25, verbose=False)
    r = results[0]

    steering_offset = 0.0    # default: go straight
    distance_m = None        # not used rn

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()      # [x1,y1,x2,y2]
        clss = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names

        biggest_area = 0     # used to pick closest / most important object
        best_offset = 0

        for box, cls_id in zip(boxes, clss):
            class_name = names[cls_id]

            x1, y1, x2, y2 = box
            bw = x2 - x1                # box width
            bh = y2 - y1                # box height
            area = bw * bh              # approximate closeness

            cx = (x1 + x2) / 2          # center of object (x)

            # normalize: -1 (left) → 0 (center) → +1 (right)
            offset = (cx - frame_w/2) / (frame_w/2)

            # choose biggest object as main obstacle
            if area > biggest_area:
                biggest_area = area
                best_offset = offset

        steering_offset = best_offset   # final steering decision
    
    # Draw YOLO boxes so we can see what it is detecting
    if r.boxes is not None and len(r.boxes) > 0:
        for box, cls_id in zip(boxes, clss):
            class_name = names[cls_id]

            x1, y1, x2, y2 = map(int, box)

            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / 2
            offset = (cx - frame_w/2) / (frame_w/2)

            label = f"{class_name} off={offset:.2f}"

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

    beta = int(steering_offset * 40)    # convert offset → turning bias

    values = []

    for i in range(4):
        # snake wave + steering bias + center offset
        v = int(alpha * math.sin(omega*t + i*delta) + beta + 127)

        v = max(0, min(255, v))   # keep in valid byte range
        values.append(v)

    try:
        for v in values:
            ser.write(bytearray([v]))   # send each segment value

        # print("offset:", round(steering_offset, 2), "sent:", values)
        angle = steering_offset * 45   # maps -1..1 → -45°..45°
        print("offset:", round(steering_offset,2), "angle:", round(angle,1), "sent:", values)


    except Exception as e:
        print("UART error:", e)
    

    cv2.imshow("Snake Robot Vision", frame)   # show camera window

    key = cv2.waitKey(1) & 0xFF               # check keyboard press
    if key == 27 or key == ord('q'):          # ESC or q exits
        break

    t += 0.1                # advance wave
    time.sleep(0.1)         # control loop speed

cap.release()
cv2.destroyAllWindows()