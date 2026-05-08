from picamera2 import Picamera2
import time
import cv2
import torch
import sys

sys.path.append("/home/joste/robot_vision/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()
time.sleep(2)

frame_rgb = picam2.capture_array()
frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
}

model = DepthAnythingV2(**model_configs["vits"])
model.load_state_dict(torch.load(
    "/home/joste/robot_vision/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth",
    map_location="cpu"
))
model.eval()

depth = model.infer_image(frame_bgr, 518)
print(depth.shape)

picam2.stop()