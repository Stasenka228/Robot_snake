from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import torch
import sys

sys.path.append("/home/joste/robot_vision/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

print("all imports worked")