import torch
import cv2
import mediapipe as mp

print("Python OK")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("OpenCV version:", cv2.__version__)
print("MediaPipe version:", mp.__version__)
