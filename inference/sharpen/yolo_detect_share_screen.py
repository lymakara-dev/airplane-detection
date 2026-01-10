import time
import cv2
import numpy as np
from ultralytics import YOLO
import mss                   # pip install mss
import os
import argparse
import glob
import sys

# ================= CONFIG =================
# Define and parse user input arguments

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model',
    help='Path to YOLO model file. If not set, use latest model in models/',
    default=None
)

parser.add_argument(
    '--source',
    help='Image source. Default: usb0',
    default='usb0'
)

parser.add_argument(
    '--thresh',
    help='Minimum confidence threshold',
    type=float,
    default=0.5
)

parser.add_argument(
    '--resolution',
    help='Resolution WxH. Default: 1280x720',
    default='1280x720'
)

parser.add_argument(
    '--record',
    help='Record output video',
    action='store_true'
)

args = parser.parse_args()

# Model selection logic
if args.model:
    model_path = args.model
else:
    model_dir = 'models/sharpen'
    model_files = glob.glob(os.path.join(model_dir, '*.pt'))

    if not model_files:
        print('ERROR: No model found in models/ folder.')
        sys.exit(1)

    model_files.sort(key=os.path.getmtime, reverse=True)
    model_path = model_files[0]

    print(f'Using latest model: {model_path}')

MODEL_PATH = model_path           
CONF_THRESHOLD = 0.4
FPS_TARGET = 12              # try to limit cpu usage

# Which monitor/screen to capture (0 = main monitor)
MONITOR_NUMBER = 0

# # Good for most laptops / focused work area
# REGION = {"top": 80, "left": 0, "width": 1440, "height": 900}

# # Quite small – good for testing / lower CPU usage
# REGION = {"top": 140, "left": 280, "width": 1024, "height": 768}

# 1080p centered-ish (assuming 1920×1080 screen)
REGION = {"top": 0, "left": 0, "width": 920, "height": 1080}

WINDOW_NAME = "YOLO Screen Detection (Press ESC or q to quit)"
# ==========================================

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found → {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    print(f"Loaded: {os.path.basename(MODEL_PATH)}")

    sct = mss.mss()
    monitors = sct.monitors

    if MONITOR_NUMBER >= len(monitors):
        print(f"Monitor {MONITOR_NUMBER} not found! Available: 0–{len(monitors)-1}")
        return

    monitor = monitors[MONITOR_NUMBER] if REGION is None else REGION

    print(f"Capturing monitor {MONITOR_NUMBER} ({monitor['width']}×{monitor['height']})")
    print("Press ESC / q to quit\n")

    prev_time = time.time()

    while True:
        # Screen capture
        screenshot = np.array(sct.grab(monitor))
        # Convert BGRA → BGR
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # Inference
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        # Draw nice results (ultralytics built-in)
        annotated = results[0].plot()

        # FPS calculation
        now = time.time()
        fps = 1 / (now - prev_time + 1e-8)
        prev_time = now

        cv2.putText(annotated, f"FPS: {fps:.1f}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, annotated)

        # Control FPS + exit
        if FPS_TARGET > 0:
            delay = max(1, int(1000 / FPS_TARGET - (time.time() - now) * 1000))
            key = cv2.waitKey(delay) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # q or ESC
            break

        # Optional: keyboard library way (cleaner exit)
        # if keyboard.is_pressed('esc'):
        #     break

    print("Exiting...")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")