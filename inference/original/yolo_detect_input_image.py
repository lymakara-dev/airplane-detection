import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
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
    model_dir = 'models/original'
    model_files = glob.glob(os.path.join(model_dir, '*.pt'))

    if not model_files:
        print('ERROR: No model found in models/ folder.')
        sys.exit(1)

    model_files.sort(key=os.path.getmtime, reverse=True)
    model_path = model_files[0]

    print(f'Using latest model: {model_path}')

MODEL_PATH = model_path           
CONF_THRESHOLD = 0.45
SOURCE = "./test_images/"                 # folder or single image path
# SOURCE = "bus.jpg"                    # ← or single image
SAVE_RESULTS = True                     # save output images?
OUTPUT_FOLDER = "runs/detect/images/"
# ==========================================

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    print(f"Loaded model: {os.path.basename(MODEL_PATH)}")
    print(f"Classes: {list(model.names.values())[:8]}{'...' if len(model.names)>8 else ''}")

    # Prepare source
    if os.path.isdir(SOURCE):
        files = sorted(glob.glob(os.path.join(SOURCE, "*.[jpJP][pnPN][gG]")))
        print(f"Found {len(files)} images in folder")
    else:
        files = [SOURCE] if os.path.isfile(SOURCE) else []
        if not files:
            print("Invalid source! (not file or folder)")
            return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for i, img_path in enumerate(files, 1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Cannot read image: {img_path}")
            continue

        print(f"\n[{i}/{len(files)}] Processing: {os.path.basename(img_path)}")

        # Inference
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        # Draw results
        annotated = results[0].plot()  # ← ultralytics nice built-in visualization

        # Show
        cv2.imshow("YOLO Detection - Image Mode", annotated)

        if SAVE_RESULTS:
            save_path = os.path.join(OUTPUT_FOLDER, f"result_{i:03d}_{os.path.basename(img_path)}")
            cv2.imwrite(save_path, annotated)
            print(f"Saved → {save_path}")

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.waitKey(0)  # extra pause

    print("\nFinished processing all images.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()