# sharpen_all_images_unsharp.py
import cv2
import os
import numpy as np
from pathlib import Path
import shutil

# ===================== CONFIGURATION =====================
INPUT_IMAGES_DIR  = r"./dataset/raw/images"           # ← CHANGE THIS to your images folder
INPUT_LABELS_DIR  = r"./dataset/raw/labels"           # ← CHANGE THIS if you have labels
OUTPUT_IMAGES_DIR = r"./dataset/enhanced/images" # Where sharpened images will be saved
OUTPUT_LABELS_DIR = r"./dataset/enhanced/labels" # Labels will be copied here unchanged

# Sharpening parameters (tweak if needed)
amount      = 1.5      # strength of sharpening (1.0–2.5)
radius      = 1.0      # blur radius (0.5–2.0)
threshold   = 0        # only sharpen high-contrast edges (0 = sharpen everything)

# Create output folders
Path(OUTPUT_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_LABELS_DIR).mkdir(parents=True, exist_ok=True)

# ===================== SHARPEN FUNCTION =====================
def unsharp_mask(image, amount=1.5, radius=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharpened = image + amount * (image - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        sharpened[low_contrast_mask] = image[low_contrast_mask]
    
    return sharpened

# ===================== PROCESS ALL IMAGES =====================
image_extensions = ('.jpg', '.jpeg', '.png')

count = 0
for filename in os.listdir(INPUT_IMAGES_DIR):
    if filename.lower().endswith(image_extensions):
        input_path = os.path.join(INPUT_IMAGES_DIR, filename)
        img = cv2.imread(input_path)
        
        if img is None:
            print(f"Failed to read: {filename}")
            continue
        
        # Apply sharpening
        sharpened_img = unsharp_mask(img, amount=amount, radius=radius, threshold=threshold)
        
        # Save sharpened image
        output_path = os.path.join(OUTPUT_IMAGES_DIR, filename)
        cv2.imwrite(output_path, sharpened_img)
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} images...")
        
        # Copy corresponding label (if exists)
        label_name = os.path.splitext(filename)[0] + ".txt"
        src_label = os.path.join(INPUT_LABELS_DIR, label_name)
        dst_label = os.path.join(OUTPUT_LABELS_DIR, label_name)
        
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

print(f"\nDone! Sharpened {count} images.")
print(f"Saved to: {OUTPUT_IMAGES_DIR}")
print(f"Labels copied to: {OUTPUT_LABELS_DIR}")