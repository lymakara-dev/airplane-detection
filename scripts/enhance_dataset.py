import cv2, os, shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------
# Fast image enhancement using multithreading
# raw -> enhanced
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

IN_IMG = ROOT / "dataset/raw/images"
IN_LBL = ROOT / "dataset/raw/labels"
OUT_IMG = ROOT / "dataset/enhanced/images"
OUT_LBL = ROOT / "dataset/enhanced/labels"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

# ---------- Pre-create reusable objects ----------
clahe = cv2.createCLAHE(2.0, (8, 8))

def gamma_corr(img, gamma=0.8):
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def unsharp(img, amount=1.4, radius=1.0, threshold=10):
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    sharp = img + amount * (img - blur)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    if threshold > 0:
        mask = np.abs(img - blur) < threshold
        sharp[mask] = img[mask]
    return sharp


# ---------- Worker function ----------
def process_image(filename):

    img_path = IN_IMG / filename
    lbl_path = IN_LBL / (Path(filename).stem + ".txt")

    img = cv2.imread(str(img_path))
    if img is None:
        return False

    # Noise removal
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # CLAHE contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Brightness fix
    img = gamma_corr(img, 0.8)

    # Sharpen edges
    img = unsharp(img)

    cv2.imwrite(str(OUT_IMG / filename), img)

    if lbl_path.exists():
        shutil.copy(lbl_path, OUT_LBL / lbl_path.name)

    return True


# ---------- Parallel processing ----------
files = [f for f in os.listdir(IN_IMG) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
total = len(files)

print(f"Enhancing {total} images using parallel workers...")

done = 0

# workers = number of CPU cores (auto)
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = [executor.submit(process_image, f) for f in files]

    for i, future in enumerate(as_completed(futures), start=1):
        if future.result():
            done += 1

        if i % 100 == 0 or i == total:
            print(f"Processed {i}/{total}")

print(f"\nEnhancement complete: {done} images saved")
print(f"Output images: {OUT_IMG}")
