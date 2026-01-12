#!/usr/bin/env python3
"""
Enhance small blurry aerial-object images (airplanes) for YOLO training.

AUTO PATH MODE (project structure):
Input:
  dataset/raw/images
  dataset/raw/labels

Output:
  dataset/enhanced/images
  dataset/enhanced/labels

Run:
  python scripts/enhance.py
  python scripts/enhance.py --limit 0
  python scripts/enhance.py --save-compare
"""

import argparse
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


# -------------------------------------------------
# Arguments (all optional now)
# -------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # Paths are optional, auto-detected if not set
    p.add_argument("--in-images", default=None, help="Input images folder (auto if not set)")
    p.add_argument("--in-labels", default=None, help="Input YOLO labels folder (.txt) (auto if not set)")
    p.add_argument("--out-images", default=None, help="Output enhanced images folder (auto if not set)")
    p.add_argument("--out-labels", default=None, help="Output labels folder (auto if not set)")

    p.add_argument("--ext", nargs="*", default=IMAGE_EXT_DEFAULT, help="Image extensions to include")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders")

    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N images (default 5). Use 0 for ALL.")

    p.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"),
                   help="Resize output images to W H (optional)")

    p.add_argument("--output-ext", default=None,
                   help="Force output extension (e.g. .jpg or .png). Default keeps original ext.")
    p.add_argument("--jpg-quality", type=int, default=95)

    p.add_argument("--clahe", action="store_true")
    p.add_argument("--clahe-clip", type=float, default=1.8)
    p.add_argument("--clahe-grid", type=int, default=8)

    p.add_argument("--sharpen", action="store_true")
    p.add_argument("--us-sigma", type=float, default=0.8)
    p.add_argument("--us-amount", type=float, default=0.25)
    p.add_argument("--us-threshold", type=int, default=0)

    p.add_argument("--denoise", action="store_true")
    p.add_argument("--dn-h", type=float, default=3.0)
    p.add_argument("--dn-hColor", type=float, default=3.0)
    p.add_argument("--dn-template", type=int, default=7)
    p.add_argument("--dn-search", type=int, default=21)

    p.add_argument("--save-compare", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    return p.parse_args()


# -------------------------------------------------
# Utils
# -------------------------------------------------
def ensure_dir(p: Path, dry_run: bool):
    if not dry_run:
        p.mkdir(parents=True, exist_ok=True)


def list_images(root: Path, exts, recursive: bool):
    exts = set(e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts)
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def safe_imread(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def safe_imwrite(path: Path, img_bgr, jpg_quality=95, dry_run=False):
    if dry_run:
        return True
    params = []
    suf = path.suffix.lower()
    if suf in [".jpg", ".jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    elif suf == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    return cv2.imwrite(str(path), img_bgr, params)


def copy_label(in_labels: Path, out_labels: Path, stem: str, dry_run=False):
    src = in_labels / f"{stem}.txt"
    dst = out_labels / f"{stem}.txt"
    if not src.exists():
        return False
    if not dry_run:
        shutil.copy2(src, dst)
    return True


# -------------------------------------------------
# Enhancement functions
# -------------------------------------------------
def apply_denoise_light(img_bgr, h, hColor, templateWindowSize, searchWindowSize):
    return cv2.fastNlMeansDenoisingColored(
        img_bgr, None,
        h=float(h), hColor=float(hColor),
        templateWindowSize=int(templateWindowSize),
        searchWindowSize=int(searchWindowSize),
    )


def apply_clahe_luminance(img_bgr, clipLimit=1.8, tileGridSize=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clipLimit),
                            tileGridSize=(int(tileGridSize), int(tileGridSize)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def unsharp_mask_mild(img_bgr, sigma=0.8, amount=0.25, threshold=0):
    blur = cv2.GaussianBlur(img_bgr, (0, 0), float(sigma))
    sharpened = cv2.addWeighted(img_bgr, 1.0 + float(amount), blur, -float(amount), 0)

    if threshold > 0:
        diff = cv2.absdiff(img_bgr, blur)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        mask = gray > int(threshold)
        out = img_bgr.copy()
        out[mask] = sharpened[mask]
        return out

    return sharpened


def enhance_small_blurry_object(img_bgr, args):
    out = img_bgr

    if args.denoise:
        out = apply_denoise_light(out, args.dn_h, args.dn_hColor,
                                  args.dn_template, args.dn_search)

    out = apply_clahe_luminance(out, clipLimit=args.clahe_clip,
                                tileGridSize=args.clahe_grid)

    out = unsharp_mask_mild(out, sigma=args.us_sigma,
                            amount=args.us_amount,
                            threshold=args.us_threshold)
    return out


def make_compare(original_bgr, enhanced_bgr, max_height=720):
    def resize_to_h(img, h):
        if img.shape[0] <= h:
            return img
        scale = h / img.shape[0]
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    o = resize_to_h(original_bgr, max_height)
    e = resize_to_h(enhanced_bgr, max_height)
    h = min(o.shape[0], e.shape[0])
    return np.hstack([o[:h], e[:h]])


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()

    ROOT = Path(__file__).resolve().parents[1]

    in_images = Path(args.in_images) if args.in_images else ROOT / "dataset/raw/images"
    in_labels = Path(args.in_labels) if args.in_labels else ROOT / "dataset/raw/labels"
    out_images = Path(args.out_images) if args.out_images else ROOT / "dataset/enhanced/images"
    out_labels = Path(args.out_labels) if args.out_labels else ROOT / "dataset/enhanced/labels"

    if not in_images.exists():
        print(f"ERROR: input images not found: {in_images}", file=sys.stderr)
        sys.exit(1)
    if not in_labels.exists():
        print(f"ERROR: input labels not found: {in_labels}", file=sys.stderr)
        sys.exit(1)

    ensure_dir(out_images, args.dry_run)
    ensure_dir(out_labels, args.dry_run)

    if not (args.clahe or args.sharpen or args.denoise):
        args.clahe = True
        args.sharpen = True

    images = sorted(list(list_images(in_images, args.ext, args.recursive)))
    if not images:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    if args.limit and args.limit > 0:
        images = images[: args.limit]

    processed = 0
    missing_labels = 0
    failed = 0

    for img_path in images:
        stem = img_path.stem

        out_ext = args.output_ext if args.output_ext else img_path.suffix
        if not out_ext.startswith("."):
            out_ext = "." + out_ext

        out_img_path = out_images / f"{stem}{out_ext}"

        img = safe_imread(img_path)
        if img is None:
            print(f"[FAIL] read {img_path}")
            failed += 1
            continue

        if args.resize:
            w, h = args.resize
            img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

        enhanced = enhance_small_blurry_object(img, args)

        if not safe_imwrite(out_img_path, enhanced, jpg_quality=args.jpg_quality, dry_run=args.dry_run):
            print(f"[FAIL] write {out_img_path}")
            failed += 1
            continue

        if args.save_compare:
            compare = make_compare(img, enhanced)
            compare_path = out_images / f"{stem}_compare.jpg"
            safe_imwrite(compare_path, compare, jpg_quality=92, dry_run=args.dry_run)

        if not copy_label(in_labels, out_labels, stem, dry_run=args.dry_run):
            missing_labels += 1
            print(f"[WARN] missing label for {stem}.txt")

        processed += 1
        print(f"[OK] {img_path.name}")

    print("\nDone")
    print(f"Processed: {processed}")
    print(f"Missing labels: {missing_labels}")
    print(f"Failed: {failed}")
    print(f"Output images: {out_images}")
    print(f"Output labels: {out_labels}")


if __name__ == "__main__":
    main()
