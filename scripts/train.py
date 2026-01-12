from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import shutil
import re


def get_next_index(model_dir: Path, tag: str, date_str: str):
    pattern = re.compile(rf"{tag}_{date_str}_(\d{{3}})_best\.pt")
    nums = []

    for f in model_dir.glob(f"{tag}_{date_str}_*_best.pt"):
        m = pattern.match(f.name)
        if m:
            nums.append(int(m.group(1)))

    if nums:
        return f"{max(nums) + 1:03d}"
    else:
        return "001"


def train_one(tag: str, data_yaml: Path):
    # Resolve project root
    root = Path(__file__).resolve().parents[1]

    runs_dir = root / "runs" / "detect"
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)

    # Date-based run name + auto number
    date_str = datetime.now().strftime("%Y_%m_%d")
    run_idx = get_next_index(model_dir, tag, date_str)
    run_name = f"{tag}_{date_str}_{run_idx}"

    print(f"\n===== TRAINING {tag.upper()} DATASET =====\n")

    # Load pretrained YOLOv11 model
    model = YOLO(str(root / "yolo11s.pt"))

    # Start training
    model.train(
        data=str(data_yaml),
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        workers=12,
        project=str(runs_dir),
        name=run_name,
        pretrained=True,
        val=True
    )

    # Copy best model to models folder
    best_model = runs_dir / run_name / "weights" / "best.pt"

    if best_model.exists():
        target = model_dir / f"{tag}_{date_str}_{run_idx}_best.pt"
        shutil.copy(best_model, target)
        print(f"Saved final model to: {target}")
    else:
        print("best.pt not found. Check training logs.")


def main():
    root = Path(__file__).resolve().parents[1]

    # Train on RAW dataset
    train_one(
        tag="raw",
        data_yaml=root / "dataset" / "data_raw.yaml"
    )

    # Train on ENHANCED dataset
    train_one(
        tag="enhanced",
        data_yaml=root / "dataset" / "data_enhanced.yaml"
    )


if __name__ == "__main__":
    main()
