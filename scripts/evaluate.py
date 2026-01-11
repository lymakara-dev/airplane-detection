from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def find_latest_model(tag: str):
    candidates = list(MODELS_DIR.glob(f"{tag}_*_best.pt"))

    if not candidates:
        print(f"No trained model found for: {tag}")
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Using model: {latest.name}")
    return latest


def evaluate(tag: str):
    model_path = find_latest_model(tag)
    if model_path is None:
        return

    data_yaml = ROOT / "dataset" / f"data_{tag}.yaml"

    print(f"\n===== EVALUATING {tag.upper()} DATASET =====\n")

    model = YOLO(str(model_path))
    model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        device=0
    )


def main():
    evaluate("raw")
    evaluate("enhanced")


if __name__ == "__main__":
    main()
