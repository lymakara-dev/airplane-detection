from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import shutil

def main():
    # Resolve project root
    root = Path(__file__).resolve().parents[1]

    # Dataset config
    data_yaml = root / "dataset" / "data.yaml"
    runs_dir = root / "runs" / "detect"
    model_dir = root / "models"

    # Create model folder if missing
    model_dir.mkdir(exist_ok=True)

    # Date-based run name
    date_str = datetime.now().strftime("%Y_%m_%d")
    run_name = f"train_{date_str}"

    # Load pretrained YOLOv8 model
    model = YOLO("yolov8s.pt")

    # Start training
    model.train(
        data=str(data_yaml),
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        project=str(runs_dir),
        name=run_name,
        pretrained=True,
        val=True    
    )

    # Copy best model to model folder
    best_model = runs_dir / run_name / "weights" / "best.pt"

    if best_model.exists():
        target = model_dir / f"{run_name}_best.pt"
        shutil.copy(best_model, target)
        print(f"Saved final model to: {target}")
    else:
        print("best.pt not found. Check training logs.")

if __name__ == "__main__":
    main()
