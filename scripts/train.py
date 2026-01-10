from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import shutil
import argparse
import sys
import os

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to data folder containing image and annotation files',
                    required=True)
parser.add_argument('--modelType', help='Path to data folder containing image and annotation files',
                    required=True)

args = parser.parse_args()

data_path = args.datapath
model_type = args.modelType

# Check for valid entries
if not os.path.isdir(data_path):
   print('Directory specified by --datapath not found. Verify the path is correct (and uses double back slashes if on Windows) and try again.')
   sys.exit(0)

def main():
    # Resolve project root
    root = Path(__file__).resolve().parents[1]

    # Dataset config
    data_yaml = root / data_path / "data.yaml"
    runs_dir = root / "runs" / model_type /"detect"
    model_dir = root / model_type

    # Create model folder if missing
    model_dir.mkdir(exist_ok=True)

    # Date-based run name
    date_str = datetime.now().strftime("%Y_%m_%d")
    run_name = f"train_{date_str}"

    # Load pretrained YOLOv11 model
    model = YOLO("yolo11s.pt")

    # Start training
    model.train(
        data=str(data_yaml),
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
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
