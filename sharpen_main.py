import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

PYTHON = sys.executable


def run(cmd):
    print(f"\nRUNNING: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("STOPPED DUE TO ERROR")
        sys.exit(1)


def main():

    # 1. Create data.yaml
    run([
        PYTHON,
        str(SCRIPTS / "sharpen_all_images_unsharp.py")
    ])

    # 2. Split dataset
    run([
        PYTHON,
        str(SCRIPTS / "split_dataset.py"),
        "--datapath", str(ROOT / "dataset/sharpen"),
        "--train_pct", "0.8"
    ])

    # 3. Create data.yaml
    run([
        PYTHON,
        str(SCRIPTS / "create_data_yaml.py"),
        "--datapath", str(ROOT / "dataset/sharpen"),
    ])

    # 4. Train model
    run([
        PYTHON,
        str(SCRIPTS / "train.py"),
        "--datapath", str(ROOT / "dataset/sharpen"),
        "--modelType", str(ROOT / "models/sharpen"),
    ])

    print("\nALL STEPS COMPLETED SUCCESSFULLY\n")


if __name__ == "__main__":
    main()
