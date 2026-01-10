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

    # 1. Split dataset
    run([
        PYTHON,
        str(SCRIPTS / "split_dataset.py"),
        "--datapath", str(ROOT / "dataset"),
        "--train_pct", "0.8"
    ])

    # 2. Create data.yaml
    run([
        PYTHON,
        str(SCRIPTS / "create_data_yaml.py")
    ])

    # 3. Train model
    run([
        PYTHON,
        str(SCRIPTS / "train.py")
    ])

    print("\nALL STEPS COMPLETED SUCCESSFULLY\n")


if __name__ == "__main__":
    main()
