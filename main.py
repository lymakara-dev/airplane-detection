import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S = ROOT / "scripts"
PY = sys.executable


def run(cmd):
    print("\nRUNNING:", " ".join(map(str, cmd)))
    if subprocess.run(cmd).returncode != 0:
        print("STOPPED DUE TO ERROR")
        exit(1)


def main():

    run([PY, S / "enhance_dataset.py"])
    run([PY, S / "split_dataset.py"])
    run([PY, S / "create_data_yaml.py"])
    run([PY, S / "train.py"])
    run([PY, S / "evaluate.py"])

    print("\nALL PIPELINE STEPS COMPLETED SUCCESSFULLY\n")


if __name__ == "__main__":
    main()
