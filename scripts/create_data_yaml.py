import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

classes = [c.strip() for c in open(ROOT/"dataset/classes.txt") if c.strip()]

def make(name):
    data = {
        "path": "dataset/splits/"+name,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(classes),
        "names": classes
    }
    with open(ROOT/f"dataset/data_{name}.yaml","w") as f:
        yaml.dump(data,f,sort_keys=False)

make("raw")
make("enhanced")
print("YAML created")
