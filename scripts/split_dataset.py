from pathlib import Path
import random, shutil

def split(source, target, train=0.8, val=0.1):
    imgs = list((source/"images").glob("*"))
    random.shuffle(imgs)

    n = len(imgs)
    t = int(n*train)
    v = int(n*val)

    parts = {
        "train": imgs[:t],
        "val": imgs[t:t+v],
        "test": imgs[t+v:]
    }

    for k, files in parts.items():
        (target/k/"images").mkdir(parents=True, exist_ok=True)
        (target/k/"labels").mkdir(parents=True, exist_ok=True)

        for p in files:
            shutil.copy(p, target/k/"images"/p.name)
            lbl = source/"labels"/(p.stem+".txt")
            if lbl.exists():
                shutil.copy(lbl, target/k/"labels"/lbl.name)

ROOT = Path(__file__).resolve().parents[1]

split(ROOT/"dataset/raw", ROOT/"dataset/splits/raw")
split(ROOT/"dataset/enhanced", ROOT/"dataset/splits/enhanced")

print("Splitting done")
