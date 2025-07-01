from pathlib import Path
import zipfile
import shutil
import random
from tqdm import tqdm

def extract_zip_all(src: str, dst: str) -> bool:
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        for file in src_path.glob('*.zip'):
            target_dir = dst_path / file.stem
            target_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(file, 'r') as z:
                z.extractall(target_dir)

            print(f"Extracted {file.name} to {target_dir}")

        return True
    except Exception as e:
        print(f"[Error] {e}")

def auto_split(src, dst=None, train=0.8, val=0.2, test=0.0):
    assert abs((train + val + test) - 1.0) < 1e-6, "Train/val/test ratio must sum to 1.0"

    src = Path(src)
    dst = Path(dst) if dst else Path(".")

    image_dir = src / "images"
    label_dir = src / "labels"

    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * train)
    n_val = int(n * val)

    subsets = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:] if test > 0 else []
    }

    for split, files in subsets.items():
        for img_path in tqdm(files, desc=f"Copying {split}", leave=False):
            name = img_path.stem
            label_path = label_dir / f"{name}.txt"

            img_dst = dst / "images" / split
            lbl_dst = dst / "labels" / split
            img_dst.mkdir(parents=True, exist_ok=True)
            lbl_dst.mkdir(parents=True, exist_ok=True)

            shutil.copy(img_path, img_dst / img_path.name)

            if label_path.exists():
                shutil.copy(label_path, lbl_dst / label_path.name)
