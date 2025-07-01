import zipfile
from pathlib import Path

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
        return False
    
def extract_zip():
    pass