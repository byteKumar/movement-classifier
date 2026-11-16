"""Download the UCI HAPT dataset and extract the raw recordings into data/hapt/.

Usage: python scripts/download_data.py
"""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/341/"
    "smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
)
SHA256 = "4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09"
ROOT = Path(__file__).resolve().parent.parent / "data"
KEEP = ("RawData/", "activity_labels.txt", "README.txt",
        "Test/subject_id_test.txt", "Train/subject_id_train.txt")


def main() -> None:
    zip_path = ROOT / "download" / "hapt.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zip_path)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch: {digest}")
    out = ROOT / "hapt"
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.startswith(KEEP)]
        for m in members:
            # refuse anything that would escape the output folder
            if not (out / m).resolve().is_relative_to(out.resolve()):
                raise SystemExit(f"unsafe path in zip: {m}")
        z.extractall(out, members)
    print(f"extracted {len(members)} files to {out}")


if __name__ == "__main__":
    main()
