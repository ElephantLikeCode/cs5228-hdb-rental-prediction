"""Import only checksum-verified course CSVs from the official Kaggle ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

try:
    from .check_setup import RAW_SHA256
except ImportError:
    from check_setup import RAW_SHA256

ROOT = Path(__file__).resolve().parents[1]


def import_archive(archive: Path, destination: Path):
    required = {k: v for k, v in RAW_SHA256.items() if k != "example-submission.csv"}
    verified = {}
    with ZipFile(archive) as source:
        for relative, expected in required.items():
            matches = [info for info in source.infolist()
                       if info.filename == relative or info.filename.endswith("/" + relative)]
            if len(matches) != 1:
                raise ValueError(f"Expected one archive entry for {relative}")
            info = matches[0]
            if info.file_size > 50_000_000:
                raise ValueError(f"Unexpected file size: {relative}")
            payload = source.read(info)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Official-data checksum mismatch: {relative}")
            target = destination / relative
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise FileExistsError(f"Refusing to replace different existing file: {target}")
            verified[relative] = payload
    # Every member is verified before the first write. Never extract ZIP paths.
    for relative, payload in verified.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(payload)
    return {"archive": str(archive.resolve()), "files": list(verified),
            "raw_dir": str(destination.resolve())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--destination", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    print(json.dumps(import_archive(args.archive, args.destination), indent=2))
