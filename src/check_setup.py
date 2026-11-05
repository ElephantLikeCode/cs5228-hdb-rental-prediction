"""Validate the local Python environment and the private course-data layout.

Run this immediately after installing dependencies and copying the supplied
data into ``data/raw``. The check is read-only and does not create artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import sys
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pytest": "pytest",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "ipykernel": "ipykernel",
    "jupyterlab": "jupyterlab",
}

CORE_FILES = ("train.csv", "test.csv")
AUXILIARY_FILES = (
    "sg-coe-prices.csv",
    "sg-hdb-block.csv",
    "sg-mrt-stations.csv",
    "sg-schools.csv",
    "sg-shopping-malls.csv",
    "sg-stock-prices.csv",
)

RAW_SHA256 = {
    "train.csv": "6d7a417de72112361f8353659ba6fe44a442b07465b488aff1b03f7ec1ea3ba6",
    "test.csv": "dfc407406d957bbf7c2f005e5d9852b1bfc6546cd4186f07e6824d8e03e64cfc",
    "example-submission.csv": "ab751676b2bfc6121ac3543075d4f2c8978a6d73a14a3442a1bb7bdbf7867678",
    "auxiliary/sg-coe-prices.csv": "e32fccd39da32780b6b7f35d069c813ed3edb0c110ea60761656bcee58d501a5",
    "auxiliary/sg-hdb-block.csv": "e6ccc60c1f901f8f476eb6fe14c07ef5941d2afd06a081a1e4c00f717da27f83",
    "auxiliary/sg-mrt-stations.csv": "899460b39fb251bd6883fffc9e6a772081f8e5217a657550afc3c49cfdcd3227",
    "auxiliary/sg-schools.csv": "118c7ce8d96bbc934bd53655e73e6d0848c27853a625394ecc6a67a5af54f192",
    "auxiliary/sg-shopping-malls.csv": "00e05b0176ac412695fc4ce1fc5a5338a1993593ffaffe26156dc2ea04ddf2c8",
    "auxiliary/sg-stock-prices.csv": "c52fef9f33cfcb11285e2b9546124e02f32488e563af44e9fb3a505d10e2edff",
}


def csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_python() -> list[str]:
    failures: list[str] = []
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info[:2] != (3, 12):
        failures.append(f"Python 3.12 is required; current interpreter is {version}")
    else:
        print(f"[PASS] Python {version}: {sys.executable}")
    return failures


def check_packages() -> list[str]:
    failures: list[str] = []
    for distribution, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            installed = metadata.version(distribution)
            print(f"[PASS] {distribution} {installed}")
        except Exception as exc:  # Import errors may include broken native wheels.
            failures.append(f"{distribution}: {type(exc).__name__}: {exc}")
    return failures


def check_data() -> list[str]:
    failures: list[str] = []
    required = [RAW_DIR / name for name in CORE_FILES]
    required += [RAW_DIR / "auxiliary" / name for name in AUXILIARY_FILES]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        failures.append("Missing data files: " + ", ".join(missing))
        return failures

    train_columns = csv_header(RAW_DIR / "train.csv")
    test_columns = csv_header(RAW_DIR / "test.csv")
    if "MONTHLY_RENT" not in train_columns:
        failures.append("train.csv is missing MONTHLY_RENT")
    if "MONTHLY_RENT" in test_columns:
        failures.append("test.csv unexpectedly contains MONTHLY_RENT")
    if [column for column in train_columns if column != "MONTHLY_RENT"] != test_columns:
        failures.append("train/test feature columns do not match")
    sample_path = RAW_DIR / "example-submission.csv"
    if sample_path.exists() and csv_header(sample_path) != ["Id", "Predicted"]:
        failures.append(
            "example-submission.csv columns must be exactly: Id, Predicted"
        )
    for relative_path, expected_hash in RAW_SHA256.items():
        if relative_path == "example-submission.csv" and not sample_path.exists():
            print("[INFO] Optional example-submission.csv absent; output layout is constructed in memory")
            continue
        actual_hash = sha256(RAW_DIR / relative_path)
        if actual_hash != expected_hash:
            failures.append(
                f"Data version mismatch for data/raw/{relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    if not failures:
        print(
            f"[PASS] All {len(required)} required data files, checksums, and core schemas"
        )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="check Python and packages before the private dataset is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = check_python() + check_packages()
    if not args.skip_data:
        failures += check_data()

    if failures:
        print("\nSETUP CHECK FAILED")
        for failure in failures:
            print(f"[FAIL] {failure}")
        print("\nSee README.md, correct the setup, then run this check again.")
        raise SystemExit(1)

    print("\nSETUP CHECK PASSED")
    if args.skip_data:
        print("Environment is ready. Add the private data, then rerun without --skip-data.")
    else:
        print("Next: python src/data_audit.py && python -m pytest -q")


if __name__ == "__main__":
    main()
