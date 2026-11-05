"""Reproducible audit of the supplied CS5228 project data.

Run from the project root:
    python src/data_audit.py

Outputs:
    artifacts/data_audit.json
    reports/data_audit.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from .submission_contract import submission_frame
except ImportError:
    from submission_contract import submission_frame


TARGET = "MONTHLY_RENT"
DATE_COLUMN = "RENT_APPROVAL_DATE"
EXPECTED_SUBMISSION_COLUMNS = ["Id", "Predicted"]


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize identifiers without changing the raw data."""
    return (
        series.astype("string")
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if count > 0
        },
        "constant_columns": [
            column for column in frame.columns if frame[column].nunique(dropna=False) <= 1
        ],
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def date_summary(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame[DATE_COLUMN].astype("string")
    return {
        "minimum": str(values.min()),
        "maximum": str(values.max()),
        "unique_months": int(values.nunique()),
        "counts_by_month": {
            str(key): int(value)
            for key, value in values.value_counts().sort_index().items()
        },
    }


def category_drift(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    categorical_columns = train.drop(columns=[TARGET]).select_dtypes(
        include=["object", "string"]
    ).columns
    for column in categorical_columns:
        train_values = set(normalize_text(train[column]).dropna().unique())
        test_values = set(normalize_text(test[column]).dropna().unique())
        result[column] = {
            "train_unique": len(train_values),
            "test_unique": len(test_values),
            "test_only": sorted(test_values - train_values),
            "train_only_count": len(train_values - test_values),
        }
    return result


def hdb_join_coverage(frame: pd.DataFrame, blocks: pd.DataFrame) -> float:
    left = frame[["TOWN", "BLOCK", "STREET"]].copy()
    right = blocks[["TOWN", "BLOCK", "STREET", "LATITUDE"]].copy()
    for table in (left, right):
        table["town_key"] = normalize_text(table["TOWN"])
        table["block_key"] = normalize_text(table["BLOCK"])
        table["street_key"] = normalize_text(table["STREET"])
    keys = ["town_key", "block_key", "street_key"]
    right = right[keys + ["LATITUDE"]].drop_duplicates(keys)
    merged = left.merge(right, on=keys, how="left", validate="many_to_one")
    return float(merged["LATITUDE"].notna().mean())


def duplicate_target_summary(train: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    grouped = train.groupby(feature_columns, dropna=False)[TARGET].agg(
        size="size", unique_targets="nunique", mean="mean", std="std"
    )
    repeated = grouped["size"] > 1
    conflicting = grouped["unique_targets"] > 1
    return {
        "feature_groups": int(len(grouped)),
        "repeated_groups": int(repeated.sum()),
        "conflicting_target_groups": int(conflicting.sum()),
        "rows_in_repeated_groups": int(grouped.loc[repeated, "size"].sum()),
    }


def audit(project_root: Path) -> dict[str, Any]:
    raw = project_root / "data" / "raw"
    train_path = raw / "train.csv"
    test_path = raw / "test.csv"
    sample_path = raw / "example-submission.csv"
    block_path = raw / "auxiliary" / "sg-hdb-block.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = submission_frame(raw, len(test))
    blocks = pd.read_csv(block_path)

    train_dates = date_summary(train)
    test_dates = date_summary(test)
    train_months = set(train[DATE_COLUMN].astype("string"))
    test_months = set(test[DATE_COLUMN].astype("string"))

    numeric_target = train[TARGET]
    quantiles = numeric_target.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])

    auxiliary: dict[str, Any] = {}
    input_files = [train_path, test_path] + ([sample_path] if sample_path.is_file() else [])
    for path in sorted((raw / "auxiliary").glob("*.csv")):
        frame = pd.read_csv(path)
        auxiliary[path.name] = frame_summary(frame)
        input_files.append(path)

    submission_valid = (
        list(sample.columns) == EXPECTED_SUBMISSION_COLUMNS
        and len(sample) == len(test)
        and sample["Id"].tolist() == list(range(len(test)))
    )

    audit_result: dict[str, Any] = {
        "audit_version": 1,
        "raw_file_sha256": {path.name: sha256(path) for path in input_files},
        "train": frame_summary(train),
        "test": frame_summary(test),
        "example_submission": frame_summary(sample),
        "submission_layout_source": "course template" if sample_path.is_file() else "in-memory zero-based positional IDs; not a course file",
        "dates": {
            "train": train_dates,
            "test": test_dates,
            "overlap_months": sorted(train_months & test_months),
        },
        "target": {
            "name": TARGET,
            "mean": float(numeric_target.mean()),
            "standard_deviation": float(numeric_target.std()),
            "unique_values": int(numeric_target.nunique()),
            "quantiles": {str(index): float(value) for index, value in quantiles.items()},
        },
        "category_drift": category_drift(train, test),
        "hdb_block_join_coverage": {
            "train": hdb_join_coverage(train, blocks),
            "test": hdb_join_coverage(test, blocks),
        },
        "duplicate_feature_targets": duplicate_target_summary(train, list(test.columns)),
        "submission_contract_valid": bool(submission_valid),
        "auxiliary": auxiliary,
        "provisional_validation": {
            "development_end": "2024-09",
            "main_validation_start": "2024-10",
            "main_validation_end": "2025-03",
            "rationale": "Most test rows are later in time than the training set.",
        },
    }
    return audit_result


def render_markdown(result: dict[str, Any]) -> str:
    train = result["train"]
    test = result["test"]
    dates = result["dates"]
    target = result["target"]
    joins = result["hdb_block_join_coverage"]
    duplicates = result["duplicate_feature_targets"]

    drift_lines = []
    for column, details in result["category_drift"].items():
        if details["test_only"] or details["train_only_count"]:
            drift_lines.append(
                f"- `{column}`: {details['train_unique']} train categories, "
                f"{details['test_unique']} test categories, "
                f"{len(details['test_only'])} normalized test-only categories."
            )

    auxiliary_lines = [
        f"- `{name}`: {details['rows']:,} rows x {details['columns']} columns"
        for name, details in result["auxiliary"].items()
    ]

    return "\n".join(
        [
            "# Data Audit",
            "",
            "Generated by `python src/data_audit.py`. Do not edit this file manually.",
            "",
            "## Core files",
            "",
            f"- Train: {train['rows']:,} rows x {train['columns']} columns.",
            f"- Test: {test['rows']:,} rows x {test['columns']} columns.",
            f"- Train period: {dates['train']['minimum']} to {dates['train']['maximum']}.",
            f"- Test period: {dates['test']['minimum']} to {dates['test']['maximum']}.",
            f"- Overlap months: {', '.join(dates['overlap_months']) or 'none'}.",
            f"- Submission contract valid: {result['submission_contract_valid']}.",
            "",
            "## Target",
            "",
            f"- Mean monthly rent: SGD {target['mean']:,.2f}.",
            f"- Median monthly rent: SGD {target['quantiles']['0.5']:,.0f}.",
            f"- Range: SGD {target['quantiles']['0.0']:,.0f} to SGD {target['quantiles']['1.0']:,.0f}.",
            f"- Unique target values: {target['unique_values']:,}.",
            "",
            "## Data-quality findings",
            "",
            f"- Constant train columns: {', '.join(f'`{c}`' for c in train['constant_columns']) or 'none'}.",
            f"- Constant test columns: {', '.join(f'`{c}`' for c in test['constant_columns']) or 'none'}.",
            f"- Exact duplicate train rows: {train['duplicate_rows']:,}.",
            f"- Rows belonging to repeated feature groups: {duplicates['rows_in_repeated_groups']:,}.",
            f"- Repeated feature groups with different rent targets: {duplicates['conflicting_target_groups']:,}.",
            f"- Normalized HDB-block join coverage: {joins['train']:.1%} train and {joins['test']:.1%} test.",
            "",
            "## Category drift",
            "",
            *(drift_lines or ["- No normalized categorical drift detected."]),
            "",
            "## Auxiliary files",
            "",
            *auxiliary_lines,
            "",
            "## Validation implication",
            "",
            "Most test rows occur after the final training month. Use chronological rolling validation as the primary design. Random splitting may be kept only as a secondary diagnostic because it can mix time periods and repeated property patterns.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    result = audit(project_root)

    artifact_dir = project_root / "artifacts"
    report_dir = project_root / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = artifact_dir / "data_audit.json"
    markdown_path = report_dir / "data_audit.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()

