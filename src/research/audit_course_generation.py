"""Reproducible fingerprints; never infer a generation rule from a single match."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.hdb_attribute_features import course_keys, prepare_resale

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/attribute_study_20261102"


def main():
    raw = pd.read_csv(ROOT / "data/raw/train.csv")
    sales = prepare_resale(pd.read_csv(ROOT / "data/external/attribute_sources_20261102/resale.csv"))
    keys = ["town", "block", "street", "type"]
    course = course_keys(raw)
    course["course_area"] = raw.FLOOR_AREA_SQM
    course = course.drop_duplicates(keys)
    grouped = sales.sort_values(["month", "_id"]).groupby(keys).floor_area_sqm
    stats = grouped.agg(["min", "max", "median", "mean", "first", "last", "nunique"])
    stats["mode"] = grouped.agg(lambda values: values.value_counts().index[0])
    merged = course.join(stats, on=keys)
    results = []
    for label, mask in [("all_matched", merged["mean"].notna()),
                        ("at_least_3_sizes", merged["nunique"].ge(3))]:
        subset = merged.loc[mask]
        for rule in ["min", "max", "median", "first", "last", "mode"]:
            results.append(dict(subset=label, rule=rule, groups=len(subset),
                                exact_match_fraction=float(np.isclose(subset.course_area, subset[rule]).mean())))
    pd.DataFrame(results).to_csv(OUT / "generation_rule_fingerprint.csv", index=False)
    selected = sales.loc[sales.town.eq("queenstown") & sales.block.eq("8") &
                         sales.street.eq("holland avenue") & sales.type.eq("3 room")]
    example = dict(sales=len(selected), areas=sorted(selected.floor_area_sqm.unique().tolist()),
                   storey_ranges=sorted(selected.storey_range.unique().tolist()),
                   flat_models=selected.flat_model.unique().tolist(),
                   lease_years=selected.lease_commence_date.unique().tolist(),
                   conclusion="Same building/type contains different areas and storey ranges. Public sources have no rental-unit link key; no particular resale floor is assigned to a lease.")
    (OUT / "holland_example_attribute_support.json").write_text(json.dumps(example, indent=2))
    (OUT / "generation_audit_limits.json").write_text(json.dumps(dict(
        conclusion="No tested first/last/median/mode/min/max rule reproduces course floor areas across all matched groups.",
        limits="Resale source starts in 2017 and is a latest-vintage snapshot. A different original source/vintage/date range or rule remains possible. This audit does not recover the instructor's generation code.",
    ), indent=2))
    print(pd.DataFrame(results).to_string(index=False))
    print(json.dumps(example))


if __name__ == "__main__":
    main()
