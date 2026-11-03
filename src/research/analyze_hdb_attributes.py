"""Paired comparisons for the external-attribute experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.research.hdb_attribute_features import course_keys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/attribute_study_20261102"
PARENT = ROOT / "artifacts/research_20261102/e925f89746af432b"


def main():
    scoring = pd.read_csv(PARENT / "short_recent_scoring_rows.csv")
    source = pd.read_csv(ROOT / "data/raw/train.csv").iloc[scoring.source_row].reset_index(drop=True)
    y = scoring.target.to_numpy()
    assert np.array_equal(source.MONTHLY_RENT.to_numpy(), y)
    keys = course_keys(source)
    group, names = pd.factorize(keys.town+"|"+keys.block+"|"+keys.street)
    counts = np.bincount(group)
    rng = np.random.default_rng(20261102)
    bootstrap = rng.multinomial(len(names), np.full(len(names), 1/len(names)), size=1000)
    mass = bootstrap@counts
    reference = np.load(PARENT / "short_recent_reference_ensemble.npy")
    predictions = {"reference": reference}
    for path in OUT.glob("*/*_ensemble.npy"):
        predictions[path.stem.replace("_ensemble", "")] = np.load(path)
    reference_square = np.bincount(group, weights=(reference-y)**2)
    reference_sample = np.sqrt(np.einsum("ij,j->i", bootstrap, reference_square, optimize=False)/mass)
    records = []
    for name, pred in predictions.items():
        assert pred.shape == y.shape and np.isfinite(pred).all()
        square = np.bincount(group, weights=(pred-y)**2)
        delta = np.sqrt(np.einsum("ij,j->i", bootstrap, square, optimize=False)/mass)-reference_sample
        records.append(dict(candidate=name, rmse=float(np.sqrt(np.mean((pred-y)**2))),
                            mae=float(np.mean(np.abs(pred-y))), bias=float(np.mean(pred-y)),
                            delta_rmse=float(np.sqrt(np.mean((pred-y)**2))-np.sqrt(np.mean((reference-y)**2))),
                            delta_ci_low=float(np.quantile(delta,.025)), delta_ci_high=float(np.quantile(delta,.975))))
    summary = pd.DataFrame(records).sort_values("rmse")
    summary.to_csv(OUT / "paired_comparison.csv", index=False)
    slices = []
    masks = {"all": np.ones(len(y), dtype=bool),
             "rent_below_1000": y < 1000, "rent_at_least_1000": y >= 1000}
    for month in sorted(source.RENT_APPROVAL_DATE.unique()):
        masks[month] = source.RENT_APPROVAL_DATE.eq(month).to_numpy()
    for name, pred in predictions.items():
        for label, mask in masks.items():
            if mask.any():
                slices.append(dict(candidate=name, subset=label, rows=int(mask.sum()),
                                   rmse=float(np.sqrt(np.mean((pred[mask]-y[mask])**2))),
                                   mae=float(np.mean(np.abs(pred[mask]-y[mask])))))
    pd.DataFrame(slices).to_csv(OUT / "subgroup_comparison.csv", index=False)
    probabilities = OUT / "low_rent_information/validation_probabilities.csv"
    if probabilities.exists():
        p = pd.read_csv(probabilities)
        gains = []
        for draw in bootstrap[:500]:
            weight = draw[group]
            if np.unique(p.event[weight>0]).size < 2:
                continue
            gains.append(roc_auc_score(p.event, p.plus_housing_information, sample_weight=weight)
                         -roc_auc_score(p.event, p.existing_inputs, sample_weight=weight))
        interval = dict(auc_gain=float(roc_auc_score(p.event,p.plus_housing_information)-roc_auc_score(p.event,p.existing_inputs)),
                        ci_low=float(np.quantile(gains,.025)), ci_high=float(np.quantile(gains,.975)),
                        samples=len(gains), warning="Exploratory low-rent event ranking, not verified bedroom rental classification.")
        (OUT / "low_rent_information/paired_auc.json").write_text(json.dumps(interval,indent=2))
        print(json.dumps(interval), flush=True)
    print(summary.to_string(index=False), flush=True)
    (OUT / "comparison_caveats.json").write_text(json.dumps(dict(
        window="2025-01..2025-03, 8924 rows", status="Reused development window, not Kaggle test score",
        prediction_rules={"resale":"sales prior to origin with two-month lag",
                          "stock_snapshot":"current external property snapshot, not historical-vintage forecast",
                          "given_macro":"contemporaneous covariates supplied by the task, conditional prediction",
                          "given_macro_frozen":"same fitted macro models, covariates frozen at origin"},
        uncertainty="1000 address-cluster draws; ignores research/model-selection reuse and future distribution shifts",
        no_external_rental_targets=True,
    ), indent=2))


if __name__ == "__main__":
    main()
