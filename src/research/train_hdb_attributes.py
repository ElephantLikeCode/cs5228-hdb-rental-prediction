"""Paired feature experiments, preserving the current selected pipeline."""

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.hdb_attribute_features import (
    course_keys, prepare_resale, property_features, provenance_audit,
    resale_features, supplied_current_macro,
)
from src.market_experiments import monthly_market
from src.modeling import load_config
from src.research.compare_models import fit_component
from src.validation import regression_metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/attribute_study_20261102"
PARENT = ROOT / "artifacts/research_20261102/e925f89746af432b"
DATA = ROOT / "data/external/attribute_sources_20261102"


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", choices=["resale", "stock_snapshot", "given_macro"],
                        default=["resale", "stock_snapshot", "given_macro"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / "data/raw/train.csv")
    rows = pd.read_csv(PARENT / "short_recent_scoring_rows.csv")
    train = data.loc[data.RENT_APPROVAL_DATE.lt("2025-01")].drop(columns="MONTHLY_RENT").reset_index(drop=True)
    query = data.iloc[rows.source_row].drop(columns="MONTHLY_RENT").reset_index(drop=True)
    fit_x, query_x, y, weights, _ = pd.read_pickle(PARENT / "short_recent_local_trend_0.pkl")
    assert len(train) == len(fit_x) and len(query) == len(query_x)
    assert np.array_equal(y, data.loc[data.RENT_APPROVAL_DATE.lt("2025-01"), "MONTHLY_RENT"])
    config = load_config()
    protocols = dict(
        interventions={
            "resale": "17 features from trailing 24-month resale records; two-month lag and fixed Jan-2025 origin; only sales through 2024-12 downloaded to final CSV",
            "stock_snapshot": "16 building-stock/context features from current HDB property snapshot; retrospective snapshot-assisted evidence, not verified historical-vintage forecasting",
            "given_macro": "Current-month supplied COE/stock covariates; conditional retrospective prediction, not a Jan-2025 fixed-origin forecast",
        },
        model="CatBoost direct/residual local-trend pair, replace same two components in four-head CPU reference; other two augmented heads unchanged",
        parameters=config, seed=42, device="CPU", threads=6,
        validation="2025-01..2025-03; reused development window; no early stopping",
        candidates=args.candidates,
        information_boundary="No external rental labels, no per-lease floor assignment, no source IDs as inferred tenancy type",
        screening="Advance only if complete ensemble improves >=2 RMSE with address-cluster upper delta bound <0. No upload or default-config replacement.",
        code_sha256=hashlib.sha256(b"".join((ROOT / "src" / file).read_bytes()
            for file in ["research/train_hdb_attributes.py", "research/hdb_attribute_features.py"])).hexdigest(),
        manifests={name: json.loads((DATA / f"{name}_manifest.json").read_text())
                   for name in ["resale", "property"] if (DATA / f"{name}_manifest.json").exists()},
        supplied_data_sha256={name: hashlib.sha256((ROOT / "data/raw/auxiliary" / name).read_bytes()).hexdigest()
                              for name in ["sg-coe-prices.csv", "sg-stock-prices.csv"]},
        packages={name: version(name) for name in ["catboost", "pandas", "numpy"]},
    )
    identity = hashlib.sha256(json.dumps(protocols, sort_keys=True).encode()).hexdigest()[:16]
    folder = OUT / identity
    folder.mkdir(exist_ok=True)
    (folder / "protocol.json").write_text(json.dumps(protocols, indent=2))
    sales = None
    if "resale" in args.candidates:
        sales = prepare_resale(pd.read_csv(DATA / "resale.csv"))
        summary, mappings = provenance_audit(data.drop(columns="MONTHLY_RENT"), sales)
        (folder / "provenance_audit.json").write_text(json.dumps(summary, indent=2))
        mappings.to_csv(folder / "attribute_template_support.csv", index=False)
        print(json.dumps(summary), flush=True)
    property_source = pd.read_csv(DATA / "property.csv") if "stock_snapshot" in args.candidates else None
    macro = monthly_market(ROOT / "data/raw/auxiliary") if "given_macro" in args.candidates else None
    reference = np.load(PARENT / "short_recent_reference_ensemble.npy")
    old_pair = [np.load(PARENT / f"short_recent_reference_hierarchical_6_{mode}_42.npy")
                for mode in ["direct", "residual"]]
    metrics = [dict(candidate="reference", **regression_metrics(rows.target, reference))]
    for name in args.candidates:
        feature_file = folder / f"{name}_features.pkl"
        if feature_file.exists():
            extra_a, extra_b = pd.read_pickle(feature_file)
        else:
            print(f"Building {name}", flush=True)
            if name == "resale":
                extra_a = resale_features(train, sales)
                extra_b = resale_features(query, sales, origin="2025-01")
            elif name == "stock_snapshot":
                extra_a = property_features(train, property_source)
                extra_b = property_features(query, property_source)
            else:
                extra_a = supplied_current_macro(macro, train)
                extra_b = supplied_current_macro(macro, query)
                pd.to_pickle((extra_a, supplied_current_macro(macro, query, origin="2025-01")),
                             folder / "macro_frozen_control_features.pkl")
            assert len(extra_a) == len(fit_x) and len(extra_b) == len(query_x)
            pd.to_pickle((extra_a, extra_b), feature_file)
        print(f"{name}: {extra_a.shape[1]} additional features", flush=True)
        predictions, started = [], time.perf_counter()
        for mode in ["direct", "residual"]:
            destination = folder / f"{name}_{mode}.npy"
            if destination.exists():
                prediction = np.load(destination)
            else:
                print(f"Fitting {name}/{mode}", flush=True)
                a, b = fit_x.join(extra_a), query_x.join(extra_b)
                model, prediction, _ = fit_component(
                    a, y, b, learner="catboost", mode=mode, seed=42,
                    config=config, threads=6, sample_weight=weights)
                model.save_model(str(folder / f"{name}_{mode}.cbm"))
                np.save(destination, prediction)
                if name == "given_macro":
                    from src.modeling import predict_model
                    frozen_extra = pd.read_pickle(folder / "macro_frozen_control_features.pkl")[1]
                    frozen = predict_model(model, query_x.join(frozen_extra), dict(config, residual_target=mode == "residual"))
                    np.save(folder / f"{name}_{mode}_frozen.npy", frozen)
            predictions.append(prediction)
        ensemble = np.mean(predictions+old_pair, axis=0)
        np.save(folder / f"{name}_ensemble.npy", ensemble)
        record = dict(candidate=name, seconds=time.perf_counter()-started,
                      **regression_metrics(rows.target, ensemble))
        metrics.append(record)
        if name == "given_macro":
            frozen = np.mean([np.load(folder / f"{name}_{mode}_frozen.npy")
                              for mode in ["direct", "residual"]]+old_pair, axis=0)
            np.save(folder / "given_macro_frozen_ensemble.npy", frozen)
            metrics.append(dict(candidate="given_macro_frozen_control", **regression_metrics(rows.target, frozen)))
        pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
        print(json.dumps(record), flush=True)
    print(f"Completed {folder}", flush=True)


if __name__ == "__main__":
    run()
