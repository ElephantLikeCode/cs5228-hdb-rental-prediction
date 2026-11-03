"""Fixed-budget RMSE probes: duplicate aggregation and market-normalized targets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

try:
    from src.research.hdb_attribute_features import course_keys
    from src.modeling import load_config
except ImportError:
    from src.research.hdb_attribute_features import course_keys
    from src.modeling import load_config

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/rmse_structure_20261102"
CACHES = {
    "short_recent": "e925f89746af432b",
    "long_recent": "17bad5d96e4b3667",
    "long_older": "17bad5d96e4b3667",
}


def market_scales(raw, target, query):
    """Training-only monthly flat-type means; no query labels accepted."""
    key = course_keys(raw)["type"]
    qkey = course_keys(query)["type"]
    frame = pd.DataFrame({"type": key, "month": pd.PeriodIndex(raw.RENT_APPROVAL_DATE, freq="M"),
                          "target": np.asarray(target)})
    stats = frame.groupby(["type", "month"]).target.agg(["mean", "count"])
    train_scale = stats["mean"].reindex(pd.MultiIndex.from_frame(frame[["type", "month"]])).to_numpy()
    query_month = pd.PeriodIndex(query.RENT_APPROVAL_DATE, freq="M")
    flat, damped = np.empty(len(query)), np.empty(len(query))
    for name in qkey.unique():
        mask = qkey.eq(name).to_numpy()
        if name not in stats.index.get_level_values(0):
            flat[mask] = damped[mask] = np.mean(target)
            continue
        table = stats.loc[name].sort_index().tail(12)
        last = table.tail(3)
        anchor = np.average(last["mean"], weights=last["count"])
        slope = 0.0
        if len(table) >= 6:
            t = np.array([m.ordinal for m in table.index], dtype=float)
            slope = np.clip(np.polyfit(t-t.max(), np.log(table["mean"]), 1)[0], -.01, .01)
        horizon = np.maximum(np.array([m.ordinal-table.index.max().ordinal for m in query_month[mask]]), 0)
        flat[mask] = anchor
        damped[mask] = anchor*np.exp(slope*6*(1-np.exp(-horizon/6)))
    assert np.isfinite(train_scale).all() and (train_scale > 0).all()
    return train_scale, {"flat": flat, "damped6": damped}


def aggregate_duplicates(features, target):
    """Preserve every training label's contribution to squared-error loss."""
    codes = pd.util.hash_pandas_object(features, index=False)
    duplicate = codes.duplicated()
    # A hash collision is not allowed to merge distinct feature rows.
    assert len(features.drop_duplicates()) == int((~duplicate).sum())
    grouped = pd.DataFrame({"hash": codes.to_numpy(), "y": np.asarray(target)}).groupby("hash", sort=False).y.agg(["mean", "count"])
    first = features.loc[~duplicate].reset_index(drop=True)
    means = grouped["mean"].to_numpy()
    counts = grouped["count"].to_numpy(float)
    assert counts.sum() == len(target)
    return first, means, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", nargs="+", choices=list(CACHES), default=["short_recent"])
    parser.add_argument("--candidates", nargs="+", choices=["duplicate_mean", "market_normalized"], default=["duplicate_mean", "market_normalized"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    config = load_config()
    protocol = dict(
        candidates=args.candidates, windows=args.windows, config=config, seed=42, threads=8,
        target="RMSE over all official validation labels, including unusually low rents",
        interventions={"duplicate_mean":"Group exact 78-feature duplicate rows; mean target, multiplicity weights; no label removal",
                       "market_normalized":"Target divided by training-month/flat-type mean; scale-squared weights; omit absolute rent history and calendar year/index; forecasts flat and fixed damped6"},
        comparison="Replace only local_trend direct head; keep other three seed42 heads frozen. Also report the new head alone.",
        gate="Advance if replacement improves >=2 RMSE and address-cluster upper delta CI <0; confirm on long_recent and long_older before full-data fitting",
        exclusions="No outer-validation early stopping, no test rental labels, no threshold tuning or deleting scoring labels",
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    identity = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
    folder = OUT / identity
    folder.mkdir(exist_ok=True)
    (folder / "protocol.json").write_text(json.dumps(protocol,indent=2))
    source = pd.read_csv(ROOT / "data/raw/train.csv")
    records = []
    for window in args.windows:
        parent = ROOT / "artifacts/research_20261102" / CACHES[window]
        fit_x, query_x, fit_y, _, fit_raw = pd.read_pickle(parent/f"{window}_local_trend_0.pkl")
        scoring = pd.read_csv(parent/f"{window}_scoring_rows.csv")
        query = source.iloc[scoring.source_row].reset_index(drop=True)
        train_raw = fit_raw.reset_index(drop=True)
        y = scoring.target.to_numpy()
        old_heads = [np.load(parent/f"{window}_reference_{view}_{mode}_42.npy")
                     for view, mode in [("local_trend_0","residual"), ("hierarchical_6","direct"), ("hierarchical_6","residual")]]
        old_direct = np.load(parent/f"{window}_reference_local_trend_0_direct_42.npy")
        reference = np.mean(old_heads+[old_direct], axis=0)
        predictions = {"reference": reference}
        for candidate in args.candidates:
            if candidate == "duplicate_mean":
                left, target, weights = aggregate_duplicates(fit_x, fit_y)
                right = query_x
                scales = {"native": np.ones(len(query))}
            else:
                train_scale, scales = market_scales(train_raw, fit_y, query)
                drop = [c for c in fit_x if c in {"RENT_YEAR", "MONTH_INDEX", "ADAPT_MARKET_LEVEL", "ADAPT_HORIZON"}
                        or c.startswith("HIST_") and not c.endswith("LOG_COUNT")
                        or c.startswith("ADAPT_") and c.endswith("_LEVEL")]
                left, right = fit_x.drop(columns=drop), query_x.drop(columns=drop)
                target = np.asarray(fit_y)/train_scale
                weights = (train_scale/3000)**2
            model_path = folder / f"{window}_{candidate}.cbm"
            print(f"{window}/{candidate}: {left.shape}; training weight {weights.sum():.2f}",flush=True)
            model = CatBoostRegressor(
                iterations=config["iterations"], learning_rate=config["learning_rate"],
                depth=config["depth"], l2_leaf_reg=config["l2_leaf_reg"],
                loss_function="RMSE", task_type="CPU", thread_count=8,
                random_seed=42, allow_writing_files=False, verbose=400)
            if model_path.exists():
                model.load_model(str(model_path))
            else:
                model.fit(left, target, sample_weight=weights,
                          cat_features=left.select_dtypes(include=["object","string","category"]).columns.tolist())
                model.save_model(str(model_path))
            raw_prediction = model.predict(right)
            for scale_name, scale in scales.items():
                pred = raw_prediction*scale
                assert np.isfinite(pred).all()
                name = f"{candidate}_{scale_name}"
                predictions[name+"_standalone"] = pred
                predictions[name+"_replacement"] = np.mean(old_heads+[pred],axis=0)
        keys = course_keys(query)
        group, names = pd.factorize(keys.town+"|"+keys.block+"|"+keys.street)
        draws = np.random.default_rng(20261102).multinomial(len(names), np.full(len(names),1/len(names)),size=1000)
        mass = draws@np.bincount(group)
        reference_bootstrap = np.sqrt(np.einsum("ij,j->i",draws,np.bincount(group,weights=(reference-y)**2),optimize=False)/mass)
        for name, pred in predictions.items():
            np.save(folder/f"{window}_{name}.npy",pred)
            delta = np.sqrt(np.einsum("ij,j->i",draws,np.bincount(group,weights=(pred-y)**2),optimize=False)/mass)-reference_bootstrap
            records.append(dict(window=window,candidate=name,rmse=float(np.sqrt(np.mean((pred-y)**2))),
                                mae=float(np.abs(pred-y).mean()),bias=float((pred-y).mean()),
                                delta_rmse=float(np.sqrt(np.mean((pred-y)**2))-np.sqrt(np.mean((reference-y)**2))),
                                delta_ci_low=float(np.quantile(delta,.025)),delta_ci_high=float(np.quantile(delta,.975))))
        pd.DataFrame(records).to_csv(folder/"metrics.csv",index=False)
        print(pd.DataFrame(records).to_string(index=False),flush=True)
    print(f"Completed {folder}",flush=True)


if __name__ == "__main__":
    main()
