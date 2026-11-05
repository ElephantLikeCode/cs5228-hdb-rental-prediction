"""Forward-only horizon-dependent blending of frozen OOF predictions.

No base-model fitting, test predictions, config changes, or public-score tuning.
Run: python -m src.research.horizon_blend --audit-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[2]
HEADS = ("local_trend_direct", "local_trend_residual", "horizon_direct", "horizon_residual")
PLANS = {
    "quarter_2023": (["bridge_2022"], "bridge_2023"),
    "quarter_2024_mid": (["bridge_2022", "bridge_2023"], "bridge_2024"),
    "quarter_2024_end": (["bridge_2022", "bridge_2023", "bridge_2024"], "short_middle"),
    "quarter_2025": (["bridge_2022", "bridge_2023", "bridge_2024", "short_middle"], "short_recent"),
    "year_2023": (["long_early"], "long_older"),
    "year_2024": (["long_early", "long_older"], "long_recent"),
    "horizon16": (["long_early"], "horizon16"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_path(root, value):
    # Historical manifests were written on Windows. Preserve their bytes.
    relative = Path(str(value).replace("\\", "/"))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Prediction path must stay inside the cache root")
    return path


def audit_sources(manifest, root, windows):
    rows = manifest.loc[manifest.window.isin(windows)].copy()
    expected = {(w, c, s) for w in windows for c in HEADS for s in (42, 2026)}
    keys = list(zip(rows.window, rows.candidate, rows.seed.astype(int)))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Manifest needs exactly four heads x two seeds for each window")
    if rows.run_id.nunique() != len(rows) or rows.data_sha256.nunique() != 1:
        raise ValueError("Duplicate run IDs or different training data")
    if not (rows.iterations.eq(1600) & rows.actual_trees.eq(1600)
            & rows.task_type.eq("GPU")
            & rows.protocol.eq("fixed_budget_no_outer_early_stopping")).all():
        raise ValueError("Sources must use the frozen fixed-budget GPU protocol")
    missing = []
    for row in rows.itertuples():
        path = cache_path(root, row.prediction_file)
        if not path.is_file():
            missing.append(str(path))
        elif digest(path) != row.prediction_sha256:
            raise ValueError(f"Prediction checksum mismatch: {row.run_id}")
    return rows, missing


def load_window(sources, window, root, raw):
    records = sources.loc[sources.window.eq(window)]
    for field in ("train_end", "validation_start", "validation_end"):
        if records[field].nunique() != 1:
            raise ValueError(f"Mixed {field}: {window}")
    cutoff = str(records.iloc[0].train_end)
    columns, first = [], None
    for head in HEADS:
        predictions = []
        for row in records.loc[records.candidate.eq(head)].sort_values("seed").itertuples():
            with np.load(cache_path(root, row.prediction_file), allow_pickle=False) as z:
                item = {k: z[k].copy() for k in ("row_id", "month", "y", "prediction")}
            item["month"] = item["month"].astype(str)
            if any(v.ndim != 1 or len(v) != len(item["y"]) for v in item.values()):
                raise ValueError("Malformed prediction arrays")
            if not np.isfinite(item["y"]).all() or not np.isfinite(item["prediction"]).all():
                raise ValueError("Non-finite targets or predictions")
            if first is None:
                first = item
            elif any(not np.array_equal(first[k], item[k]) for k in ("row_id", "month", "y")):
                raise ValueError("Component validation rows/targets/months do not match")
            predictions.append(item["prediction"])
        columns.append(np.mean(predictions, axis=0))
    ids = first["row_id"]
    if ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("Row IDs must be unique integers")
    start, end = records.iloc[0][["validation_start", "validation_end"]]
    expected = np.flatnonzero(raw.RENT_APPROVAL_DATE.between(start, end))
    if not np.array_equal(ids, expected):
        raise ValueError("Cache does not contain every official validation row in order")
    if not np.array_equal(raw.iloc[ids].MONTHLY_RENT.to_numpy(), first["y"]):
        raise ValueError("Cache labels differ from official training labels")
    if not np.array_equal(raw.iloc[ids].RENT_APPROVAL_DATE.to_numpy(), first["month"]):
        raise ValueError("Cache months differ from official training months")
    h = pd.PeriodIndex(first["month"], freq="M").asi8 - pd.Period(cutoff, freq="M").ordinal
    if len(h) == 0 or np.any(h <= 0):
        raise ValueError("Every prediction must follow its base-model cutoff")
    p = np.column_stack(columns)
    return dict(row_id=ids, month=first["month"], y=first["y"], h=h,
                normal=p[:, :2].mean(axis=1), stale=p[:, 2:].mean(axis=1),
                cutoff=cutoff, run_ids=records.run_id.tolist())


def forward_training(training, evaluation):
    result = {k: np.concatenate([w[k] for w in training])
              for k in ("row_id", "month", "y", "h", "normal", "stale")}
    if len(np.unique(result["row_id"])) != len(result["row_id"]):
        raise ValueError("Meta-training windows overlap")
    if (max(result["month"]) > evaluation["cutoff"]
            or max(result["month"]) >= min(evaluation["month"])
            or np.intersect1d(result["row_id"], evaluation["row_id"]).size):
        raise ValueError("Meta-training labels are unavailable at evaluation origin")
    return result


def basis(h, low, high, dynamic):
    if not dynamic or high == low:
        return np.ones((len(h), 1))
    t = np.clip((np.asarray(h) - low) / (high - low), 0, 1)
    return np.column_stack((1 - t, t))


def fit_gate(train, dynamic):
    """Convex least squares with fixed regularization, no outer-label tuning.

    Two endpoint weights give a linear smooth gate; beyond observed horizons,
    clamp to the closest endpoint. Fit deviations within +/-0.5, then shrink
    them by 50%, yielding deployed normal weights within [0.25, 0.75].
    Ridge strength 0.1 is prespecified and scaled by mean branch disagreement².
    """
    h = np.asarray(train["h"], dtype=float)
    if not len(h) or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError("Invalid training horizons")
    low, high = float(h.min()), float(h.max())
    B = basis(h, low, high, dynamic)
    delta = train["normal"] - train["stale"]
    baseline = (train["normal"] + train["stale"]) / 2
    X = B * delta[:, None]
    penalty = .1 * max(float(np.mean(delta ** 2)), 1.)
    X = np.vstack((X / np.sqrt(len(h)), np.sqrt(penalty) * np.eye(B.shape[1])))
    y = np.r_[(train["y"] - baseline) / np.sqrt(len(h)), np.zeros(B.shape[1])]
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite gate training values")
    fitted = lsq_linear(X, y, bounds=(-.5, .5), tol=1e-12)
    if not fitted.success:
        raise RuntimeError(fitted.message)
    return dict(dynamic=dynamic, low=low, high=high, coefficients=(.5 * fitted.x).tolist(),
                ridge_ratio=.1, shrinkage=.5, weight_bounds=[.25, .75])


def predict_gate(model, normal, stale, horizons):
    h = np.asarray(horizons, dtype=float)
    if not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError("Invalid prediction horizons")
    w = .5 + basis(h, model["low"], model["high"], model["dynamic"]) @ model["coefficients"]
    return w * normal + (1 - w) * stale, w


def metrics(y, prediction):
    error = prediction - y
    return dict(mse=float(np.mean(error ** 2)), rmse=float(np.sqrt(np.mean(error ** 2))),
                mae=float(np.mean(abs(error))), bias=float(np.mean(error)))


def run(sources, root, raw, plans, output):
    windows = sorted({w for p in plans for w in PLANS[p][0] + [PLANS[p][1]]})
    cache = {w: load_window(sources, w, root, raw) for w in windows}
    records, monthly, models = [], [], {}
    for plan in plans:
        fit_windows, eval_window = PLANS[plan]
        evaluation = cache[eval_window]
        train = forward_training([cache[w] for w in fit_windows], evaluation)
        ref = (evaluation["normal"] + evaluation["stale"]) / 2
        predictions = {"equal_50_50": (ref, np.full(len(ref), .5))}
        models[plan] = dict(fit_windows=fit_windows, evaluation_window=eval_window,
                            meta_train_end=str(max(train["month"])),
                            evaluation_cutoff=evaluation["cutoff"], gates={})
        for method in ("learned_constant", "horizon_linear"):
            gate = fit_gate(train, method == "horizon_linear")
            models[plan]["gates"][method] = gate
            predictions[method] = predict_gate(gate, evaluation["normal"], evaluation["stale"], evaluation["h"])
        reference = metrics(evaluation["y"], ref)
        for method, (prediction, weights) in predictions.items():
            score = metrics(evaluation["y"], prediction)
            records.append(dict(plan=plan, method=method, rows=len(ref), **score,
                                rmse_gain_vs_equal=reference["rmse"] - score["rmse"],
                                normal_weight_min=float(weights.min()), normal_weight_max=float(weights.max()),
                                horizons_beyond_fit=int(np.sum(evaluation["h"] > train["h"].max()))))
            for h in np.unique(evaluation["h"]):
                mask = evaluation["h"] == h
                monthly.append(dict(plan=plan, method=method, horizon=int(h), rows=int(mask.sum()),
                                    normal_weight=float(weights[mask].mean()),
                                    **metrics(evaluation["y"][mask], prediction[mask])))
    # Do not leave result artifacts until every input and forward split passes.
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(records).to_csv(output / "results.csv", index=False)
    pd.DataFrame(monthly).to_csv(output / "by_horizon.csv", index=False)
    sources.to_csv(output / "sources.csv", index=False)
    metadata = dict(plans=models, code_sha256=digest(Path(__file__)),
                    protocol="Frozen development windows; no evaluation labels used in gate fitting; overlapping evaluation windows are not independent.",
                    decision="Research only; no automatic promotion or submission. Require recent and long-window gains plus stability before confirmation.")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/stacking_sources.csv")
    parser.add_argument("--cache-root", type=Path, default=ROOT)
    parser.add_argument("--raw-train", type=Path, default=ROOT / "data/raw/train.csv")
    parser.add_argument("--plans", nargs="+", choices=PLANS, default=list(PLANS))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/horizon_blend_v1")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    windows = sorted({w for p in args.plans for w in PLANS[p][0] + [PLANS[p][1]]})
    sources, missing = audit_sources(pd.read_csv(args.manifest), args.cache_root, windows)
    print(f"Required caches: {len(sources)}; missing: {len(missing)}")
    if missing:
        print("Restore original checksum-matched caches from the training machine; aggregate CSV metrics cannot replace row predictions.")
        print("\n".join(missing[:8]))
        raise SystemExit(2)
    if not args.raw_train.is_file() or digest(args.raw_train) != sources.data_sha256.iloc[0]:
        raise ValueError("Official training CSV is absent or its checksum differs")
    raw = pd.read_csv(args.raw_train)
    if args.audit_only:
        loaded = {w: load_window(sources, w, args.cache_root, raw) for w in windows}
        for p in args.plans:
            training, evaluation = PLANS[p]
            forward_training([loaded[w] for w in training], loaded[evaluation])
        print("Cache hashes, complete validation rows and forward splits verified.")
        return
    if args.output.exists():
        raise FileExistsError("Use a fresh --output directory to preserve previous results")
    print(run(sources, args.cache_root, raw, args.plans, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
