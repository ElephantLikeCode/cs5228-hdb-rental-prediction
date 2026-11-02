"""One declared follow-up for sparse high-cardinality categories."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from research_20261102 import encode_lgbm
from validation import regression_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    folder = args.run_folder
    destination = folder / "sparse_categories_followup"
    destination.mkdir(exist_ok=True)
    params = dict(
        objective="regression", n_estimators=1600, learning_rate=0.04,
        num_leaves=63, min_child_samples=50, reg_lambda=10,
        colsample_bytree=0.9, subsample=0.9, subsample_freq=1, max_bin=255,
        cat_smooth=1, cat_l2=1, min_data_per_group=10, max_cat_threshold=64,
        random_state=42, n_jobs=args.threads, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    protocol = dict(
        rationale="Initial LGBM ensemble RMSE 452.2773. Address/block are high-cardinality; check one less-smoothed category / higher-leaf preset.",
        scope="Two local-trend heads only; combine with the two unchanged augmented-history heads. Exploratory, not an independent test.",
        params=params, parent_protocol=json.loads((folder / "protocol.json").read_text()),
        code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stopping="One predeclared preset, no early stopping or post-hoc weight search.",
    )
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2))
    train, query, y_train, weights, _ = pd.read_pickle(folder / "short_recent_local_trend_0.pkl")
    left, right, vocabulary = encode_lgbm(train, query)
    truth = pd.read_csv(folder / "short_recent_scoring_rows.csv").target.to_numpy()
    predictions = []
    started = time.perf_counter()
    for mode in ("direct", "residual"):
        train_offset = train.ADAPT_ADDRESS_FLAT_AREA_LEVEL.to_numpy() if mode == "residual" else np.zeros(len(train))
        query_offset = query.ADAPT_ADDRESS_FLAT_AREA_LEVEL.to_numpy() if mode == "residual" else np.zeros(len(query))
        print(f"Fitting sparse-category {mode}", flush=True)
        model = LGBMRegressor(**params)
        model.fit(left, np.asarray(y_train) - train_offset, sample_weight=weights)
        prediction = model.predict(right) + query_offset
        model.booster_.save_model(str(destination / f"{mode}.txt"))
        np.save(destination / f"{mode}.npy", prediction)
        predictions.append(prediction)
    (destination / "categories.json").write_text(json.dumps(vocabulary))
    normal = np.mean(predictions, axis=0)
    stale = np.mean([
        np.load(folder / f"short_recent_lgbm_hierarchical_6_{mode}_42.npy")
        for mode in ("direct", "residual")
    ], axis=0)
    prediction = (normal + stale) / 2
    np.save(destination / "ensemble.npy", prediction)
    metrics = dict(candidate="lgbm_sparse_categories", seconds=time.perf_counter() - started,
                   **regression_metrics(truth, prediction))
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
