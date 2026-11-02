"""Refit the four-head stock-feature probe on all official training labels."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from features import build_block_lookup
from hdb_attribute_features_20261102 import property_features
from modeling import build_features, build_training_features, load_config, predict_model
from research_20261102 import checked_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stock_probe_full_20261102"
SUBMISSION = ROOT / "submissions/submission_stock_probe4_20261102.csv"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source, test = checked_data(ROOT / "data/raw")
    raw = source.drop(columns="MONTHLY_RENT")
    y = source.MONTHLY_RENT
    sample = pd.DataFrame({"Id": np.arange(len(test)), "Predicted": np.zeros(len(test))})
    assert sample.columns.tolist() == ["Id", "Predicted"]
    assert sample.Id.tolist() == list(range(len(test))) and len(test) == 50000
    property_path = ROOT / "data/external/attribute_sources_20261102/property.csv"
    properties = pd.read_csv(property_path)
    config = load_config()
    protocol = {
        "purpose": "User-authorized single Kaggle probe, not automatic model promotion",
        "local_validation": "stock_snapshot RMSE 441.78297566301273 vs matched CPU reference 441.84292024142957; advancement gate failed",
        "recipe": "seed42, equal four heads; stock features on local_trend:0 direct/residual only; hierarchical:6 direct/residual unchanged",
        "train_rows": len(raw), "test_rows": len(test), "config": config,
        "seed": 42, "task_type": "CPU", "threads": 8,
        "information": "Official training labels only; current building snapshot covariates; no external rental targets or test-label lookup",
        "property_sha256": sha256(property_path),
        "raw_sha256": {name: sha256(ROOT / "data/raw" / name)
                       for name in ["train.csv", "test.csv"]},
        "submission_schema": "Id,Predicted; zero-based positional IDs per existing verified final-model metadata",
        "code_sha256": {name: sha256(ROOT / "src" / name) for name in
                        ["submit_stock_probe_20261102.py", "features.py", "modeling.py",
                         "hdb_attribute_features_20261102.py"]},
        "clipping": "none", "preserve_mainline": True,
        "comparison_limit": "Earlier Kaggle mainline uses two seeds and GPU; this probe uses one seed and CPU as in its local screen. Score difference cannot isolate stock-feature value.",
    }
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Protocol changed; choose a new output folder")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    lookup = build_block_lookup(ROOT / "data/raw/auxiliary")
    components, predictions = [], []
    for recipe, lag, enrich in [("local_trend", 0, True), ("hierarchical", 6, False)]:
        cache = OUT / f"{recipe}_{lag}_features.pkl"
        if cache.exists():
            fit_x, query_x, fit_y, weights = pd.read_pickle(cache)
        else:
            print(f"Building full-data {recipe}:{lag} features", flush=True)
            fit_x, fit_y, weights = build_training_features(
                raw, y, lookup, dict(config, history=recipe, augmentation_lag=lag))
            query_x = build_features(raw, y, test, lookup, recipe)
            if enrich:
                fit_x = fit_x.join(property_features(raw, properties))
                query_x = query_x.join(property_features(test, properties))
            pd.to_pickle((fit_x, query_x, fit_y, weights), cache)
        print(f"Ready {recipe}: {fit_x.shape}, total weight {weights.sum()}", flush=True)
        for mode in ["direct", "residual"]:
            stem = f"{recipe}_{lag}_{mode}_42"
            model_path = OUT / f"{stem}.cbm"
            config_head = dict(config, residual_target=mode == "residual")
            started = time.perf_counter()
            if model_path.exists():
                model = CatBoostRegressor()
                model.load_model(str(model_path))
                print(f"Reusing {stem}", flush=True)
            else:
                offset = fit_x.ADAPT_ADDRESS_FLAT_AREA_LEVEL.to_numpy() if mode == "residual" else 0
                print(f"Fitting {stem}", flush=True)
                model = CatBoostRegressor(
                    loss_function="RMSE", iterations=config["iterations"],
                    learning_rate=config["learning_rate"], depth=config["depth"],
                    l2_leaf_reg=config["l2_leaf_reg"], random_seed=42,
                    task_type="CPU", thread_count=8, allow_writing_files=False,
                    verbose=200,
                )
                model.fit(fit_x, np.asarray(fit_y)-offset, sample_weight=weights,
                          cat_features=fit_x.select_dtypes(include=["object", "string", "category"]).columns.tolist())
                model.save_model(str(model_path))
            pred = predict_model(model, query_x, config_head)
            assert pred.shape == (50000,) and np.isfinite(pred).all()
            restored = CatBoostRegressor()
            restored.load_model(str(model_path))
            np.testing.assert_allclose(predict_model(restored, query_x, config_head), pred, rtol=0, atol=1e-9)
            np.save(OUT / f"{stem}.npy", pred)
            components.append(dict(name=stem, feature_count=fit_x.shape[1],
                                   fit_rows=len(fit_x), model_sha256=sha256(model_path),
                                   elapsed_seconds=time.perf_counter()-started))
            predictions.append(pred)
            (OUT / "progress.json").write_text(json.dumps(components, indent=2))
            print(f"Completed {stem}; {len(components)}/4", flush=True)
    sample["Predicted"] = np.mean(predictions, axis=0)
    SUBMISSION.parent.mkdir(exist_ok=True)
    if SUBMISSION.exists():
        np.testing.assert_allclose(pd.read_csv(SUBMISSION).Predicted, sample.Predicted, rtol=0, atol=1e-9)
    else:
        with SUBMISSION.open("x") as handle:
            sample.to_csv(handle, index=False)
    saved = pd.read_csv(SUBMISSION)
    assert saved.Id.equals(sample.Id)
    np.testing.assert_allclose(saved.Predicted, sample.Predicted, rtol=0, atol=1e-9)
    manifest = dict(protocol=protocol, components=components, submission=str(SUBMISSION),
                    submission_sha256=sha256(SUBMISSION), rows=len(saved),
                    prediction_min=float(saved.Predicted.min()),
                    prediction_max=float(saved.Predicted.max()),
                    status="ready_for_user_authorized_kaggle_upload")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ["submission", "submission_sha256", "rows", "status"]}), flush=True)


if __name__ == "__main__":
    main()
