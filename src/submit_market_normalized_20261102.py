"""Refit the user-selected 441.63 four-head candidate for one Kaggle probe."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from modeling import load_config
from probe_rmse_structure_20261102 import market_scales
from research_20261102 import checked_data

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/stock_probe_full_20261102"
SELECTED = ROOT / "artifacts/rmse_structure_20261102/e696b0893b2eaafe"
OUT = ROOT / "artifacts/market_normalized_full_20261102"
SUBMISSION = ROOT / "submissions/submission_market_normalized_damped6_20261102.csv"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    source, test = checked_data(ROOT / "data/raw")
    raw = source.drop(columns="MONTHLY_RENT")
    target = source.MONTHLY_RENT.to_numpy(float)
    previous = json.loads((PARENT / "manifest.json").read_text())
    for name, expected in previous["protocol"]["raw_sha256"].items():
        assert digest(ROOT / "data/raw" / name) == expected
    for name in ["features.py", "modeling.py"]:
        assert digest(ROOT / "src" / name) == previous["protocol"]["code_sha256"][name]
    fit_x, query_x, cached_y, _ = pd.read_pickle(PARENT / "local_trend_0_features.pkl")
    np.testing.assert_array_equal(cached_y, target)
    # Restore precisely the 78 original features, without the previous stock probe.
    stock_columns = [c for c in fit_x if c.startswith("EXT_STOCK_")]
    assert len(stock_columns) == 16
    fit_x, query_x = fit_x.drop(columns=stock_columns), query_x.drop(columns=stock_columns)
    assert fit_x.shape == (150000, 78) and query_x.shape == (50000, 78)
    config = load_config()
    protocol = dict(
        user_request="Explicit request to submit the 441.63 market-normalized candidate despite the local screening gate not passing",
        candidate="market_normalized_damped6_replacement", local_rmse=441.629997,
        reference_local_rmse=441.84292024142957,
        composition="Equal four heads: normalized-target local direct + unchanged local residual + hierarchical lag6 direct/residual; seed42 CPU",
        train_rows=len(raw), test_rows=len(test), config=config,
        new_fits=2, reused_hierarchical_fits=2,
        information="Only original official training labels; no external rental targets or new historical augmentation; no stock snapshot features",
        raw_sha256=previous["protocol"]["raw_sha256"],
        code_sha256={name: digest(ROOT/"src"/name) for name in
                     ["submit_market_normalized_20261102.py", "probe_rmse_structure_20261102.py", "features.py", "modeling.py"]},
        clipping="none", default_config_changed=False,
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    train_scale, future_scales = market_scales(raw, target, test)
    drop = [c for c in fit_x if c in {"RENT_YEAR", "MONTH_INDEX", "ADAPT_MARKET_LEVEL", "ADAPT_HORIZON"}
            or c.startswith("HIST_") and not c.endswith("LOG_COUNT")
            or c.startswith("ADAPT_") and c.endswith("_LEVEL")]
    selected_model = CatBoostRegressor()
    selected_model.load_model(str(SELECTED / "short_recent_market_normalized.cbm"))
    assert fit_x.drop(columns=drop).columns.tolist() == selected_model.feature_names_
    specifications = [
        ("market_normalized", fit_x.drop(columns=drop), query_x.drop(columns=drop),
         target/train_scale, (train_scale/3000)**2, future_scales["damped6"], np.zeros(len(test))),
        ("local_residual", fit_x, query_x,
         target-fit_x.ADAPT_ADDRESS_FLAT_AREA_LEVEL.to_numpy(), np.ones(len(target)),
         np.ones(len(test)), query_x.ADAPT_ADDRESS_FLAT_AREA_LEVEL.to_numpy()),
    ]
    components, predictions = [], []
    for name, left, right, y, weight, scale, offset in specifications:
        file = OUT / f"{name}_42.cbm"
        model = CatBoostRegressor(
            iterations=1600, learning_rate=.04, depth=8, l2_leaf_reg=6,
            random_seed=42, loss_function="RMSE", task_type="CPU", thread_count=8,
            allow_writing_files=False, verbose=400)
        if file.exists():
            model.load_model(str(file))
        else:
            print(f"Fitting {name}: {left.shape}", flush=True)
            model.fit(left, y, sample_weight=weight,
                      cat_features=left.select_dtypes(include=["object", "string", "category"]).columns.tolist())
            model.save_model(str(file))
        pred = model.predict(right)*scale+offset
        restored = CatBoostRegressor()
        restored.load_model(str(file))
        np.testing.assert_allclose(restored.predict(right)*scale+offset, pred, rtol=0, atol=1e-9)
        np.save(OUT / f"{name}_42.npy", pred)
        components.append(dict(name=name, model=str(file), model_sha256=digest(file), reused=False))
        predictions.append(pred)
        print(f"Completed {name}", flush=True)
    for mode in ["direct", "residual"]:
        stem = f"hierarchical_6_{mode}_42"
        file = PARENT / f"{stem}.cbm"
        expected = next(item for item in previous["components"] if item["name"] == stem)
        assert digest(file) == expected["model_sha256"]
        predictions.append(np.load(PARENT / f"{stem}.npy"))
        components.append(dict(name=stem, model=str(file), model_sha256=digest(file), reused=True))
    prediction = np.mean(predictions, axis=0)
    assert prediction.shape == (50000,) and np.isfinite(prediction).all()
    frame = pd.DataFrame({"Id": np.arange(len(test)), "Predicted": prediction})
    SUBMISSION.parent.mkdir(exist_ok=True)
    if SUBMISSION.exists():
        np.testing.assert_allclose(pd.read_csv(SUBMISSION).Predicted, prediction, rtol=0, atol=1e-9)
    else:
        with SUBMISSION.open("x") as handle:
            frame.to_csv(handle, index=False)
    reloaded = pd.read_csv(SUBMISSION)
    assert reloaded.columns.tolist() == ["Id", "Predicted"] and reloaded.Id.tolist() == list(range(50000))
    np.testing.assert_allclose(reloaded.Predicted, prediction, rtol=0, atol=1e-9)
    archive = SUBMISSION.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.write(SUBMISSION, arcname=SUBMISSION.name)
    with zipfile.ZipFile(archive) as package:
        assert package.namelist() == [SUBMISSION.name]
        assert package.read(SUBMISSION.name) == SUBMISSION.read_bytes()
    manifest = dict(protocol=protocol, components=components, submission=str(SUBMISSION),
                    submission_sha256=digest(SUBMISSION), prediction_archive=str(archive),
                    prediction_archive_sha256=digest(archive), test_rows=len(frame),
                    prediction_min=float(prediction.min()), prediction_max=float(prediction.max()),
                    status="ready_for_user_authorized_kaggle_upload")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ["submission", "submission_sha256", "test_rows", "status"]}),flush=True)


if __name__ == "__main__":
    main()
