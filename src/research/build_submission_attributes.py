"""Abandoned eight-head prototype: no submission artifact was produced by this script.

Original design (2026-11-02 evening): the mainline eight-component CatBoost ensemble
(configs/model.json: local_trend lag 0 + hierarchical lag 6 views x direct/residual x seeds 42/2026) with Codex's 16
building-stock snapshot features (hdb_attribute_features_20261102.property_features) appended to every component.
Validation evidence: the two-head replacement improved the four-head CPU reference by 0.06 RMSE on Jan-Mar 2025
(artifacts/attribute_study_20261102/paired_comparison.csv) - inside noise; this file exists because the user wants to
try it on the public leaderboard, not because the pre-registered advancement rule passed. Nothing uploaded here."""
import json, sys, time
if __name__ != "__main__" or "--allow-abandoned-prototype" not in sys.argv:
    raise SystemExit("Abandoned prototype; not a maintained submission entry. Explicit --allow-abandoned-prototype is required.")
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
from src.modeling import load_config, build_training_features, build_features
from src.features import build_block_lookup
from src.research.compare_models import fit_component
from src.research.hdb_attribute_features import property_features
OUT = ROOT / "artifacts/submission_attributes_20261102"; OUT.mkdir(parents=True, exist_ok=True)
THREADS = 8
train_full = pd.read_csv(ROOT / "data/raw/train.csv"); test = pd.read_csv(ROOT / "data/raw/test.csv")
y = train_full["MONTHLY_RENT"]; train = train_full.drop(columns="MONTHLY_RENT")
prop = pd.read_csv(ROOT / "data/external/attribute_sources_20261102/property.csv")
lookup = build_block_lookup(ROOT / "data/raw/auxiliary"); config = load_config()
extra_test = property_features(test, prop); print("test extra features", extra_test.shape, "missing share", float(extra_test.EXT_STOCK_MISSING.mean()), flush=True)
preds = []; log = []
for view in config["history_views"]:
    recipe, lag = view["history"], view["lag"]
    t0 = time.time(); fit_x, fit_y, fit_w = build_training_features(train, y, lookup, dict(config, history=recipe, augmentation_lag=lag))
    fit_raw = train.copy()
    if lag:   # same duplication as research_20261102.py / train_final_model.py
        dates = pd.to_datetime(train.RENT_APPROVAL_DATE); months = dates.dt.year * 12 + dates.dt.month
        fit_raw = pd.concat([train, train.loc[months.ge(months.min() + lag + 6)]], ignore_index=True)
    assert len(fit_raw) == len(fit_x) == len(fit_y) == len(fit_w)
    query_x = build_features(train, y, test, lookup, recipe)
    extra_fit = property_features(fit_raw, prop); assert len(extra_fit) == len(fit_x)
    a, b = fit_x.join(extra_fit), query_x.join(extra_test)
    print(f"view {recipe} lag {lag}: features {a.shape[1]} built in {time.time()-t0:.0f}s", flush=True)
    for seed in config["seeds"]:
        for mode in ["direct", "residual"]:
            t1 = time.time(); model, p, _ = fit_component(a, fit_y, b, learner="catboost", mode=mode, seed=seed, config=config, threads=THREADS, sample_weight=fit_w)
            stem = f"{recipe}_{lag}_{mode}_{seed}"; model.save_model(str(OUT / f"{stem}.cbm")); np.save(OUT / f"{stem}.npy", p); preds.append(p)
            log.append({"component": stem, "seconds": round(time.time() - t1), "pred_mean": float(p.mean())}); print(log[-1], flush=True)
pred = np.clip(np.mean(preds, axis=0), 300, 15000)
sub = pd.DataFrame({"Id": np.arange(len(test)), "Predicted": pred})
assert len(sub) == 50000 and sub.Id.tolist() == list(range(50000)) and np.isfinite(sub.Predicted).all()
sub.to_csv(OUT / "submission_mainline8_plus_stock_snapshot.csv", index=False)
json.dump({"components": log, "n_heads": len(preds), "extra_features": [c for c in extra_test.columns], "config": config,
           "pred_summary": sub.Predicted.describe().round(1).to_dict()}, open(OUT / "run.json", "w"), indent=1)
print("written", OUT / "submission_mainline8_plus_stock_snapshot.csv"); print(sub.Predicted.describe().round(1).to_string())
