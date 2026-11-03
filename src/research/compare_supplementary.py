"""Supplementary fixed-recipe comparisons (2026-11-02); no noise floor is established.

Same feature code as the mainline (features.py), one seed, fixed 1,600-tree budget, no early stopping on
scoring rows. Windows mirror the competition gap. Oracle rows use validation labels and are context only.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
from src.features import (make_core_features, build_block_lookup, add_spatial_features, add_past_rent_features,
                      suspected_low_rent_mask, DATE_COLUMN)

WINDOWS = {"jan_mar25": ("2024-12", "2025-01", "2025-03"),        # comparable to repo reference 441.84
           "gap16": ("2023-11", "2023-12", "2025-03"),            # 16-month horizon like the real test set
           "long_older": ("2023-03", "2023-04", "2024-03")}
Y = "MONTHLY_RENT"
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
def log(msg, f):
    line = f"{time.strftime('%H:%M:%S')} {msg}"; print(line, flush=True); f.write(line + "\n"); f.flush()

def build(raw, lookup, cutoff, query):
    ref = raw[raw[DATE_COLUMN] <= cutoff].reset_index(drop=True)
    hist = add_past_rent_features(ref, ref[Y], query.reset_index(drop=True), detailed=True)
    hist = hist.drop(columns=[Y, DATE_COLUMN, "FURNISHED", "FEE"], errors="ignore")   # the helper echoes query columns back; never let the label in
    core = make_core_features(query.drop(columns=[Y], errors="ignore").reset_index(drop=True))
    sp = add_spatial_features(core, lookup, "all_spatial")
    X = pd.concat([sp, hist], axis=1)
    dup = X.columns[X.columns.duplicated()].tolist()
    if dup: print("dropping duplicate columns:", dup, flush=True); X = X.loc[:, ~X.columns.duplicated()]
    assert Y not in X.columns, "label leaked into features"
    for c in X.columns:
        if X[c].dtype.name in ("string", "category") or X[c].dtype == object: X[c] = X[c].astype(str).fillna("missing")
    return X

def to_cat(Xtr, Xva):
    cats = [c for c in Xtr.columns if Xtr[c].dtype == object]
    Xtr = Xtr.copy(); Xva = Xva.copy()
    for c in cats:
        t = pd.CategoricalDtype(sorted(Xtr[c].unique())); Xtr[c] = Xtr[c].astype(t); Xva[c] = Xva[c].astype(t)
    return Xtr, Xva, cats

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", nargs="+", default=["jan_mar25"]); ap.add_argument("--iters", type=int, default=1600)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--candidates", nargs="+", default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args(); out = ROOT / "artifacts/legit_ceiling_20261102"; out.mkdir(parents=True, exist_ok=True)
    logf = open(ROOT / "logs/legit_ceiling.log", "a")
    raw = pd.read_csv(ROOT / "data/raw/train.csv"); lookup = build_block_lookup(ROOT / "data/raw/auxiliary")
    from catboost import CatBoostRegressor; import lightgbm as lgb; import xgboost as xgb
    res_path = out / "results.csv"; results = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
    for w in a.windows:
        cutoff, s, e = WINDOWS[w]; tr = raw[raw[DATE_COLUMN] <= cutoff].reset_index(drop=True); va = raw[(raw[DATE_COLUMN] >= s) & (raw[DATE_COLUMN] <= e)].reset_index(drop=True)
        log(f"[{w}] train {len(tr)} rows <= {cutoff}; val {len(va)} rows {s}..{e}", logf)
        Xtr = build(raw, lookup, cutoff, tr); Xva = build(raw, lookup, cutoff, va); ytr = tr[Y].astype(float).values; yva = va[Y].astype(float).values
        drop = [c for c in ("POSTAL_CODE",) if c in Xtr]; Xtr = Xtr.drop(columns=drop); Xva = Xva.drop(columns=drop)
        cat_cols = [c for c in Xtr.columns if Xtr[c].dtype == object]
        log(f"[{w}] {Xtr.shape[1]} features, {len(cat_cols)} categorical", logf)
        preds = {}
        # ---- oracle / floor context (uses validation labels) and legit no-ML baselines
        va_ = va.copy(); va_["ab"] = va_.FLOOR_AREA_SQM // 10
        preds["ORACLE_same_month_town_ft_model_area10_median"] = va_.groupby([DATE_COLUMN, "TOWN", "FLAT_TYPE", "FLAT_MODEL", "ab"])[Y].transform("median").values
        anchor = Xva["HIST_TOWN_FLAT_RECENT_MEDIAN"].fillna(Xva["HIST_GLOBAL_MEAN"]).fillna(float(ytr.mean())).values
        preds["base_town_ft_recent_median"] = anchor
        addr = Xva["HIST_ADDRESS_FLAT_RECENT_MEDIAN"].fillna(Xva["HIST_STREET_FLAT_RECENT_MEDIAN"]).fillna(pd.Series(anchor)).values
        preds["base_comparable_address_flat"] = addr
        cands = a.candidates or ["cat_direct", "cat_resid", "cat_recent_w", "cat_clean", "cat_log", "lgbm_direct", "xgb_direct"]
        anchor_tr = Xtr["HIST_TOWN_FLAT_RECENT_MEDIAN"].fillna(Xtr["HIST_GLOBAL_MEAN"]).fillna(float(ytr.mean())).values
        age = (pd.to_datetime(cutoff) - pd.to_datetime(tr[DATE_COLUMN])).dt.days.values / 30.4
        def cb(seed=42):
            return CatBoostRegressor(loss_function="RMSE", iterations=a.iters, learning_rate=0.04, depth=8, l2_leaf_reg=6.0, random_seed=seed,
                                     thread_count=a.threads, allow_writing_files=False, verbose=False)
        for c in cands:
            t0 = time.time()
            if c == "cat_direct":
                m = cb(); m.fit(Xtr, ytr, cat_features=cat_cols); p = m.predict(Xva)
            elif c == "cat_resid":
                m = cb(); m.fit(Xtr, ytr - anchor_tr, cat_features=cat_cols); p = m.predict(Xva) + anchor
            elif c == "cat_recent_w":
                m = cb(); m.fit(Xtr, ytr, cat_features=cat_cols, sample_weight=0.5 ** (age / 24.0)); p = m.predict(Xva)
            elif c == "cat_clean":
                keep = ~suspected_low_rent_mask(tr, tr[Y], 0.5); log(f"[{w}] cat_clean drops {int((~keep).sum())} suspected low-rent rows", logf)
                m = cb(); m.fit(Xtr[keep], ytr[keep], cat_features=cat_cols); p = m.predict(Xva)
            elif c == "cat_log":
                m = cb(); m.fit(Xtr, np.log(ytr), cat_features=cat_cols); r = np.log(ytr) - m.predict(Xtr); p = np.exp(m.predict(Xva)) * float(np.mean(np.exp(r)))
            elif c == "lgbm_direct":
                A, B, _ = to_cat(Xtr, Xva)
                m = lgb.LGBMRegressor(objective="regression", n_estimators=a.iters, learning_rate=0.04, num_leaves=127, min_child_samples=50, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, reg_lambda=6.0, cat_smooth=20, random_state=42, n_jobs=a.threads, verbose=-1)
                m.fit(A, ytr); p = m.predict(B)
            elif c == "xgb_direct":
                A, B, _ = to_cat(Xtr, Xva)
                m = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=a.iters, learning_rate=0.04, max_depth=8, reg_lambda=6.0, subsample=0.8, colsample_bytree=0.8,
                                     min_child_weight=5, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, random_state=42, n_jobs=a.threads)
                m.fit(A, ytr); p = m.predict(B)
            else: raise ValueError(c)
            preds[c] = np.asarray(p, float); log(f"[{w}] {c}: RMSE {rmse(p, yva):.2f}  ({time.time()-t0:.0f}s)", logf)
        # ---- fixed, pre-specified equal-weight blends (no weight search)
        for name, parts in {"blend_direct_resid": ["cat_direct", "cat_resid"], "blend_cat3_lgbm": ["cat_direct", "cat_resid", "cat_recent_w", "lgbm_direct"],
                            "blend_all_trees": ["cat_direct", "cat_resid", "cat_recent_w", "cat_clean", "lgbm_direct", "xgb_direct"]}.items():
            if all(k in preds for k in parts): preds[name] = np.mean([preds[k] for k in parts], axis=0)
        P = pd.DataFrame(preds); P.insert(0, "y", yva); P.insert(0, DATE_COLUMN, va[DATE_COLUMN].values); P.to_csv(out / f"preds_{w}{a.tag}.csv", index=False)
        for k, p in preds.items():
            results.append({"window": w, "candidate": k, "rmse": round(rmse(p, yva), 3), "mae": round(float(np.mean(np.abs(p - yva))), 2),
                            "bias": round(float(np.mean(p - yva)), 2), "n": len(yva), "iters": a.iters, "tag": a.tag})
            log(f"[{w}] {k:48s} RMSE {rmse(p, yva):8.2f}  bias {np.mean(p - yva):+7.1f}", logf)
        pd.DataFrame(results).to_csv(res_path, index=False)
    log("done", logf)

if __name__ == "__main__": main()
