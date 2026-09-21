"""Shared feature recipes and fixed-budget models for evaluation and submission."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

try:
    from .features import (add_past_rent_features, add_spatial_features,
                           make_core_features, prepare_for_catboost,
                           add_adaptive_rent_features)
except ImportError:
    from features import (add_past_rent_features, add_spatial_features,
                          make_core_features, prepare_for_catboost,
                          add_adaptive_rent_features)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "model.json"
BASE_CONFIG = dict(history="legacy", residual_target=False, iterations=1600,
                   learning_rate=0.04, depth=8, l2_leaf_reg=6.0,
                   seeds=[42, 2026], outlier_policy="keep")
CANDIDATES = {
    "legacy": dict(BASE_CONFIG, name="legacy"),
    "recent": dict(BASE_CONFIG, name="recent", history="recent"),
    "hierarchical": dict(BASE_CONFIG, name="hierarchical", history="hierarchical"),
    "market_residual": dict(BASE_CONFIG, name="market_residual",
                            history="hierarchical", residual_target=True),
    "hierarchical_depth6": dict(BASE_CONFIG, name="hierarchical_depth6",
                                history="hierarchical", depth=6),
    "legacy_depth6": dict(BASE_CONFIG, name="legacy_depth6", depth=6),
}
for family, extra in [("horizon", {"augmentation_lag": 6}),
                      ("weighted", {"weight_half_life": 18.0})]:
    for mode in ("direct", "residual"):
        name = f"{family}_{mode}"
        CANDIDATES[name] = dict(BASE_CONFIG, name=name, history="hierarchical",
                                residual_target=(mode == "residual"), **extra)
CANDIDATES["relative_rent"] = dict(BASE_CONFIG, name="relative_rent", history="hierarchical",
                                   residual_target=True, normalize_target=True)


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = dict(BASE_CONFIG, **json.loads(path.read_text(encoding="utf-8")))
    if config["history"] not in {"legacy", "recent", "hierarchical"}:
        raise ValueError("Unknown history recipe")
    if (config["residual_target"] or config.get("normalize_target")) and config["history"] != "hierarchical":
        raise ValueError("Residual target requires the adaptive market baseline")
    return config


def build_features(reference, target, query, lookup, recipe, history_lag=0):
    """Reference contains only training labels; query labels are never accepted."""
    reference = reference.reset_index(drop=True)
    target = pd.Series(np.asarray(target), dtype=float)
    query = query.reset_index(drop=True)
    result = add_spatial_features(make_core_features(query), lookup, "all_spatial")
    history = add_past_rent_features(
        reference, target, query, detailed=True, market_features=False,
        include_recent=(recipe == "recent"),
        history_lag=history_lag,
    )
    columns = [c for c in history if c.startswith("HIST_")]
    result[columns] = history[columns]
    if recipe == "hierarchical":
        adaptive = add_adaptive_rent_features(reference, target, query, history_lag=history_lag)
        result[adaptive.columns] = adaptive
    return prepare_for_catboost(result)


def build_training_features(reference, target, lookup, config):
    """Optionally simulate stale histories; never duplicate a row's total weight.

    A six-month lag hides the most recent six months of reference labels while
    retaining the true query date. Eligible original/augmented rows each carry
    half the row's original weight. Query/validation labels are never accepted.
    """
    reference = reference.reset_index(drop=True)
    target = pd.Series(np.asarray(target), dtype=float)
    features = build_features(reference, target, reference, lookup, config["history"])
    dates = pd.to_datetime(reference["RENT_APPROVAL_DATE"])
    month = dates.dt.year * 12 + dates.dt.month
    half_life = config.get("weight_half_life")
    if half_life is not None and half_life <= 0:
        raise ValueError("weight_half_life must be positive")
    weights = np.ones(len(target)) if half_life is None else np.exp2((month-month.max()) / half_life).to_numpy()
    lag = config.get("augmentation_lag", 0)
    if lag < 0:
        raise ValueError("augmentation_lag must be nonnegative")
    if lag:
        eligible = month.ge(month.min() + lag + 6).to_numpy()
        if eligible.any():
            stale = build_features(reference, target, reference.loc[eligible], lookup,
                                   config["history"], history_lag=lag)
            weights[eligible] *= 0.5
            weights = np.concatenate([weights, weights[eligible]])
            features = pd.concat([features, stale], ignore_index=True)
            target = pd.concat([target, target.loc[eligible]], ignore_index=True)
    return features, target, weights


def target_offset(features, config):
    if config["residual_target"]:
        return features["ADAPT_ADDRESS_FLAT_AREA_LEVEL"].to_numpy(float)
    return np.zeros(len(features))


def target_scale(features, config):
    if config.get("normalize_target"):
        return np.maximum(features["ADAPT_ADDRESS_FLAT_AREA_LEVEL"].to_numpy(float), 500.) / 3000.
    return np.ones(len(features))


def fit_model(features, target, config, seed, task_type="GPU", verbose=False, sample_weight=None):
    """No outer validation labels or early stopping: same budget at deployment."""
    model = CatBoostRegressor(
        loss_function="RMSE", iterations=config["iterations"],
        learning_rate=config["learning_rate"], depth=config["depth"],
        l2_leaf_reg=config["l2_leaf_reg"], random_seed=seed,
        thread_count=-1, task_type=task_type,
        devices="0" if task_type == "GPU" else None,
        allow_writing_files=False, verbose=verbose,
    )
    scale = target_scale(features, config)
    if config.get("normalize_target"):
        # Preserve squared rent error: scale^2 * ((y-offset)/scale - f)^2.
        sample_weight = scale**2 * (1. if sample_weight is None else sample_weight)
    model.fit(features, (np.asarray(target) - target_offset(features, config)) / scale,
              sample_weight=sample_weight,
              cat_features=features.select_dtypes(include=["object"]).columns.tolist())
    return model


def predict_model(model, features, config):
    return model.predict(features) * target_scale(features, config) + target_offset(features, config)
