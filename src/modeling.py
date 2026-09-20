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


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = dict(BASE_CONFIG, **json.loads(path.read_text(encoding="utf-8")))
    if config["history"] not in {"legacy", "recent", "hierarchical"}:
        raise ValueError("Unknown history recipe")
    if config["residual_target"] and config["history"] != "hierarchical":
        raise ValueError("Residual target requires the adaptive market baseline")
    return config


def build_features(reference, target, query, lookup, recipe):
    """Reference contains only training labels; query labels are never accepted."""
    reference = reference.reset_index(drop=True)
    target = pd.Series(np.asarray(target), dtype=float)
    query = query.reset_index(drop=True)
    result = add_spatial_features(make_core_features(query), lookup, "all_spatial")
    history = add_past_rent_features(
        reference, target, query, detailed=True, market_features=False,
        include_recent=(recipe == "recent"),
    )
    columns = [c for c in history if c.startswith("HIST_")]
    result[columns] = history[columns]
    if recipe == "hierarchical":
        adaptive = add_adaptive_rent_features(reference, target, query)
        result[adaptive.columns] = adaptive
    return prepare_for_catboost(result)


def target_offset(features, config):
    if config["residual_target"]:
        return features["ADAPT_ADDRESS_FLAT_AREA_LEVEL"].to_numpy(float)
    return np.zeros(len(features))


def fit_model(features, target, config, seed, task_type="GPU", verbose=False):
    """No outer validation labels or early stopping: same budget at deployment."""
    model = CatBoostRegressor(
        loss_function="RMSE", iterations=config["iterations"],
        learning_rate=config["learning_rate"], depth=config["depth"],
        l2_leaf_reg=config["l2_leaf_reg"], random_seed=seed,
        thread_count=-1, task_type=task_type,
        devices="0" if task_type == "GPU" else None,
        allow_writing_files=False, verbose=verbose,
    )
    model.fit(features, np.asarray(target) - target_offset(features, config),
              cat_features=features.select_dtypes(include=["object"]).columns.tolist())
    return model


def predict_model(model, features, config):
    return model.predict(features) + target_offset(features, config)
