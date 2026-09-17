"""Shared chronological validation and regression metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


@dataclass(frozen=True)
class TimeFold:
    name: str
    train_mask: np.ndarray
    validation_mask: np.ndarray
    train_end: str
    validation_start: str
    validation_end: str


def rolling_time_folds(
    frame: pd.DataFrame,
    date_column: str = "RENT_APPROVAL_DATE",
    validation_months: int = 3,
    folds: int = 3,
) -> list[TimeFold]:
    """Create expanding-window folds ending at the latest observed month."""
    dates = pd.to_datetime(frame[date_column], format="%Y-%m", errors="raise")
    months = pd.DatetimeIndex(sorted(dates.unique()))
    required = validation_months * folds + 1
    if len(months) < required:
        raise ValueError(f"Need at least {required} unique months, found {len(months)}")
    result: list[TimeFold] = []
    for offset in range(folds - 1, -1, -1):
        end_position = len(months) - offset * validation_months
        start_position = end_position - validation_months
        validation_start = months[start_position]
        validation_end = months[end_position - 1]
        train_mask = (dates < validation_start).to_numpy()
        validation_mask = ((dates >= validation_start) & (dates <= validation_end)).to_numpy()
        train_end = months[start_position - 1]
        result.append(
            TimeFold(
                name=f"{validation_start:%Y-%m}_to_{validation_end:%Y-%m}",
                train_mask=train_mask,
                validation_mask=validation_mask,
                train_end=train_end.strftime("%Y-%m"),
                validation_start=validation_start.strftime("%Y-%m"),
                validation_end=validation_end.strftime("%Y-%m"),
            )
        )
    return result


def regression_metrics(y_true: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(prediction) - np.asarray(y_true)
    return {
        "rmse": float(root_mean_squared_error(y_true, prediction)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "bias": float(residual.mean()),
    }


def subgroup_metrics(
    metadata: pd.DataFrame,
    y_true: pd.Series,
    prediction: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    """Return count, RMSE, MAE, and bias for requested categorical groups."""
    work = metadata[columns].copy()
    work["_TRUE"] = np.asarray(y_true)
    work["_PRED"] = np.asarray(prediction)
    work["_ERROR"] = work["_PRED"] - work["_TRUE"]
    rows: list[dict[str, object]] = []
    for column in columns:
        for value, group in work.groupby(column, dropna=False):
            rows.append(
                {
                    "group_column": column,
                    "group_value": value,
                    "rows": len(group),
                    "rmse": float(np.sqrt(np.mean(group["_ERROR"] ** 2))),
                    "mae": float(np.mean(np.abs(group["_ERROR"]))),
                    "bias": float(group["_ERROR"].mean()),
                }
            )
    return pd.DataFrame(rows)
