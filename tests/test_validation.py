import numpy as np
import pandas as pd

from src.validation import regression_metrics, rolling_time_folds


def test_rolling_time_folds_are_ordered_and_disjoint() -> None:
    frame = pd.DataFrame(
        {"RENT_APPROVAL_DATE": pd.period_range("2024-01", periods=12, freq="M").astype(str)}
    )
    folds = rolling_time_folds(frame, validation_months=2, folds=3)
    assert [fold.name for fold in folds] == [
        "2024-07_to_2024-08",
        "2024-09_to_2024-10",
        "2024-11_to_2024-12",
    ]
    for fold in folds:
        assert not np.any(fold.train_mask & fold.validation_mask)
        assert fold.train_mask.sum() > 0
        assert fold.validation_mask.sum() == 2


def test_regression_metrics_include_signed_bias() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    assert metrics["rmse"] == np.sqrt(2.5)
    assert metrics["mae"] == 1.5
    assert metrics["bias"] == 1.5
