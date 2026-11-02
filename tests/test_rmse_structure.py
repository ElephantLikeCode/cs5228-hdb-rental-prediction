"""Loss-equivalence and temporal information checks for the recorded probes."""

import csv

import numpy as np
import pandas as pd

from src.probe_historical_augmentation_20261102 import historical_records
from src.probe_rmse_structure_20261102 import aggregate_duplicates


def test_duplicate_aggregation_preserves_prediction_dependent_loss():
    features = pd.DataFrame({"place": ["a", "a", "b", "b", "b"], "area": [60, 60, 90, 90, 90]})
    target = np.array([750., 3100., 1200., 2500., 4000.])
    _, means, counts = aggregate_duplicates(features, target)
    within = np.sum((target-np.repeat(means, counts.astype(int)))**2)
    for prediction in [np.array([2000., 2800.]), np.array([1000., 3500.])]:
        original = np.sum((target-np.repeat(prediction, counts.astype(int)))**2)
        compressed = np.sum(counts*(means-prediction)**2)
        np.testing.assert_allclose(original, compressed+within)


def test_scale_squared_weights_preserve_currency_squared_loss():
    scale = np.array([1200., 2600., 3500.])
    target = np.array([750., 3000., 4000.])
    normalized_prediction = np.array([.8, 1., 1.1])
    normalized_loss = (scale/3000)**2*(target/scale-normalized_prediction)**2
    currency_loss = ((target-scale*normalized_prediction)/3000)**2
    np.testing.assert_allclose(normalized_loss, currency_loss)


def test_historical_loader_never_parses_excluded_period_targets(tmp_path):
    path = tmp_path / "public.csv"
    fields = ["rent_approval_date", "town", "block", "street_name", "flat_type", "monthly_rent"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for date, rent in [("2024-12", "2500"), ("2025-01", "MUST_NOT_BE_PARSED")]:
            writer.writerow(dict(rent_approval_date=date, town="Test", block="1",
                                 street_name="Test Rd", flat_type="3-room", monthly_rent=rent))
    result = historical_records(path, "2025-01")
    assert sum(sum(values.values()) for values in result.values()) == 1
    assert next(iter(result.values()))[2500.] == 1
