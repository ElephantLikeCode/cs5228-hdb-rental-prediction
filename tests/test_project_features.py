from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_past_rent_features,
    add_spatial_features,
    build_block_lookup,
    make_core_features,
    suspected_low_rent_mask,
    add_adaptive_rent_features,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def test_core_and_spatial_features_have_full_coverage() -> None:
    train = pd.read_csv(RAW / "train.csv", nrows=200)
    core = make_core_features(train.drop(columns=["MONTHLY_RENT"]))
    lookup = build_block_lookup(RAW / "auxiliary")
    spatial = add_spatial_features(core, lookup)
    expected = {
        "RENT_YEAR", "RENT_MONTH", "RENT_QUARTER", "MONTH_INDEX", "LEASE_AGE",
        "ADDRESS", "LATITUDE", "LONGITUDE", "DIST_MRT_KM", "DIST_MALL_KM",
        "DIST_SCHOOL_KM", "COUNT_MRT_1P0KM", "COUNT_MALL_1P0KM",
        "COUNT_SCHOOL_1P0KM",
    }
    assert expected.issubset(spatial.columns)
    assert spatial[list(expected)].notna().all().all()
    assert len(spatial) == len(train)


def test_core_features_canonicalize_equivalent_flat_types() -> None:
    frame = pd.DataFrame(
        {
            "RENT_APPROVAL_DATE": ["2021-01", "2022-01"],
            "TOWN": ["ANG MO KIO", "ang mo kio"],
            "BLOCK": ["1", "1"],
            "STREET": ["TEST ROAD", "test road"],
            "FLAT_TYPE": ["3 room", "3-room"],
            "FLAT_MODEL": ["Model A", "model-a"],
            "FLOOR_AREA_SQM": [70, 70],
            "LEASE_COMMENCE_DATE": [1980, 1980],
            "FURNISHED": ["yes", "yes"],
            "FEE": [0, 0],
        }
    )
    features = make_core_features(frame)
    assert features["FLAT_TYPE"].nunique() == 1
    assert features["FLAT_MODEL"].nunique() == 1
    assert features["TOWN"].nunique() == 1


def test_suspected_low_rent_rule_is_relative_and_room_aware() -> None:
    frame = pd.DataFrame(
        {
            "RENT_APPROVAL_DATE": ["2025-01"] * 5,
            "TOWN": ["A"] * 5,
            "FLAT_TYPE": ["4 room", "4-room", "4 room", "4 room", "2 room"],
        }
    )
    target = pd.Series([4000.0, 4200.0, 3800.0, 1000.0, 500.0])
    mask = suspected_low_rent_mask(frame, target, ratio_threshold=0.5)
    assert mask.tolist() == [False, False, False, True, False]


def test_spatial_feature_groups_are_strictly_additive() -> None:
    train = pd.read_csv(RAW / "train.csv", nrows=20)
    core = make_core_features(train.drop(columns=["MONTHLY_RENT"]))
    lookup = build_block_lookup(RAW / "auxiliary")
    block = add_spatial_features(core, lookup, "block")
    mrt = add_spatial_features(core, lookup, "block_mrt")
    all_spatial = add_spatial_features(core, lookup, "all_spatial")
    assert "LATITUDE" in block and "DIST_MRT_KM" not in block
    assert "DIST_MRT_KM" in mrt and "DIST_MALL_KM" not in mrt
    assert {"DIST_MRT_KM", "DIST_MALL_KM", "DIST_SCHOOL_KM"}.issubset(all_spatial)
    assert len(core) == len(block) == len(mrt) == len(all_spatial)


def test_past_rent_features_exclude_same_and_future_months() -> None:
    frame = pd.DataFrame(
        {
            "RENT_APPROVAL_DATE": ["2024-01", "2024-01", "2024-02", "2024-03"],
            "TOWN": ["a", "b", "a", "a"],
            "BLOCK": ["1", "2", "1", "1"],
            "STREET": ["road", "road", "road", "road"],
            "FLAT_TYPE": ["3 room"] * 4,
            "FLAT_MODEL": ["model a"] * 4,
        }
    )
    target = pd.Series([1000.0, 3000.0, 5000.0, 9000.0])
    features = add_past_rent_features(frame, target, frame, smoothing=0.0)
    assert pd.isna(features.loc[0, "HIST_GLOBAL_MEAN"])
    assert features.loc[2, "HIST_GLOBAL_MEAN"] == 2000.0
    assert features.loc[2, "HIST_ADDRESS_MEAN"] == 1000.0
    assert features.loc[2, "HIST_ADDRESS_LOG_COUNT"] == pytest.approx(
        np.log1p(1)
    )
    assert features.loc[3, "HIST_GLOBAL_MEAN"] == 3000.0
    assert features.loc[3, "HIST_ADDRESS_MEAN"] == 3000.0


def test_detailed_history_separates_flat_types_within_address() -> None:
    frame = pd.DataFrame(
        {
            "RENT_APPROVAL_DATE": ["2024-01", "2024-01", "2024-02"],
            "TOWN": ["a", "a", "a"],
            "BLOCK": ["1", "1", "1"],
            "STREET": ["road", "road", "road"],
            "FLAT_TYPE": ["3 room", "5-room", "3-room"],
            "FLAT_MODEL": ["model a"] * 3,
            "FLOOR_AREA_SQM": [70, 110, 70],
        }
    )
    target = pd.Series([2000.0, 4000.0, 2500.0])
    features = add_past_rent_features(
        frame, target, frame, smoothing=0.0, detailed=True
    )
    assert features.loc[2, "HIST_ADDRESS_MEAN"] == 3000.0
    assert features.loc[2, "HIST_ADDRESS_FLAT_MEAN"] == 2000.0
    assert features.loc[2, "HIST_ADDRESS_FLAT_AREA_MEAN"] == 2000.0


def test_adaptive_features_ignore_same_month_and_future_labels():
    frame = pd.DataFrame({
        "RENT_APPROVAL_DATE": ["2024-01", "2024-01", "2024-02", "2024-03"],
        "TOWN": ["a"] * 4, "BLOCK": ["1", "2", "1", "1"],
        "STREET": ["road"] * 4, "FLAT_TYPE": ["4-room"] * 4,
        "FLAT_MODEL": ["model a"] * 4, "FLOOR_AREA_SQM": [90.] * 4,
    })
    original = pd.Series([3000., 4000., 3500., 3600.])
    query = frame.iloc[[2]].copy()
    expected = add_adaptive_rent_features(frame, original, query)
    changed = original.copy()
    changed.iloc[2:] = [100000., 500000.]
    pd.testing.assert_frame_equal(expected, add_adaptive_rent_features(frame, changed, query))
    # Future rows cannot influence eligibility, decay anchor, or market estimate.
    pd.testing.assert_frame_equal(expected, add_adaptive_rent_features(frame.iloc[:2], original.iloc[:2], query))


def test_local_trend_features_separate_town_flat_momentum_without_leakage():
    months = pd.date_range("2023-01-01", periods=14, freq="MS").strftime("%Y-%m")
    rows = []
    target = []
    for i, month in enumerate(months):
        for town, premium in [("rising", 1.0 + 0.01 * i), ("flat", 1.0)]:
            rows.append(dict(RENT_APPROVAL_DATE=month, TOWN=town, BLOCK="1",
                             STREET="road", FLAT_TYPE="4 room", FLAT_MODEL="model",
                             FLOOR_AREA_SQM=90.0))
            target.append(3000.0 * premium)
    frame = pd.DataFrame(rows)
    y = pd.Series(target)
    query = frame.loc[frame.RENT_APPROVAL_DATE.eq("2024-01")].reset_index(drop=True)
    base = add_adaptive_rent_features(frame, y, query, include_local_trend=True)
    assert base.loc[0, "ADAPT_TOWN_FLAT_PREMIUM_CHANGE"] > 0
    assert base.loc[1, "ADAPT_TOWN_FLAT_PREMIUM_CHANGE"] < 0
    changed = y.copy()
    changed.loc[frame.RENT_APPROVAL_DATE.ge("2024-01")] = 100000.0
    pd.testing.assert_frame_equal(
        base, add_adaptive_rent_features(frame, changed, query, include_local_trend=True)
    )


def test_adaptive_fallbacks_and_long_horizon_keep_comparable_mass():
    frame = pd.DataFrame({
        "RENT_APPROVAL_DATE": ["2024-01", "2024-02"],
        "TOWN": ["a"] * 2, "BLOCK": ["1"] * 2,
        "STREET": ["road"] * 2, "FLAT_TYPE": ["4 room"] * 2,
        "FLAT_MODEL": ["model a"] * 2, "FLOOR_AREA_SQM": [90.] * 2,
    })
    query = pd.concat([frame.iloc[[0]], frame.iloc[[0]]], ignore_index=True)
    query["RENT_APPROVAL_DATE"] = ["2024-03", "2025-07"]
    query["BLOCK"] = "unseen"
    result = add_adaptive_rent_features(frame, pd.Series([3000., 3200.]), query)
    assert np.isfinite(result.to_numpy()).all()
    assert result.loc[0, "ADAPT_STREET_FLAT_LOG_MASS"] == result.loc[1, "ADAPT_STREET_FLAT_LOG_MASS"]
    assert result.loc[0, "ADAPT_ADDRESS_FLAT_LEVEL"] == result.loc[0, "ADAPT_STREET_FLAT_LEVEL"]
    assert result.loc[0, "ADAPT_ADDRESS_FLAT_LOG_MASS"] == 0.0
    assert result.loc[1, "ADAPT_HORIZON"] == 17


def test_saved_direct_and_residual_models_predict_in_rent_units(tmp_path):
    from catboost import CatBoostRegressor
    from src.modeling import BASE_CONFIG, fit_model, predict_model

    frame = pd.DataFrame({"area": np.arange(16) + 70.,
                          "ADAPT_ADDRESS_FLAT_AREA_LEVEL": np.arange(16) * 10. + 2000.})
    target = frame["ADAPT_ADDRESS_FLAT_AREA_LEVEL"].to_numpy() + 200. + np.arange(16) * 3.
    original = target.copy()
    for residual, normalize in [(False, False), (True, False), (True, True)]:
        config = dict(BASE_CONFIG, iterations=4, depth=2, history="hierarchical",
                      residual_target=residual, normalize_target=normalize)
        model = fit_model(frame, target, config, 42, task_type="CPU")
        path = tmp_path / f"model_{residual}_{normalize}.cbm"
        model.save_model(path)
        loaded = CatBoostRegressor()
        loaded.load_model(path)
        prediction = predict_model(loaded, frame, config)
        np.testing.assert_allclose(prediction, predict_model(model, frame, config))
        assert np.isfinite(prediction).all() and prediction.min() > 1800
    np.testing.assert_array_equal(target, original)


def test_lagged_histories_exclude_hidden_recent_labels():
    frame = pd.DataFrame({
        "RENT_APPROVAL_DATE": pd.date_range("2024-01-01", periods=9, freq="MS").strftime("%Y-%m"),
        "TOWN": "a", "BLOCK": "1", "STREET": "road", "FLAT_TYPE": "4 room",
        "FLAT_MODEL": "model", "FLOOR_AREA_SQM": 90.,
    })
    y = pd.Series(np.arange(9) * 100. + 3000.)
    query = frame.iloc[[-1]]
    changed = y.copy()
    changed.iloc[2:] = 100000.
    for builder in [add_adaptive_rent_features, add_past_rent_features]:
        expected = builder(frame, y, query, history_lag=6)
        pd.testing.assert_frame_equal(expected, builder(frame, changed, query, history_lag=6))
    adaptive = add_adaptive_rent_features(frame, y, query, history_lag=6)
    assert adaptive.loc[0, "ADAPT_HORIZON"] == 7


def test_horizon_augmentation_preserves_each_observations_weight(monkeypatch):
    import src.modeling as modeling

    def fake_features(reference, target, query, lookup, recipe, history_lag=0):
        return pd.DataFrame({"row": query["row"].to_numpy(), "lag": history_lag})

    monkeypatch.setattr(modeling, "build_features", fake_features)
    raw = pd.DataFrame({"RENT_APPROVAL_DATE": pd.date_range("2022-01-01", periods=24, freq="MS").strftime("%Y-%m"),
                        "row": np.arange(24)})
    y = pd.Series(np.arange(24) + 2000.)
    config = dict(modeling.BASE_CONFIG, augmentation_lag=6, weight_half_life=18.)
    features, targets, weights = modeling.build_training_features(raw, y, None, config)
    totals = pd.Series(weights).groupby(features.row).sum().to_numpy()
    np.testing.assert_allclose(totals, np.exp2((np.arange(24)-23)/18.))
    assert len(features) == 36 and features.lag.eq(6).sum() == 12
    np.testing.assert_array_equal(targets, y.iloc[features.row])


def test_summary_matches_rows_and_counts_all_history_components(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    import generate_figures as figures

    reports = tmp_path / "reports"
    cache = tmp_path / "artifacts" / "temporal_comparison"
    reports.mkdir()
    cache.mkdir(parents=True)
    rows = []
    for name, error, code in [("hierarchical", 10., "old"), ("market_residual", -10., "old"),
                               ("horizon_direct", 30., "new"), ("horizon_residual", 10., "new")]:
        for seed in [42, 2026]:
            run = f"{name}_{seed}"
            path = cache / f"{run}.npz"
            y = np.array([1000., 2000.])
            np.savez(path, row_id=np.array([1, 2]), y=y, prediction=y+error,
                     month=np.array(["2025-01", "2025-01"]))
            rows.append(dict(candidate=name, seed=seed, code_sha256=code, data_sha256="same",
                             window="short_recent", iterations=1600, task_type="GPU",
                             train_end="2024-12", run_id=run,
                             prediction_file=str(path.relative_to(tmp_path))))
    pd.DataFrame(rows).to_csv(reports / "temporal_comparison.csv", index=False)
    monkeypatch.setattr(figures, "ROOT", tmp_path)
    monkeypatch.setattr(figures, "REPORTS", reports)
    monkeypatch.setattr(figures, "save", lambda name: figures.plt.close())
    figures.summarize_temporal(reference_candidate="hierarchical_market_blend")
    summary = pd.read_csv(reports / "temporal_summary.csv")
    result = summary.loc[summary.candidate.eq("history_diversity_blend")].iloc[0]
    assert result.model_count == 8 and result.rmse == pytest.approx(10.)
    assert result.baseline_rmse == 0. and result.baseline_code_sha256 == "old"
    # Corrupt one saved row ID: comparisons must fail instead of pairing by position.
    bad = cache / "horizon_direct_42.npz"
    np.savez(bad, row_id=np.array([1, 3]), y=y, prediction=y+30.,
             month=np.array(["2025-01", "2025-01"]))
    with pytest.raises(ValueError, match="different validation rows"):
        figures.summarize_temporal(reference_candidate="hierarchical_market_blend")


def test_relative_rent_weighting_keeps_the_squared_rent_objective():
    from src.modeling import target_scale, target_offset, CANDIDATES

    frame = pd.DataFrame({"ADAPT_ADDRESS_FLAT_AREA_LEVEL": [2000., 3000., 4000.]})
    config = CANDIDATES["relative_rent"]
    scale, offset = target_scale(frame, config), target_offset(frame, config)
    y = np.array([2300., 2800., 4500.])
    f = np.array([100., -100., 200.])
    np.testing.assert_allclose(scale**2 * ((y-offset)/scale-f)**2, (y-(f*scale+offset))**2)


def test_final_training_overrides_require_isolated_outputs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    import train_final_model as final
    monkeypatch.setattr("sys.argv", ["train_final_model.py"])
    args = final.parse_args()
    final.validate_output_paths(args)
    for field, value in [("iterations", 5), ("candidate", "legacy"),
                         ("augmentation_lags", [0]), ("history_views", ["local_trend:0"]),
                         ("seeds", [42]),
                         ("outlier_policy", "remove"), ("outlier_ratio", 0.4)]:
        changed = final.argparse.Namespace(**vars(args))
        setattr(changed, field, value)
        with pytest.raises(ValueError, match="Experimental overrides"):
            final.validate_output_paths(changed)
    args.iterations = 5
    args.submission = ROOT / "submissions/probe.csv"
    args.metadata = ROOT / "artifacts/probe/metadata.json"
    args.model_prefix = "probe/model"
    final.validate_output_paths(args)
    args.model_prefix = "../data/raw/model"
    with pytest.raises(ValueError, match="model-prefix"):
        final.validate_output_paths(args)
    args.model_prefix = "probe/model"
    args.submission = ROOT / "data/raw/train.csv"
    with pytest.raises(ValueError, match="submission"):
        final.validate_output_paths(args)


def test_previous_run_backup_is_complete_and_checks_hashes(tmp_path, monkeypatch):
    import hashlib
    import json
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    import train_final_model as final
    monkeypatch.setattr(final, "ROOT", tmp_path)
    model = tmp_path / "artifacts/model.cbm"
    submission = tmp_path / "submissions/final_submission.csv"
    metadata = tmp_path / "reports/metadata.json"
    for path in [model, submission, metadata]:
        path.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"original model")
    submission.write_bytes(b"Id,Predicted\n0,3000\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(dict(submission_sha256=digest(submission),
        components=[dict(model_path="artifacts/model.cbm", sha256=digest(model))])))
    archive = final.archive_previous_run(metadata, submission)
    for path in [model, submission, metadata]:
        assert (archive / path.relative_to(tmp_path)).read_bytes() == path.read_bytes()
    assert final.archive_previous_run(metadata, submission) == archive
    saved_model = archive / "artifacts/model.cbm"
    saved_model.write_bytes(b"unexpected content")
    with pytest.raises(ValueError, match="Archive collision"):
        final.archive_previous_run(metadata, submission)
    saved_model.write_bytes(model.read_bytes())
    model.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="hash mismatch"):
        final.archive_previous_run(metadata, submission)
    assert saved_model.read_bytes() == b"original model"
    with pytest.raises(ValueError, match="Incomplete previous run"):
        final.archive_previous_run(tmp_path / "absent.json", submission)


def test_mainline_config_selects_eight_history_diversity_components():
    from src.modeling import load_config
    config = load_config()
    assert config["name"] == "history_diversity_blend"
    assert config["augmentation_lags"] == [0, 6]
    assert len(config["seeds"]) * len(config["target_modes"]) * len(config["augmentation_lags"]) == 8
