import numpy as np
import pandas as pd
import pytest
import hashlib
import json

from src.feedback_experiments import flat_group, nearest_predictions, rent_class, soft_prediction
from src.feedback_experiments import crossfit_statistics, premium_statistics, past_statistics
from src.market_experiments import align_market, monthly_market, correction_features, honest_residual_rows
from src.market_experiments import clip_to_training_range, method_budget
from src.market_experiments import paired_gain_interval
from src.market_experiments import align_public_macro, station_calendar, rail_features, public_macro_tables


def test_monthly_query_control_reuses_the_same_two_fitted_models(monkeypatch):
    from src import market_experiments as market
    fits = []
    def fake_fit(*args):
        fits.append(args[2]["residual_target"])
        return object()
    monkeypatch.setattr(market, "fit_model", fake_fit)
    monkeypatch.setattr(market, "predict_model", lambda model, query, cfg: query.signal.to_numpy())
    fit = pd.DataFrame({"signal": [1., 2.]})
    frozen = pd.DataFrame({"signal": [10., 20.]})
    monthly = pd.DataFrame({"signal": [20., 30.]})
    components = {(f"local_trend_{mode}", 42): np.zeros(2) for mode in ("direct", "residual")}
    values, _ = market.paired_methods(fit, [1., 2.], frozen, {}, 42, "GPU",
        np.full(2, 30.), components, {"macro_both": (fit, frozen, "catboost")},
        "public", {"macro_both": {"monthly": monthly}})
    assert fits == [False, True]
    assert values["public_macro_both_normal_view"].tolist() == [10., 20.]
    assert values["public_macro_both_monthly_normal_view"].tolist() == [20., 30.]
    assert values["public_macro_both_monthly_replace_seed"].tolist() == [35., 37.5]


def test_public_macro_release_and_origin_freeze_ignore_future_values():
    table = pd.DataFrame({"PUBLIC_CPI_LEVEL": [1., 2., 999.]},
        index=pd.to_datetime(["2024-11-01", "2025-01-01", "2025-02-01"]))
    query = pd.DataFrame({"RENT_APPROVAL_DATE": ["2024-10", "2025-01", "2025-03"]})
    frozen = align_public_macro(table, query, "2025-01")
    assert frozen.PUBLIC_CPI_LEVEL.tolist() == [0., 2., 2.]
    changed = table.copy()
    changed.iloc[-1] = -999.
    pd.testing.assert_frame_equal(frozen, align_public_macro(changed, query, "2025-01"))
    assert frozen.PUBLIC_CPI_MISSING.tolist() == [1., 0., 0.]


def test_public_macro_tables_activate_only_after_release_month(tmp_path):
    cpi = {"Data": {"row": [{"rowText": "All Items", "columns": [
        {"key": "2024 Nov", "value": "100"}, {"key": "2024 Dec", "value": "101"}]}]}}
    ura = {"result": {"total": 1, "records": [{"quarter": "2024-Q4",
        "property_type": "Non-Landed", "locality": "Whole Island", "index": "160"}]}}
    (tmp_path / "cpi.json").write_text(json.dumps(cpi))
    (tmp_path / "ura.json").write_text(json.dumps(ura))
    tables = public_macro_tables(tmp_path)
    assert tables["cpi"].index.min() == pd.Timestamp("2025-01-01")
    assert tables["ura"].index.min() == pd.Timestamp("2025-02-01")


def rail_example():
    stations = pd.DataFrame({"CODE": ["ns1", "ew1", "te4", "cr1"],
        "NAME": ["legacy", "legacy", "new", "future"], "STATUS": ["open"]*4,
        "LATITUDE": [1.3, 1.3, 1.3, 1.3], "LONGITUDE": [103.81, 103.81, 103.8, 103.8]})
    schedule = {"events": [{"codes": ["te4"], "opening_date": "2021-08-28",
        "announced_date": "2021-06-30", "source": "official"}]}
    return station_calendar(stations, schedule)


def test_rail_excludes_future_codes_and_deduplicates_interchanges():
    calendar = rail_example()
    assert pd.isna(calendar.loc[calendar.CODE.eq("cr1"), "effective"]).all()
    query = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-08", "2021-09"]})
    core = pd.DataFrame({"LATITUDE": [1.3]*2, "LONGITUDE": [103.8]*2})
    features = rail_features(query, core, calendar, events=True)
    assert features.NEAREST_MRT.tolist() == ["legacy", "new"]
    assert features.COUNT_MRT_2P0KM.tolist() == [1, 2]
    assert features.RAIL_NEW_LINE_ACCESS.iloc[0] == 0.
    assert features.RAIL_NEW_LINE_ACCESS.iloc[1] == 1.


def test_rail_forecast_cannot_use_announcement_after_origin():
    query = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-09"]})
    core = pd.DataFrame({"LATITUDE": [1.3], "LONGITUDE": [103.8]})
    assert rail_features(query, core, rail_example(), "2021-06").NEAREST_MRT.item() == "legacy"
    assert rail_features(query, core, rail_example(), "2021-07").NEAREST_MRT.item() == "new"
    start = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-01"]})
    assert rail_features(start, core, rail_example(), "2021-01").NEAREST_MRT.item() == "legacy"


def test_specialist_routing_uses_normalised_input_types():
    raw = pd.DataFrame({"FLAT_TYPE": ["3-ROOM", "4 ROOM", "5-room", "EXECUTIVE", "2 ROOM"]})
    assert flat_group(raw).tolist() == ["3 room", "4 room", "large", "large", "small"]


def test_rent_band_boundaries_and_soft_routing():
    assert rent_class([2500, 2501, 4000, 4001]).tolist() == [0, 1, 1, 2]
    probability = np.array([[.25, .5, .25]])
    assert soft_prediction(probability, np.array([[1000, 3000, 5000]])).item() == 3000
    with pytest.raises(ValueError, match="probabilities"):
        soft_prediction(np.array([[.2, .2, .2]]), np.ones((1, 3)))


def reference_example():
    train = pd.DataFrame({"FLAT_TYPE": ["4-ROOM"]*12,
                          "RENT_APPROVAL_DATE": pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m")})
    valid = pd.DataFrame({"FLAT_TYPE": ["4 ROOM"], "RENT_APPROVAL_DATE": ["2025-01"]})
    features = pd.DataFrame({"LATITUDE": [1.3]*12, "LONGITUDE": [103.8]*12,
                             "FLOOR_AREA_SQM": [90]*12, "LEASE_AGE": [20]*12})
    return train, valid, features


def test_nearest_preserves_constant_rents_without_query_labels():
    train, valid, features = reference_example()
    predictions = nearest_predictions(train, pd.Series([2000.]*12), valid, features, features.iloc[:1])
    for values in predictions.values():
        assert values == pytest.approx([2000.])


def test_nearest_rejects_same_month_reference_labels():
    train, valid, features = reference_example()
    valid["RENT_APPROVAL_DATE"] = "2024-12"
    with pytest.raises(ValueError, match="query/future"):
        nearest_predictions(train, pd.Series([2000.]*12), valid, features, features.iloc[:1])


def crossfit_example():
    n = 30
    return pd.DataFrame({"RENT_APPROVAL_DATE": [f"2024-{i%12+1:02d}" for i in range(n)],
        "TOWN": ["bedok"]*n, "BLOCK": [str(i%7) for i in range(n)],
        "STREET": ["test road"]*n, "FLAT_TYPE": ["4-ROOM"]*n,
        "FLAT_MODEL": ["improved"]*n, "FLOOR_AREA_SQM": [90.]*n,
        "LEASE_COMMENCE_DATE": [1980]*n, "FEE": [0.]*n, "FURNISHED": ["yes"]*n})


def test_crossfit_excludes_all_targets_in_query_fold():
    raw = crossfit_example()
    y = pd.Series(np.linspace(1800, 4000, len(raw)))
    a, _, folds = crossfit_statistics(raw, y, raw.iloc[:2])
    changed = y.copy()
    changed.loc[folds == 0] = 20000.
    b, _, folds_again = crossfit_statistics(raw, changed, raw.iloc[:2])
    assert np.array_equal(folds, folds_again)
    np.testing.assert_allclose(a.loc[folds == 0], b.loc[folds == 0])


def test_crossfit_keeps_identical_canonical_inputs_together():
    raw = crossfit_example()
    duplicate = raw.iloc[:1].copy()
    duplicate["FLAT_TYPE"] = "4 room"
    raw = pd.concat([raw, duplicate], ignore_index=True)
    _, _, folds = crossfit_statistics(raw, pd.Series([3000.]*len(raw)), raw.iloc[:1])
    assert folds[0] == folds[-1]


def test_premium_unseen_groups_have_finite_shrunk_fallback():
    raw = crossfit_example()
    unseen = raw.iloc[:1].copy()
    unseen["TOWN"], unseen["BLOCK"], unseen["FLAT_TYPE"] = "unknown", "999", "unknown"
    encoded = premium_statistics(raw, pd.Series([3000.]*len(raw)), unseen)
    assert np.isfinite(encoded.to_numpy()).all()
    assert encoded.filter(like="LOG_COUNT").eq(0).all().all()
    assert encoded.filter(like="MEAN").eq(1).all().all()


def test_past_comparables_exclude_query_and_later_month_targets():
    raw = crossfit_example()
    y = pd.Series(np.linspace(1800, 4000, len(raw)))
    a = past_statistics(raw, y)
    changed = y.copy()
    changed.loc[raw.RENT_APPROVAL_DATE.ge("2024-06")] = 20000.
    b = past_statistics(raw, changed)
    np.testing.assert_allclose(a.loc[raw.RENT_APPROVAL_DATE.le("2024-06")],
                               b.loc[raw.RENT_APPROVAL_DATE.le("2024-06")])


def test_market_join_uses_previous_month_and_never_backward_fills():
    table = pd.DataFrame({"MACRO_COE_a_LEVEL": [1., 2., 3.]},
                         index=pd.period_range("2024-01", periods=3, freq="M"))
    query = pd.DataFrame({"RENT_APPROVAL_DATE": ["2024-01", "2024-02", "2024-04"]})
    values = align_market(table, query)
    assert values.MACRO_COE_a_LEVEL.tolist() == [0., 1., 3.]
    assert values.MACRO_MISSING.tolist() == [1., 0., 0.]
    frozen = align_market(table, query, "2024-03")
    assert frozen.MACRO_COE_a_LEVEL.tolist() == [0., 1., 2.]
    changed = table.copy()
    changed.loc[pd.Period("2024-03"), :] = 999.
    pd.testing.assert_frame_equal(frozen, align_market(changed, query, "2024-03"))


def test_market_rolling_summaries_do_not_depend_on_future_prices(tmp_path):
    coe = pd.DataFrame({"YEAR": [2024]*4, "MONTH": ["jan", "feb", "mar", "apr"],
        "CATEGORY": ["a"]*4, "QUOTA_PREMIUM": ["$40,000"]*4,
        "PREVAILING_QUOTA_PREMIUM": ["$38,000"]*4, "QUOTA": [100]*4, "BIDS_RECEIVED": [120]*4})
    stock = pd.DataFrame({"DATE": ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"],
                          "SYMBOL": ["TEST"]*4, "CLOSE": [10., 11., 12., 13.]})
    coe.to_csv(tmp_path / "sg-coe-prices.csv", index=False)
    stock.to_csv(tmp_path / "sg-stock-prices.csv", index=False)
    original = monthly_market(tmp_path)
    stock.loc[3, "CLOSE"] = 200.
    coe.loc[3, "QUOTA_PREMIUM"] = "$999,000"
    coe.to_csv(tmp_path / "sg-coe-prices.csv", index=False)
    stock.to_csv(tmp_path / "sg-stock-prices.csv", index=False)
    changed = monthly_market(tmp_path)
    pd.testing.assert_frame_equal(original.loc[:"2024-03"], changed.loc[:"2024-03"])


def test_correction_features_do_not_include_label_based_routing():
    features = pd.DataFrame({"FLAT_TYPE": ["4 room"], "FLOOR_AREA_SQM": [90.],
        "RENT_BAND": ["low"], "MONTHLY_RENT": [600.], "ACTUAL_RATIO": [.2]})
    values = correction_features(features, np.array([3000.]))
    assert set(values) == {"FLAT_TYPE", "FLOOR_AREA_SQM", "BASE_PREDICTION"}


def test_residual_pool_excludes_outer_validation_targets_and_deduplicates():
    raw = pd.DataFrame({"RENT_APPROVAL_DATE": ["2024-01"]*1100 + ["2025-01"]*5})
    y = pd.Series(np.arange(len(raw), dtype=float))
    def fake_baseline(window, frame, target):
        ids = np.arange(len(frame))
        return ids, np.full(len(ids), 3000.), [window]
    pool = honest_residual_rows(raw, y, "2025-01", fake_baseline)
    assert len(pool) == 1100
    assert pool.row_id.is_unique
    assert raw.iloc[pool.row_id].RENT_APPROVAL_DATE.lt("2025-01").all()
    changed = y.copy()
    changed.iloc[1100:] = 99999.
    pd.testing.assert_frame_equal(pool, honest_residual_rows(raw, changed, "2025-01", fake_baseline))


def test_ridge_guard_is_fitted_only_on_training_input_ranges():
    train = pd.DataFrame({"ADAPT_HORIZON": [0., 1.], "TOWN": ["a", "b"]})
    query = pd.DataFrame({"ADAPT_HORIZON": [2., 16.], "TOWN": ["c", "d"]})
    clipped = clip_to_training_range(train, query)
    assert clipped.ADAPT_HORIZON.tolist() == [1., 1.]
    assert clipped.TOWN.tolist() == ["c", "d"]
    assert query.ADAPT_HORIZON.tolist() == [2., 16.]


def test_report_budgets_distinguish_cpu_linear_and_gpu_correction():
    assert method_budget("learner_ridge_market") == ("ridge_cpu", 0)
    assert method_budget("correction_residual_shrink25") == ("catboost", 600)
    assert method_budget("learner_xgb_market_normal_view") == ("xgboost", 1600)


def test_paired_interval_has_correct_gain_sign_and_zero_difference():
    raw = pd.DataFrame({"TOWN": ["a"]*4, "BLOCK": ["1", "1", "2", "2"], "STREET": ["road"]*4})
    y = np.full(4, 2000.)
    baseline = y + 100.
    assert paired_gain_interval(raw, np.arange(4), y, baseline, baseline, samples=100) == (0., 0.)
    assert paired_gain_interval(raw, np.arange(4), y, baseline, y, samples=100) == (100., 100.)


@pytest.mark.parametrize("bad_source", [False, True])
def test_saved_run_summary_verifies_original_source_and_repairs_cpu_metadata(tmp_path, monkeypatch, bad_source):
    from src import feedback_experiments as study
    run_id = "1111111111111111"
    folder = tmp_path / "artifacts" / "feedback_study" / run_id
    (folder / "source").mkdir(parents=True)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    for name in ("feedback_experiments.py", "market_experiments.py", "features.py", "modeling.py", "validation.py"):
        (folder / "source" / name).write_bytes(b"a")
    raw_path = tmp_path / "data" / "raw" / "train.csv"
    pd.DataFrame({"MONTHLY_RENT": [2000., 3000., 4000.]}).to_csv(raw_path, index=False)
    manifest = dict(code_sha256="bad" if bad_source else hashlib.sha256(b"aaaaa").hexdigest(),
        data_sha256=study.file_hash(raw_path), auxiliary_sha256={}, smoke=False)
    (folder / "config.json").write_text(json.dumps(manifest))
    values = np.array([2100., 3000., 3900.])
    score = study.regression_metrics([2000., 3000., 4000.], values)["rmse"]
    ledger_path = tmp_path / "reports" / "feedback_results.csv"
    pd.DataFrame([dict(run_id=run_id, window="short_recent", method="learner_ridge_market",
                       rmse=score, trees=1600, task_type="GPU")]).to_csv(ledger_path, index=False)
    np.savez(folder / "short_recent_learners.npz", row_id=np.arange(3),
        y=[2000., 3000., 4000.], methods=["learner_ridge_market"], prediction=values[:, None])
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "frozen_baseline", lambda *a: (np.arange(3), values, []))
    if bad_source:
        with pytest.raises(ValueError, match="fingerprint"):
            study.summarize_runs([run_id])
    else:
        study.summarize_runs([run_id])
        saved = pd.read_csv(ledger_path)
        assert saved.trees.item() == 0
        assert saved.actual_device.item() == "CPU"
        assert saved.rmse.item() == pytest.approx(score)
