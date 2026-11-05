"""Contracts for November research; synthetic fixtures are not score evidence.

All generated predictions, data and models live in pytest temporary directories.
The committed aggregate checks verify arithmetic, not the original training run.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research import analyze_shrinkage_errors as attribution
from src.research import dispersion_shrinkage as shrinkage
from src.research import history_reliability as history
from src.research import horizon_blend as horizon
from src.research import parent_child_reference as references
from src.research import support_blend as support


@pytest.fixture
def frozen_sources(tmp_path):
    """Two disjoint synthetic OOF windows, each with all eight components."""
    raw = pd.DataFrame({
        'RENT_APPROVAL_DATE': ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05'],
        'MONTHLY_RENT': [2000., 2100., 2200., 2300., 2400.],
        'TOWN': ['Town'] * 5, 'BLOCK': ['1'] * 5,
        'STREET': ['Test St'] * 5, 'FLAT_TYPE': ['4-room'] * 5,
    })
    rows = []
    for window, ids, cutoff in [('fit', np.array([0, 1]), '2023-12'),
                                ('evaluate', np.array([3, 4]), '2024-03')]:
        for head_index, head in enumerate(horizon.HEADS):
            for seed in (42, 2026):
                path = tmp_path / f'{window}_{head}_{seed}.npz'
                np.savez(path, row_id=ids, month=raw.iloc[ids].RENT_APPROVAL_DATE.to_numpy(dtype=str),
                         y=raw.iloc[ids].MONTHLY_RENT.to_numpy(),
                         prediction=raw.iloc[ids].MONTHLY_RENT.to_numpy() + 10 + head_index * 5)
                rows.append(dict(window=window, candidate=head, seed=seed,
                    run_id=path.stem, data_sha256='synthetic', iterations=1600, actual_trees=1600,
                    task_type='GPU', protocol='fixed_budget_no_outer_early_stopping',
                    prediction_file=path.name, prediction_sha256=horizon.digest(path),
                    train_end=cutoff, validation_start=raw.iloc[ids[0]].RENT_APPROVAL_DATE,
                    validation_end=raw.iloc[ids[-1]].RENT_APPROVAL_DATE))
    return pd.DataFrame(rows), raw, tmp_path


@pytest.mark.parametrize('path', ['../outside.npz', '/outside.npz', r'..\outside.npz'])
def test_cache_paths_cannot_escape_root(tmp_path, path):
    with pytest.raises(ValueError, match='inside'):
        horizon.cache_path(tmp_path, path)
    assert horizon.cache_path(tmp_path, r'cache\example.npz') == tmp_path / 'cache/example.npz'


@pytest.mark.parametrize('damage', ['duplicate', 'missing_head', 'data_hash', 'budget', 'protocol', 'checksum'])
def test_source_audit_rejects_invalid_evidence(frozen_sources, damage):
    sources, _, root = frozen_sources
    if damage == 'duplicate':
        sources = pd.concat([sources, sources.iloc[[0]]], ignore_index=True)
    elif damage == 'missing_head':
        sources = sources.iloc[1:]
    else:
        column, value = {'data_hash': ('data_sha256', 'different'), 'budget': ('actual_trees', 10),
                         'protocol': ('protocol', 'outer_early_stopping'),
                         'checksum': ('prediction_sha256', 'incorrect')}[damage]
        sources.loc[0, column] = value
    with pytest.raises(ValueError):
        horizon.audit_sources(sources, root, ['fit', 'evaluate'])


def test_source_audit_reports_missing_cache_without_creating_it(frozen_sources):
    sources, _, root = frozen_sources
    path = root / sources.iloc[0].prediction_file
    path.unlink()  # Only a pytest-generated synthetic cache.
    _, missing = horizon.audit_sources(sources, root, ['fit', 'evaluate'])
    assert missing == [str(path)]
    assert not path.exists()


@pytest.mark.parametrize('damage', ['order', 'labels', 'months', 'nonfinite', 'cutoff'])
def test_window_loader_rejects_misalignment_or_future_labels(frozen_sources, damage):
    sources, raw, root = frozen_sources
    path = root / sources.loc[sources.window.eq('evaluate')].iloc[0].prediction_file
    with np.load(path, allow_pickle=False) as z:
        item = {k: z[k].copy() for k in z.files}
    if damage == 'order':
        item['row_id'] = item['row_id'][::-1]
    elif damage == 'labels':
        item['y'][0] += 1
    elif damage == 'months':
        item['month'][0] = '2024-03'
    elif damage == 'nonfinite':
        item['prediction'][0] = np.nan
    else:
        sources.loc[sources.window.eq('evaluate'), 'train_end'] = '2024-04'
    np.savez(path, **item)
    with pytest.raises(ValueError):
        horizon.load_window(sources, 'evaluate', root, raw)


@pytest.mark.parametrize('damage', ['duplicate_fit', 'overlap', 'late_labels'])
def test_forward_training_rejects_unavailable_labels(frozen_sources, damage):
    sources, raw, root = frozen_sources
    fit = horizon.load_window(sources, 'fit', root, raw)
    evaluation = horizon.load_window(sources, 'evaluate', root, raw)
    training = [fit]
    if damage == 'duplicate_fit':
        training.append(fit)
    elif damage == 'overlap':
        evaluation['row_id'][0] = fit['row_id'][0]
    else:
        evaluation['cutoff'] = '2024-01'
    with pytest.raises(ValueError):
        horizon.forward_training(training, evaluation)


@pytest.mark.parametrize('dynamic', [False, True])
def test_gate_bounds_clamping_and_equal_branch_identity(dynamic):
    train = dict(h=np.tile([1., 2., 3.], 100), normal=np.full(300, 3000.),
                 stale=np.full(300, 2000.), y=np.full(300, 4000.))
    gate = horizon.fit_gate(train, dynamic)
    prediction, weights = horizon.predict_gate(gate, np.full(4, 3000.),
                                               np.full(4, 2000.), np.array([1., 2., 3., 99.]))
    assert np.all((weights >= .25) & (weights <= .75))
    assert weights[-1] == pytest.approx(weights[-2])
    np.testing.assert_allclose(prediction, weights * 3000 + (1 - weights) * 2000)
    train['stale'] = train['normal'].copy()
    equal = horizon.fit_gate(train, dynamic)
    np.testing.assert_allclose(equal['coefficients'], 0, atol=1e-12)
    with pytest.raises(ValueError, match='horizons'):
        horizon.predict_gate(gate, np.ones(1), np.ones(1), np.array([0.]))


def test_blend_end_to_end_outer_labels_do_not_fit_weights(frozen_sources, monkeypatch):
    sources, raw, root = frozen_sources
    monkeypatch.setitem(horizon.PLANS, 'synthetic', (['fit'], 'evaluate'))
    audited, missing = horizon.audit_sources(sources, root, ['fit', 'evaluate'])
    assert missing == []
    output = root / 'horizon'
    scores = horizon.run(audited, root, raw, ['synthetic'], output)
    plans_before = json.loads((output / 'metadata.json').read_text())['plans']
    # Mutate all outer labels consistently; meta-models must remain unchanged.
    raw.loc[3:, 'MONTHLY_RENT'] += 1000
    for row in sources.loc[sources.window.eq('evaluate')].itertuples():
        path = root / row.prediction_file
        with np.load(path, allow_pickle=False) as z:
            item = {k: z[k].copy() for k in z.files}
        item['y'] += 1000
        np.savez(path, **item)
    second = root / 'horizon_changed_labels'
    changed = horizon.run(sources, root, raw, ['synthetic'], second)
    assert json.loads((second / 'metadata.json').read_text())['plans'] == plans_before
    assert not np.allclose(scores.rmse, changed.rmse)
    with pytest.raises(FileExistsError):
        horizon.run(sources, root, raw, ['synthetic'], output)
    support_scores = support.run_support(sources, root, raw, ['synthetic'], root / 'support')
    assert set(support_scores.method) == {'equal_50_50', 'learned_constant', 'support_group'}
    assert (root / 'support/metadata.json').is_file()


def test_support_is_past_only_normalized_and_retains_distinct_records():
    raw = pd.DataFrame({
        'RENT_APPROVAL_DATE': ['2024-01'] * 5 + ['2024-02'] * 3,
        'TOWN': ['Town', ' town ', 'TOWN', 'Town', None, 'town', 'Town', 'Town'],
        'BLOCK': ['1', '1', '1', '2', '1', '1', None, '2'],
        'STREET': ['Test-St', 'Test St', 'test st', 'Test St', 'Test St', 'TEST ST', 'Test St', 'Test St'],
        'FLAT_TYPE': ['4-room', '4 room', '4 room', '4 room', '4 room', '4 room', '4 room', '4 room'],
    })
    counts = support.support_counts(raw, np.array([5, 6, 7]), '2024-01')
    np.testing.assert_array_equal(counts, [3, 0, 1])
    np.testing.assert_array_equal(support.support_group(np.array([0, 4, 5, 19, 20])), [0, 0, 1, 1, 2])


@pytest.mark.parametrize('n', [199, 200, 1000])
def test_support_gate_fallback_and_fixed_sample_shrinkage(n):
    train = dict(h=np.ones(n), normal=np.full(n, 3000.), stale=np.full(n, 2000.), y=np.full(n, 4000.))
    gates = support.fit_support(train, np.zeros(n, dtype=int))
    first = gates[support.GROUPS[0]]
    assert first['fallback'] == (n < 200)
    expected = .5 if n < 200 else .5 + n / (n + 1000) * horizon.fit_gate(train, False)['coefficients'][0]
    assert first['normal_weight'] == pytest.approx(expected)
    assert gates[support.GROUPS[2]] == dict(rows=0, normal_weight=.5, fallback=True)


@pytest.fixture
def historical_rows():
    return pd.DataFrame({
        'RENT_APPROVAL_DATE': ['2024-01'] * 6 + ['2024-02', '2025-01'],
        'MONTHLY_RENT': [1000., 2000., 3000., 4000., 5000., 3000., 3000., 999999.],
        'TOWN': ['Town'] * 8, 'BLOCK': ['1'] * 5 + ['2', '1', '1'],
        'STREET': ['Test St'] * 8, 'FLAT_TYPE': ['4-room'] * 8,
        'FLAT_MODEL': ['Model A'] * 8, 'FLOOR_AREA_SQM': [90.] * 8,
        'LEASE_COMMENCE_DATE': [1990] * 8,
    })


def test_reliability_exact_statistics_and_query_target_invariance(historical_rows):
    ref = historical_rows
    query = pd.concat([ref.iloc[[0]]] * 3, ignore_index=True)
    query['RENT_APPROVAL_DATE'] = ['2024-03', '2024-04', '2024-03']
    query.loc[2, 'BLOCK'] = None
    before = ref.copy(deep=True)
    stats = history.reliability(ref, query, '2024-02')
    query['MONTHLY_RENT'] = -999999.
    pd.testing.assert_frame_equal(stats, history.reliability(ref, query, '2024-02'))
    pd.testing.assert_frame_equal(before, ref)
    weight = np.exp2(-1 / 12)
    assert stats['count'].tolist() == [6, 6, 0]
    assert stats.mass.iloc[0] == pytest.approx(5 * weight + 1)
    assert stats.kish_ess.iloc[0] == pytest.approx((5 * weight + 1) ** 2 / (5 * weight ** 2 + 1))
    assert stats.age.iloc[:2].tolist() == [1, 2]
    rents = np.array([1000., 2000., 3000., 4000., 5000., 3000.])
    expected = (np.quantile(rents, .75) - np.quantile(rents, .25)) / np.median(rents)
    assert stats.adjusted_relative_iqr.iloc[0] == pytest.approx(expected)
    assert stats.dispersion.iloc[0] == 'high_gt_0.25'
    assert stats.age_group.iloc[2] == 'no_history'
    with pytest.raises(ValueError, match='precede'):
        history.reliability(ref, ref.iloc[[0]], '2024-02')


def test_group_scores_conserve_rows_sse_and_share():
    frame = pd.DataFrame({'slice': ['a', 'a', 'b'], 'N': [1., -2., 3.],
        'S': [3., -2., 1.], 'E': [2., -2., 2.],
        'count': [5, 6, 7], 'mass': [4., 5., 6.], 'age': [1., 2., 3.]})
    scores = pd.DataFrame(history.group_scores(frame, ['slice'], 'synthetic'))
    for branch, group in scores.groupby('branch'):
        assert group.rows.sum() == len(frame)
        assert group.row_share.sum() == pytest.approx(1.)
        assert group.sse.sum() == pytest.approx((frame[branch] ** 2).sum())
        assert group.sse_share.sum() == pytest.approx(1.)
        np.testing.assert_allclose(group.rmse ** 2 * group.rows, group.sse)


def test_stratified_contrast_uses_common_cells_and_pooled_weights():
    frame = pd.DataFrame({'stratum': ['a'] * 40 + ['b'] * 60 + ['excluded'],
        'risk': ['low'] * 20 + ['high'] * 20 + ['low'] * 20 + ['high'] * 40 + ['low'],
        'E': [1.] * 20 + [3.] * 20 + [2.] * 20 + [4.] * 40 + [1000.]})
    result = history.stratified_contrast(frame, 'risk', 'low', 'high', ['stratum'], 'synthetic')
    assert result['cells'] == 2
    assert result['standardized_low_rmse'] == pytest.approx(np.sqrt(.4 * 1 + .6 * 4))
    assert result['standardized_high_rmse'] == pytest.approx(np.sqrt(.4 * 9 + .6 * 16))
    assert result['low_coverage'] == pytest.approx(40 / 41)
    assert history.stratified_contrast(frame.iloc[:10], 'risk', 'low', 'high', ['stratum'], 'tiny')['cells'] == 0


@pytest.mark.parametrize('damage', [None, 'ledger', 'config', 'cutoff'])
def test_reliability_verifies_source_ledger_and_branch_recipes(frozen_sources, monkeypatch, damage):
    sources, raw, root = frozen_sources
    data_dir = root / 'data/raw'
    data_dir.mkdir(parents=True)
    raw.to_csv(data_dir / 'train.csv', index=False)
    sources = sources.loc[sources.window.eq('evaluate')].copy().reset_index(drop=True)
    sources['data_sha256'] = horizon.digest(data_dir / 'train.csv')
    sources['features'] = 'synthetic'
    sources['feature_count'] = 2
    sources['code_sha256'] = 'synthetic'
    sources['train_rows'] = 3
    configs = []
    for row in sources.itertuples():
        normal = row.candidate.startswith('local_trend')
        configs.append(json.dumps(dict(history='local_trend' if normal else 'hierarchical',
            augmentation_lag=0 if normal else 6, depth=8, iterations=1600, learning_rate=.04,
            l2_leaf_reg=6., residual_target=row.candidate.endswith('residual'), outlier_policy='keep')))
    sources['config_json'] = configs
    ledger = sources.copy(deep=True)
    monkeypatch.setattr(history, 'ROOT', root)
    if damage == 'ledger':
        ledger.loc[0, 'feature_count'] = 3
    elif damage == 'config':
        cfg = json.loads(configs[0])
        cfg['history'] = 'incorrect_recipe'
        sources.loc[0, 'config_json'] = ledger.loc[0, 'config_json'] = json.dumps(cfg)
    elif damage == 'cutoff':
        sources['train_rows'] = 99
    if damage is None:
        assert len(history.verify_sources(sources, ledger, raw)) == 8
    else:
        with pytest.raises(ValueError):
            history.verify_sources(sources, ledger, raw)


def test_training_dispersion_excludes_whole_query_month_and_future(historical_rows):
    ref = historical_rows.drop(columns='MONTHLY_RENT')
    y = historical_rows.MONTHLY_RENT.copy()
    query = ref.iloc[[0, 6]].copy()
    stats = shrinkage.past_dispersion(ref, y, query)
    assert stats['count'].tolist() == [0, 5]
    y.loc[historical_rows.RENT_APPROVAL_DATE.ge('2024-02')] = 1.e9
    pd.testing.assert_frame_equal(stats, shrinkage.past_dispersion(ref, y, query))


def test_feature_builder_does_not_use_own_month_targets(historical_rows, monkeypatch):
    # Isolate the spatial join; real production historical feature builders run.
    monkeypatch.setattr(shrinkage, 'add_spatial_features', lambda frame, lookup, mode: frame)
    ref = historical_rows.drop(columns='MONTHLY_RENT')
    query = ref.iloc[[6]]
    y = historical_rows.MONTHLY_RENT.copy()
    first = shrinkage.build_features(ref, y, query, None, 'local_trend')
    y.loc[historical_rows.RENT_APPROVAL_DATE.ge('2024-02')] = 1.e9
    second = shrinkage.build_features(ref, y, query, None, 'local_trend')
    pd.testing.assert_frame_equal(first, second)


def test_shrinkage_only_changes_two_columns_above_both_strict_thresholds():
    original = pd.DataFrame({shrinkage.PREMIUM: [1.4] * 4, shrinkage.LEVEL: [4200.] * 4,
        'ADAPT_ADDRESS_FLAT_PREMIUM': [1.] * 4, 'ADAPT_MARKET_LEVEL': [3000.] * 4,
        'ADAPT_ADDRESS_FLAT_AREA_LOG_MASS': np.log1p([10., 10., 10., 10.]),
        'other_feature': [7.] * 4}, index=[10, 20, 30, 40])
    stats = pd.DataFrame({'count': [5, 4, 5, 6], 'adjusted_relative_iqr': [.3, .9, .25, np.nan]})
    before = original.copy(deep=True)
    changed, high = shrinkage.shrink_features(original, stats)
    np.testing.assert_array_equal(high, [True, False, False, False])
    expected = (15 * 1.4 + 5 * 1.) / 20
    assert changed.loc[10, shrinkage.PREMIUM] == pytest.approx(expected)
    assert changed.loc[10, shrinkage.LEVEL] == pytest.approx(3000 * expected)
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(changed.loc[[20, 30, 40]], original.loc[[20, 30, 40]])
    pd.testing.assert_frame_equal(changed.drop(columns=[shrinkage.LEVEL, shrinkage.PREMIUM]),
                                  original.drop(columns=[shrinkage.LEVEL, shrinkage.PREMIUM]))
    with pytest.raises(ValueError, match='row mismatch'):
        shrinkage.shrink_features(original, stats.iloc[:1])


@pytest.mark.parametrize('residual', [False, True])
def test_fit_head_retains_original_offset_and_training_budget(monkeypatch, residual):
    original = pd.DataFrame({shrinkage.LEVEL: [2000., 2100.]})
    candidate = pd.DataFrame({shrinkage.LEVEL: [2500., 2600.]})
    target = np.array([2400., 2500.])
    captured = {}

    def fake_fit(X, y, config, seed, device, verbose):
        captured.update(X=X, y=y, config=config, seed=seed, device=device)
        return 'synthetic-model'

    monkeypatch.setattr(shrinkage, 'fit_model', fake_fit)
    assert shrinkage.fit_head(candidate, target, original, residual, 13, 'CPU', 42) == 'synthetic-model'
    offset = np.array([2000., 2100.]) if residual else np.zeros(2)
    np.testing.assert_array_equal(captured['y'], target - offset)
    assert captured['X'] is candidate
    assert captured['config']['iterations'] == 13
    assert captured['config']['residual_target'] is False
    assert captured['device'] == 'CPU' and captured['seed'] == 42
    isolated = shrinkage.fixed_offset(original, residual)
    isolated[0] = -1
    assert original.iloc[0, 0] == 2000


def test_shrinkage_real_cpu_fit_smoke():
    X = pd.DataFrame({'area': np.arange(24, dtype=float), 'place': ['a', 'b'] * 12})
    config = dict(iterations=3, learning_rate=.04, depth=2, l2_leaf_reg=6.)
    model = shrinkage.fit_model(X, 2000 + X.area.to_numpy(), config, 42, 'CPU', False)
    prediction = model.predict(X)
    assert model.tree_count_ == 3
    assert prediction.shape == (24,) and np.isfinite(prediction).all()


@pytest.mark.parametrize('better', ['parent', 'child', 'tie'])
def test_cluster_bootstrap_gain_sign_determinism_and_nonmultiple_batch(better):
    y = np.zeros(6)
    parent = np.zeros(6) if better != 'child' else np.full(6, 2.)
    child = np.full(6, 2.) if better == 'parent' else np.zeros(6)
    clusters = pd.MultiIndex.from_tuples([('town', block) for block in ['1', '1', '2', '2', '2', '3']])
    result = references.paired(parent, child, y, clusters, bootstrap=113)
    assert result == references.paired(parent, child, y, clusters, bootstrap=113)
    assert result['clusters'] == 3 and result['rows'] == 6
    expected = {'parent': 2., 'child': -2., 'tie': 0.}[better]
    assert result['rmse_gain_child_minus_parent'] == pytest.approx(expected)
    assert result['rmse_gain_ci_low'] == pytest.approx(expected)
    assert result['rmse_gain_ci_high'] == pytest.approx(expected)
    assert result['conclusion'] == {'parent': 'parent_better', 'child': 'child_better', 'tie': 'uncertain'}[better]


def test_cluster_bootstrap_resamples_whole_addresses_not_individual_rows():
    parent = np.array([1., 1., 1., 4., 2., 2.])
    child = np.array([3., 3., 3., 1., 4., 4.])
    labels = np.array(['a', 'a', 'a', 'b', 'c', 'c'])
    result = references.paired(parent, child, np.zeros(6), labels, bootstrap=113)
    # Independently compute the distribution with full address blocks retained.
    draw = np.random.default_rng(42).integers(3, size=(113, 3))
    gains = []
    for sample in draw:
        indices = np.concatenate([np.flatnonzero(labels == ['a', 'b', 'c'][c]) for c in sample])
        gains.append(np.sqrt(np.mean(child[indices] ** 2)) - np.sqrt(np.mean(parent[indices] ** 2)))
    low, high = np.quantile(gains, [.025, .975])
    assert result['rmse_gain_ci_low'] == pytest.approx(low)
    assert result['rmse_gain_ci_high'] == pytest.approx(high)


@pytest.fixture
def attribution_inputs(tmp_path, monkeypatch):
    raw_dir = tmp_path / 'data/raw'
    raw_dir.mkdir(parents=True)
    raw = pd.DataFrame({'RENT_APPROVAL_DATE': ['2023-12', '2024-08', '2025-01', '2025-03'],
        'MONTHLY_RENT': [2000., 2100., 2200., 2300.], 'FLAT_TYPE': ['4-room', '5-room'] * 2,
        'FLOOR_AREA_SQM': [90., 110.] * 2})
    raw.to_csv(raw_dir / 'train.csv', index=False)
    src = tmp_path / 'artifacts/dispersion_shrinkage_cpu_v1'
    src.mkdir(parents=True)
    (src / 'metadata.json').write_text(json.dumps({'hashes': {'data/raw/train.csv': horizon.digest(raw_dir / 'train.csv')}}))
    rows = []
    for window, start, end in [('short_recent', '2025-01', '2025-03'), ('horizon16', '2023-12', '2025-03')]:
        ids = np.flatnonzero(raw.RENT_APPROVAL_DATE.between(start, end))
        y = raw.iloc[ids].MONTHLY_RENT.to_numpy()
        for recipe, error in [('original', 10.), ('candidate', 12.)]:
            predictions = []
            for mode, shift in [('direct', 0.), ('residual', 2.)]:
                p = y + error + shift
                np.savez(src / f'{window}_{recipe}_{mode}.npz', row_id=ids, y=y, prediction=p)
                predictions.append(p)
                rows.append(dict(window=window, recipe=recipe, mode=mode, **horizon.metrics(y, p)))
            rows.append(dict(window=window, recipe=recipe, mode='pair', **horizon.metrics(y, np.mean(predictions, axis=0))))
        stats = pd.DataFrame({'count': [6] * len(ids), 'adjusted_relative_iqr': [.3] * len(ids)})
        pd.to_pickle({'stats_valid': stats, 'high_valid': np.ones(len(ids), dtype=bool)}, src / f'{window}_features.pkl')
    pd.DataFrame(rows).to_csv(src / 'results.csv', index=False)
    monkeypatch.setattr(attribution, 'ROOT', tmp_path)
    return tmp_path, src


def test_attribution_end_to_end_conserves_sse_and_refuses_overwrite(attribution_inputs):
    root, _ = attribution_inputs
    attribution.main()
    out = root / 'artifacts/dispersion_shrinkage_attribution_v1'
    result = pd.read_csv(out / 'error_changes.csv')
    for (_, _), frame in result.groupby(['window', 'mode']):
        total = frame.loc[frame.dimension.eq('all'), 'delta_sse'].iloc[0]
        assert total > 0
        for _, group in frame.groupby('dimension'):
            assert group.delta_sse.sum() == pytest.approx(total)
            assert group.original_sse_share.sum() == pytest.approx(1.)
    assert all(c['row_ids_labels_and_scores_verified'] for c in json.loads((out / 'audit.json').read_text())['checks'])
    with pytest.raises(FileExistsError):
        attribution.main()


@pytest.mark.parametrize('damage', ['data_hash', 'row_id', 'labels', 'saved_score'])
def test_attribution_rejects_inconsistent_inputs_before_output(attribution_inputs, damage):
    root, src = attribution_inputs
    if damage == 'data_hash':
        (src / 'metadata.json').write_text(json.dumps({'hashes': {'data/raw/train.csv': 'incorrect'}}))
    elif damage == 'saved_score':
        table = pd.read_csv(src / 'results.csv')
        table.loc[0, 'rmse'] += 1
        table.to_csv(src / 'results.csv', index=False)
    else:
        path = src / 'short_recent_original_direct.npz'
        with np.load(path, allow_pickle=False) as z:
            item = {k: z[k].copy() for k in z.files}
        item['row_id' if damage == 'row_id' else 'y'][0] += 1
        np.savez(path, **item)
    with pytest.raises((ValueError, AssertionError)):
        attribution.main()
    assert not (root / 'artifacts/dispersion_shrinkage_attribution_v1').exists()


def test_committed_aggregate_arithmetic_and_gain_direction():
    root = Path(__file__).resolve().parents[1] / 'reports/history_research'
    results = pd.read_csv(root / 'dispersion_shrinkage_cpu_v1/results.csv')
    np.testing.assert_allclose(results.rmse ** 2, results.mse, rtol=1e-12)
    checks = json.loads((root / 'dispersion_shrinkage_cpu_v1/local_evaluation_check.json').read_text())
    for row in checks['windows']:
        pair = results.loc[results.window.eq(row['window']) & results['mode'].eq('pair')].set_index('recipe')
        gain = pair.loc['original', 'rmse'] - pair.loc['candidate', 'rmse']
        assert row['rmse_gain_child_minus_parent'] == pytest.approx(gain)
    changes = pd.read_csv(root / 'dispersion_shrinkage_attribution_v1/error_changes.csv')
    np.testing.assert_allclose(changes.candidate_rmse - changes.original_rmse, changes.rmse_change, atol=1e-10)
    for _, frame in changes.groupby(['window', 'mode']):
        total = frame.loc[frame.dimension.eq('all'), 'delta_sse'].iloc[0]
        for _, group in frame.groupby('dimension'):
            np.testing.assert_allclose(group.delta_sse.sum(), total, rtol=1e-10, atol=1e-5)
    for study in ['horizon_blend_v1', 'support_blend_v1']:
        table = pd.read_csv(root / study / 'results.csv')
        np.testing.assert_allclose(table.rmse ** 2, table.mse, rtol=1e-12)
        for _, frame in table.groupby('plan'):
            baseline = frame.loc[frame.method.eq('equal_50_50'), 'rmse'].iloc[0]
            np.testing.assert_allclose(baseline - frame.rmse, frame.rmse_gain_vs_equal, atol=1e-10)
    groups = pd.read_csv(root / 'history_reliability_diagnostic_v1/group_metrics.csv')
    np.testing.assert_allclose(groups.mse * groups.rows, groups.sse, rtol=1e-12)
    for _, group in groups.groupby(['window', 'dimension', 'branch']):
        assert group.row_share.sum() == pytest.approx(1.)
        assert group.sse_share.sum() == pytest.approx(1.)
    contrasts = pd.read_csv(root / 'history_reliability_diagnostic_v1/stratified_contrasts.csv')
    contrasts = contrasts.loc[contrasts.cells.gt(0)]
    np.testing.assert_allclose(contrasts.standardized_high_rmse ** 2 - contrasts.standardized_low_rmse ** 2,
                              contrasts.standardized_mse_high_minus_low, rtol=1e-10, atol=1e-7)
    pairs = pd.read_csv(root / 'parent_child_reference_v1/paired_comparisons.csv')
    assert pairs.clusters.le(pairs.rows).all()
    assert pairs.rmse_gain_ci_low.le(pairs.rmse_gain_ci_high).all()
    expected = np.where(pairs.rmse_gain_ci_low.gt(0), 'parent_better',
                        np.where(pairs.rmse_gain_ci_high.lt(0), 'child_better', 'uncertain'))
    np.testing.assert_array_equal(pairs.conclusion, expected)
