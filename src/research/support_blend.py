"""Past-only support-group weighting; reuse frozen predictions, never refit CBMs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import normalize_text
from src.research.horizon_blend import (
    ROOT, PLANS, audit_sources, digest, load_window, forward_training,
    fit_gate, predict_gate, metrics,
)

GROUPS = ('sparse_0_4', 'medium_5_19', 'dense_20_plus')
MIN_ROWS = 200


def support_counts(raw, row_ids, cutoff):
    """Count original records through origin, never future records or rent labels.

    These address/type groups are not unique dwelling identities. Missing keys
    receive zero support, rather than aggregating unrelated unknown addresses.
    """
    keys = raw[['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE']].apply(normalize_text)
    valid = keys.notna().all(axis=1) & keys.fillna('').ne('').all(axis=1)
    fit = keys.loc[valid & raw.RENT_APPROVAL_DATE.le(cutoff)]
    counts = fit.groupby(list(keys.columns)).size()
    query = keys.iloc[row_ids]
    values = counts.reindex(pd.MultiIndex.from_frame(query)).fillna(0).to_numpy(dtype=int)
    values[~valid.iloc[row_ids].to_numpy()] = 0
    return values


def support_group(counts):
    return np.digitize(counts, [5, 20])


def fit_support(train, counts):
    group = support_group(counts)
    gates = {}
    for g, name in enumerate(GROUPS):
        mask = group == g
        n = int(mask.sum())
        weight = .5
        if n >= MIN_ROWS:
            subset = {k: train[k][mask] for k in ('h', 'normal', 'stale', 'y')}
            gate = fit_gate(subset, False)
            # Additional sample-size shrinkage; fixed before seeing scores.
            weight += (n / (n + 1000.)) * gate['coefficients'][0]
        gates[name] = dict(rows=n, normal_weight=float(weight), fallback=n < MIN_ROWS)
    return gates


def run_support(sources, root, raw, plans, output):
    windows = sorted({w for p in plans for w in PLANS[p][0] + [PLANS[p][1]]})
    cache = {w: load_window(sources, w, root, raw) for w in windows}
    counts = {w: support_counts(raw, v['row_id'], v['cutoff']) for w, v in cache.items()}
    results, groups, metadata = [], [], {}
    for plan in plans:
        fit_windows, eval_window = PLANS[plan]
        evaluation = cache[eval_window]
        train = forward_training([cache[w] for w in fit_windows], evaluation)
        fit_counts = np.concatenate([counts[w] for w in fit_windows])
        gates = fit_support(train, fit_counts)
        labels = support_group(counts[eval_window])
        weight = np.array([gates[GROUPS[g]]['normal_weight'] for g in labels])
        baseline = (evaluation['normal'] + evaluation['stale']) / 2
        constant = fit_gate(train, False)
        predictions = {
            'equal_50_50': baseline,
            'learned_constant': predict_gate(constant, evaluation['normal'], evaluation['stale'], evaluation['h'])[0],
            'support_group': weight * evaluation['normal'] + (1 - weight) * evaluation['stale'],
        }
        metadata[plan] = dict(fit_windows=fit_windows, evaluation_window=eval_window,
                              cutoff=evaluation['cutoff'], support_gates=gates, constant=constant)
        ref = metrics(evaluation['y'], baseline)
        for method, prediction in predictions.items():
            score = metrics(evaluation['y'], prediction)
            results.append(dict(plan=plan, method=method, rows=len(prediction), **score,
                                rmse_gain_vs_equal=ref['rmse'] - score['rmse']))
            for g, name in enumerate(GROUPS):
                mask = labels == g
                if mask.any():
                    groups.append(dict(plan=plan, method=method, group=name, rows=int(mask.sum()),
                                       **metrics(evaluation['y'][mask], prediction[mask])))
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(results).to_csv(output / 'results.csv', index=False)
    pd.DataFrame(groups).to_csv(output / 'by_support.csv', index=False)
    sources.to_csv(output / 'sources.csv', index=False)
    record = dict(plans=metadata, bins=[5, 20], min_meta_rows=MIN_ROWS, shrink_pseudocount=1000,
                  definition='Origin-frozen count of normalized town/block/street/flat-type records; missing keys get zero.',
                  limitations='Overlapping development windows; group counts are not unique dwelling IDs; no promotion.',
                  code_hashes={str(p.relative_to(ROOT)): digest(p) for p in
                               [Path(__file__), ROOT/'src/research/horizon_blend.py', ROOT/'src/features.py']})
    (output/'metadata.json').write_text(json.dumps(record, indent=2) + '\n')
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/support_blend_v1')
    parser.add_argument('--manifest', type=Path, default=ROOT/'reports/stacking_sources.csv')
    parser.add_argument('--plans', nargs='+', choices=PLANS, default=list(PLANS))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    windows = sorted({w for p in args.plans for w in PLANS[p][0] + [PLANS[p][1]]})
    sources, missing = audit_sources(pd.read_csv(args.manifest), ROOT, windows)
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} frozen prediction caches')
    path = ROOT/'data/raw/train.csv'
    if digest(path) != sources.data_sha256.iloc[0]:
        raise ValueError('Training data hash mismatch')
    print(run_support(sources, ROOT, pd.read_csv(path), args.plans, args.output).to_string(index=False))


if __name__ == '__main__':
    main()
