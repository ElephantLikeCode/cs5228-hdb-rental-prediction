"""Read saved shrinkage predictions; descriptive attribution, no fitting."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from src.features import normalize_text
from src.research.horizon_blend import digest,metrics
from src.research.parent_child_reference import paired
ROOT=Path(__file__).resolve().parents[2]

def verify_bundle(bundle):
    """Verify the claimed feature intervention without changing cached inputs."""
    from src.research.dispersion_shrinkage import shrink_features
    for split in ['train','valid']:
        expected,high=shrink_features(bundle['original_'+split],bundle['stats_'+split])
        pd.testing.assert_frame_equal(expected,bundle['candidate_'+split])
        np.testing.assert_array_equal(high,bundle['high_'+split])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'artifacts/dispersion_shrinkage_cpu_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/dispersion_shrinkage_attribution_v1')
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--reference-results',type=Path,
        help='Optional prior aggregate table; record differences without substituting old predictions')
    parser.add_argument('--verify-models',action='store_true',
        help='Reload every saved CBM and verify its budget and cached predictions')
    args=parser.parse_args(argv)
    if args.bootstrap<1:raise ValueError('bootstrap must be positive')
    src=args.source_dir;out=args.output
    if out.exists():raise FileExistsError(out)
    meta=json.loads((src/'metadata.json').read_text())
    for name,h in meta['hashes'].items():
        if digest(ROOT/name)!=h:raise ValueError(f'Hash mismatch {name}')
    if args.verify_models and ('iterations' not in meta or 'windows' not in meta):
        raise ValueError('Model verification requires training budget/window metadata')
    if args.verify_models and set(meta['windows'])!={'short_recent','horizon16'}:
        raise ValueError('Complete verification requires both validation windows')
    saved=pd.read_csv(src/'results.csv')
    raw=pd.read_csv(ROOT/'data/raw/train.csv');rows=[];profiles=[];checks=[];paired_checks=[];model_checks=[]
    for window,start,end in [('short_recent','2025-01','2025-03'),('horizon16','2023-12','2025-03')]:
        ids=np.flatnonzero(raw.RENT_APPROVAL_DATE.between(start,end));q=raw.iloc[ids].reset_index(drop=True)
        y=q.MONTHLY_RENT.to_numpy(); pred={}
        # Only trusted, locally generated pandas pickle caches may be loaded.
        bundle=pd.read_pickle(src/f'{window}_features.pkl')
        if args.verify_models:verify_bundle(bundle)
        for recipe in ['original','candidate']:
            for mode in ['direct','residual']:
                with np.load(src/f'{window}_{recipe}_{mode}.npz') as z:
                    np.testing.assert_array_equal(ids,z['row_id']);np.testing.assert_array_equal(y,z['y'])
                    pred[recipe,mode]=z['prediction'].copy()
                if pred[recipe,mode].shape!=y.shape or not np.isfinite(pred[recipe,mode]).all():
                    raise ValueError('Malformed or non-finite saved predictions')
                if args.verify_models:
                    from catboost import CatBoostRegressor
                    from src.research.dispersion_shrinkage import fixed_offset
                    path=src/f'{window}_{recipe}_{mode}.cbm'
                    model=CatBoostRegressor();model.load_model(str(path))
                    if model.tree_count_!=meta['iterations']:raise ValueError('Model tree budget mismatch')
                    params=model.get_all_params()
                    if 'seed' in meta and params['random_seed']!=meta['seed']:
                        raise ValueError('Model random seed mismatch')
                    if 'device' in meta and params['task_type']!=meta['device']:
                        raise ValueError('Model training device mismatch')
                    predicted=model.predict(bundle[recipe+'_valid'])+fixed_offset(bundle['original_valid'],mode=='residual')
                    np.testing.assert_allclose(predicted,pred[recipe,mode],rtol=1e-12,atol=1e-9)
                    model_checks.append(dict(window=window,recipe=recipe,mode=mode,
                        actual_trees=model.tree_count_,seed=params['random_seed'],device=params['task_type'],
                        depth=params['depth'],learning_rate=params['learning_rate'],l2_leaf_reg=params['l2_leaf_reg'],
                        feature_count=len(model.feature_names_),sha256=digest(path),reload_predictions_verified=True))
            pred[recipe,'pair']=(pred[recipe,'direct']+pred[recipe,'residual'])/2
        stat=bundle['stats_valid'];high=(stat['count'].ge(5)&stat.adjusted_relative_iqr.gt(.25)).to_numpy()
        np.testing.assert_array_equal(high,bundle['high_valid'])
        q['month']=q.RENT_APPROVAL_DATE;q['flat']=normalize_text(q.FLAT_TYPE)
        q['dispersion']=np.select([high,stat['count'].lt(5),stat.adjusted_relative_iqr.le(.10)],['high','insufficient','low'],default='medium')
        h=pd.PeriodIndex(q.month,freq='M').asi8-pd.Period(start,freq='M').ordinal+1
        q['period']=np.where(h<=8,'first_8','last_8') if window=='horizon16' else 'quarter'
        q['all']='all'
        for month,g in q.groupby('month'):
            profiles.append(dict(window=window,month=month,rows=len(g),mean_rent=g.MONTHLY_RENT.mean(),
                median_rent=g.MONTHLY_RENT.median(),sd_rent=g.MONTHLY_RENT.std(),mean_area=g.FLOOR_AREA_SQM.mean(),
                high_share=g.dispersion.eq('high').mean(),low_rent_share=g.MONTHLY_RENT.le(2500).mean()))
        for mode in ['direct','residual','pair']:
            a=pred['original',mode];b=pred['candidate',mode];d=(b-y)**2-(a-y)**2
            positive_total=float(np.maximum(d,0).sum())
            for dims in [['all'],['month'],['period'],['flat'],['dispersion'],['month','dispersion'],['month','flat'],['period','dispersion']]:
                for key,indices in q.groupby(dims,observed=True).groups.items():
                    if not isinstance(key,tuple):key=(key,)
                    idx=np.asarray(indices);ma=metrics(y[idx],a[idx]);mb=metrics(y[idx],b[idx])
                    rows.append(dict(window=window,mode=mode,dimension='|'.join(dims),group='|'.join(map(str,key)),rows=len(idx),
                        original_rmse=ma['rmse'],candidate_rmse=mb['rmse'],rmse_change=mb['rmse']-ma['rmse'],
                        original_mae=ma['mae'],candidate_mae=mb['mae'],original_bias=ma['bias'],candidate_bias=mb['bias'],
                        delta_sse=float(d[idx].sum()),delta_mse=float(d[idx].mean()),
                        original_sse_share=float(np.sum((a[idx]-y[idx])**2)/np.sum((a-y)**2)),
                        positive_loss_share=float(np.maximum(d[idx],0).sum()/positive_total) if positive_total else 0.,
                        row_mean_rent=float(y[idx].mean())))
            for recipe,p in [('original',a),('candidate',b)]:
                matched=saved[(saved.window==window)&(saved.recipe==recipe)&(saved['mode']==mode)]
                if len(matched)!=1:raise ValueError('Saved results must have one row per window/recipe/mode')
                r=matched.iloc[0]
                for metric,value in metrics(y,p).items():
                    np.testing.assert_allclose(value,r[metric],rtol=1e-10,atol=1e-10)
                if 'rows' in matched and int(r['rows'])!=len(y):raise ValueError('Saved row count mismatch')
        checks.append(dict(window=window,rows=len(q),row_ids_labels_and_scores_verified=True))
        clusters=pd.MultiIndex.from_frame(q[['TOWN','BLOCK','STREET']].apply(normalize_text))
        # paired() calls the first argument "parent": here it is the candidate.
        # Its positive child-minus-parent gain therefore means candidate better.
        pair=paired(pred['candidate','pair'],pred['original','pair'],y,clusters,bootstrap=args.bootstrap)
        expected_gain=metrics(y,pred['original','pair'])['rmse']-metrics(y,pred['candidate','pair'])['rmse']
        np.testing.assert_allclose(pair['rmse_gain_child_minus_parent'],expected_gain,atol=1e-10)
        paired_checks.append(dict(window=window,**pair))
    result=pd.DataFrame(rows)
    for (w,m),g in result.groupby(['window','mode']):
        total=g.loc[g.dimension.eq('all'),'delta_sse'].iloc[0]
        for d in g.dimension.unique():np.testing.assert_allclose(g.loc[g.dimension.eq(d),'delta_sse'].sum(),total,atol=1e-6)
    comparison=None;reference_check=None
    if args.reference_results is not None:
        keys=['window','recipe','mode'];columns=keys+['rows','mse','rmse','mae','bias']
        previous=pd.read_csv(args.reference_results)
        comparison=saved[columns].merge(previous[columns],on=keys,how='outer',
            suffixes=('_rerun','_reference'),validate='one_to_one',indicator=True)
        if not comparison['_merge'].eq('both').all():raise ValueError('Reference comparison keys differ')
        if not comparison.rows_rerun.eq(comparison.rows_reference).all():raise ValueError('Reference row counts differ')
        comparison=comparison.drop(columns='_merge')
        for metric in ['mse','rmse','mae','bias']:
            comparison[metric+'_delta']=comparison[metric+'_rerun']-comparison[metric+'_reference']
        reference_check=dict(sha256=digest(args.reference_results),
            max_absolute_rmse_difference=float(comparison.rmse_delta.abs().max()),
            all_rmse_match=bool(np.allclose(comparison.rmse_rerun,comparison.rmse_reference,rtol=1e-12,atol=1e-9)),
            note='Independent fits, not a claim of identical original model binaries')
    out.mkdir(parents=True)
    saved.to_csv(out/'results.csv',index=False)
    if comparison is not None:comparison.to_csv(out/'reference_comparison.csv',index=False)
    result.to_csv(out/'error_changes.csv',index=False)
    pd.DataFrame(profiles).to_csv(out/'monthly_data_profile.csv',index=False)
    (out/'audit.json').write_text(json.dumps(dict(checks=checks,code_hash=digest(Path(__file__)),
        bootstrap_code_sha256=digest(Path(paired.__code__.co_filename)),
        input_hashes={p.name:digest(p) for p in src.glob('*.npz')},
        source_dir=str(src.resolve()),source_metadata_sha256=digest(src/'metadata.json'),
        source_metadata=meta,
        reference_comparison=reference_check,
        model_checks=model_checks,feature_intervention_verified=args.verify_models,
        sign='candidate minus original: positive means deterioration',
        positive_loss_share='Group positive SSE increments / all positive increments; zero when none exist',
        limitations='One seed; descriptive reused overlapping windows; no causal attribution or parameter fitting'),indent=2)+'\n')
    (out/'local_evaluation_check.json').write_text(json.dumps(dict(checks_passed=True,
        paired_gain_positive_means_candidate_better=True,
        paired_argument_mapping='parent=candidate; child=original; applies to win rate and all gain fields',
        bootstrap=dict(unit='normalized town/block/street',replicates=args.bootstrap,seed=42,
            interval='percentile 95%; exploratory, overlapping windows, not multiplicity adjusted'),
        windows=paired_checks),indent=2)+'\n')
    print(result[(result['mode']=='pair')&result.dimension.isin(['month','period','flat','dispersion'])].to_string(index=False))
if __name__=='__main__':main()
