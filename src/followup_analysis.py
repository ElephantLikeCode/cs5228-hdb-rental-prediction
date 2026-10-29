"""Post-feedback transfer diagnostics and bounded interchange-access experiment.

python src/followup_analysis.py [--train-transit]
Frozen label predictions stay local; only aggregate evidence is shared.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
try:
    from .feedback_experiments import frozen_baseline
    from .features import normalize_text, build_block_lookup, _haversine_matrix
    from .modeling import build_features, CANDIDATES, fit_model, predict_model
    from .validation import regression_metrics
    from .market_experiments import station_calendar, paired_gain_interval
except ImportError:
    from feedback_experiments import frozen_baseline
    from features import normalize_text, build_block_lookup, _haversine_matrix
    from modeling import build_features, CANDIDATES, fit_model, predict_model
    from validation import regression_metrics
    from market_experiments import station_calendar, paired_gain_interval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/followup'


def grouped_risk(fit, events, query):
    """One fixed observable partition; training OOF labels only, shrinkage 30."""
    def keys(x):
        return normalize_text(x.TOWN).astype(str)+'|'+normalize_text(x.FLAT_TYPE).astype(str)
    stats = pd.DataFrame({'key':keys(fit).to_numpy(),'event':events}).groupby('key').event.agg(['sum','count'])
    rates = (stats['sum']+30*np.mean(events))/(stats['count']+30)
    return keys(query).map(rates).fillna(np.mean(events)).to_numpy()


def interchange_features(raw, core, calendar, origin=None):
    """Observable station-line diversity; no invented track edges/travel times.

    Supplied NAME identifies physical interchanges. Announcements must predate
    the evaluation origin and service must begin by query month. Legacy dates
    are lower-bound availability flags inherited from the audited calendar.
    """
    months=pd.to_datetime(raw.RENT_APPROVAL_DATE).reset_index(drop=True)
    result=pd.DataFrame(index=np.arange(len(raw)),columns=['ACCESS_INTERCHANGE_KM','ACCESS_LINES_1KM','ACCESS_LINES_2KM'],dtype=float)
    known=calendar if origin is None else calendar.loc[calendar.source.eq('operational_before_study')|calendar.announced.lt(pd.Timestamp(origin))]
    for month in sorted(months.unique()):
        ids=np.flatnonzero(months.eq(month)); active=known.loc[known.effective.le(month)].copy()
        active['line']=active.CODE.str.extract(r'^([a-z]+)',expand=False)
        groups=active.groupby('NAME').agg(LATITUDE=('LATITUDE','mean'),LONGITUDE=('LONGITUDE','mean'),lines=('line','nunique'))
        inter=groups.loc[groups.lines.ge(2)]
        coords=core.iloc[ids]
        distance,_,_=_haversine_matrix(coords.LATITUDE.to_numpy(),coords.LONGITUDE.to_numpy(),inter.LATITUDE.to_numpy(),inter.LONGITUDE.to_numpy())
        result.loc[ids,'ACCESS_INTERCHANGE_KM']=distance
        # Count distinct line prefixes accessible inside the radius, not platforms.
        for radius in [1,2]:
            reachable=[]
            for _, stations in active.groupby('line'):
                d,_,_=_haversine_matrix(coords.LATITUDE.to_numpy(),coords.LONGITUDE.to_numpy(),stations.LATITUDE.to_numpy(),stations.LONGITUDE.to_numpy())
                reachable.append(d<=radius)
            result.loc[ids,f'ACCESS_LINES_{radius}KM']=np.sum(reachable,axis=0)
    assert np.isfinite(result.to_numpy()).all()
    return result


def analyze(raw):
    rows=[]; diagnostics=[]; risk=[]
    ledger=pd.read_csv(ROOT/'reports/temporal_comparison.csv')
    for window,previous in [('long_older','bridge_2022'),('short_recent','short_middle')]:
        ids,pred,sources,components=frozen_baseline(window,raw,raw.MONTHLY_RENT,True)
        y=raw.MONTHLY_RENT.iloc[ids].to_numpy(); q=raw.iloc[ids].copy()
        refs=[]
        for candidate in ['hierarchical','market_residual']:
            for seed in [42,2026]:
                r=ledger.loc[ledger.window.eq(window)&ledger.candidate.eq(candidate)&ledger.seed.eq(seed)&ledger.iterations.eq(1600)&ledger.task_type.eq('GPU')].iloc[-1]
                with np.load(ROOT/r.prediction_file) as f:
                    assert np.array_equal(ids,f['row_id']); refs.append(f['prediction'].copy())
        reference=np.mean(refs,axis=0)
        variants={'adaptive_reference':reference,'selected':pred,
            'normal_only':np.mean([p for (name,s),p in components.items() if name.startswith('local_trend')],axis=0),
            'stale_only':np.mean([p for (name,s),p in components.items() if name.startswith('horizon')],axis=0),
            'direct_only':np.mean([p for (name,s),p in components.items() if name.endswith('_direct')],axis=0),
            'residual_only':np.mean([p for (name,s),p in components.items() if name.endswith('_residual')],axis=0)}
        oldids,oldpred,_=frozen_baseline(previous,raw,raw.MONTHLY_RENT)
        assert raw.iloc[oldids].RENT_APPROVAL_DATE.max()<q.RENT_APPROVAL_DATE.min()
        correction=float(np.mean(oldpred-raw.MONTHLY_RENT.iloc[oldids]))
        variants['prior_origin_half_bias']=pred-.5*correction
        for name,p in variants.items():
            rows.append(dict(window=window,method=name,**regression_metrics(y,p),prior_bias=correction,rows=len(ids)))
        q['delta_sse']=(pred-y)**2-(reference-y)**2
        q['selected_error']=pred-y; q['reference_error']=reference-y
        q['flat']=normalize_text(q.FLAT_TYPE)
        for field in ['RENT_APPROVAL_DATE','flat','TOWN']:
            for value,g in q.groupby(field):
                diagnostics.append(dict(window=window,field=field,value=value,rows=len(g),
                    delta_sse=float(g.delta_sse.sum()),selected_bias=float(g.selected_error.mean()),
                    selected_rmse=float(np.sqrt(np.mean(g.selected_error**2))),reference_rmse=float(np.sqrt(np.mean(g.reference_error**2)))))
        train_events=raw.MONTHLY_RENT.iloc[oldids].to_numpy()<.65*oldpred
        events=y<.65*pred
        probability=grouped_risk(raw.iloc[oldids],train_events,q)
        risk.append(dict(window=window,training_window=previous,training_rows=len(oldids),event_rate=float(events.mean()),
            auc=float(roc_auc_score(events,probability)),ap=float(average_precision_score(events,probability)),
            event_definition='observed rent < 0.65 * frozen selected prediction',partition='town/type; smoothing 30; no validation fit'))
    pd.DataFrame(rows).to_csv(ROOT/'reports/followup_transfer.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(ROOT/'reports/followup_groups.csv',index=False)
    pd.DataFrame(risk).to_csv(ROOT/'reports/followup_low_rent.csv',index=False)


def train_transit(raw, prepare_only=False):
    lookup=build_block_lookup(ROOT/'data/raw/auxiliary')
    calendar=station_calendar(pd.read_csv(ROOT/'data/raw/auxiliary/sg-mrt-stations.csv'),json.loads((ROOT/'configs/mrt_openings.json').read_text()))
    results=[]
    for window in ['short_recent','long_older','horizon16']:
        ids,baseline,sources,components=frozen_baseline(window,raw,raw.MONTHLY_RENT,True)
        valid=raw.iloc[ids].drop(columns='MONTHLY_RENT').reset_index(drop=True); origin=valid.RENT_APPROVAL_DATE.min()
        subset=raw.loc[raw.RENT_APPROVAL_DATE.lt(origin)].reset_index(drop=True)
        y=subset.MONTHLY_RENT.copy(); train=subset.drop(columns='MONTHLY_RENT')
        cache=OUT/f'{window}_transit_features_label_free.pkl'
        signature=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'data/raw/train.csv').read_bytes()+(ROOT/'configs/mrt_openings.json').read_bytes()).hexdigest()
        if cache.exists():
            saved=pd.read_pickle(cache)
            assert saved['signature']==signature, 'Delete stale local feature cache before recomputing'
            a,b=saved['train'],saved['valid']
        else:
            a=build_features(train,y,train,lookup,'local_trend'); b=build_features(train,y,valid,lookup,'local_trend')
            extra_a=interchange_features(train,a,calendar); extra_b=interchange_features(valid,b,calendar,origin)
            a=a.join(extra_a); b=b.join(extra_b)
            pd.to_pickle({'signature':signature,'train':a,'valid':b},cache)
        assert 'MONTHLY_RENT' not in a and 'MONTHLY_RENT' not in b
        if prepare_only:
            print(f'PREPARED {window}: {a.shape}, {b.shape}',flush=True)
            continue
        predictions=[]; start=time.perf_counter()
        for mode in ['direct','residual']:
            cfg=dict(CANDIDATES[f'local_trend_{mode}'])
            print(f'TRANSIT {window} {mode} trees=1600',flush=True)
            model=fit_model(a,y,cfg,42,'GPU'); predictions.append(predict_model(model,b,cfg))
        pair=np.mean(predictions,axis=0)
        replacement=baseline+(predictions[0]-components[('local_trend_direct',42)]+predictions[1]-components[('local_trend_residual',42)])/8
        for name,p in [('replace_seed',replacement),('blend25',.75*baseline+.25*pair)]:
            actual=raw.MONTHLY_RENT.iloc[ids].to_numpy()
            metrics=regression_metrics(actual,p)
            lo,hi=paired_gain_interval(raw,ids,actual,baseline,p)
            results.append(dict(window=window,method='interchange_'+name,**metrics,gain=regression_metrics(actual,baseline)['rmse']-metrics['rmse'],ci_lower=lo,ci_upper=hi,seed=42,trees=1600,train_end=train.RENT_APPROVAL_DATE.max(),rows=len(ids),model_seconds=time.perf_counter()-start,source_runs=json.dumps(sources),feature_group='origin-known interchange distance and distinct lines within 1/2 km',data_sha256=hashlib.sha256((ROOT/'data/raw/train.csv').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol='fixed_budget_origin_known_service_no_scoring_labels'))
            np.savez_compressed(OUT/f'{window}_{name}.npz',row_id=ids,y=actual,prediction=p)
        pd.DataFrame(results).to_csv(ROOT/'reports/followup_transit.csv',index=False)


def summarize_reproduction(raw):
    """Audit the isolated full-budget run without promoting its artifacts."""
    clone=ROOT/'tmp/submission_acceptance'
    metadata=json.loads((clone/'reports/acceptance_full_metadata.json').read_text())
    assert len(metadata['components'])==8 and metadata['iterations']==1600
    for component in metadata['components']:
        assert component['actual_trees']==1600
        assert hashlib.sha256((clone/component['model_path']).read_bytes()).hexdigest()==component['sha256']
    old=pd.read_csv(ROOT/'submissions/final_submission.csv')
    new=pd.read_csv(clone/'submissions/acceptance_full.csv')
    assert len(new)==50000 and old.Id.equals(new.Id) and np.isfinite(new.Predicted).all()
    assert hashlib.sha256((clone/'submissions/acceptance_full.csv').read_bytes()).hexdigest()==metadata['submission_sha256']
    ledger=pd.read_csv(clone/'reports/temporal_comparison.csv')
    outputs=[]; ids=None
    for candidate in ['local_trend_direct','local_trend_residual','horizon_direct','horizon_residual']:
        for seed in [42,2026]:
            row=ledger.loc[ledger.candidate.eq(candidate)&ledger.seed.eq(seed)&ledger.window.eq('short_recent')&ledger.iterations.eq(1600)].iloc[-1]
            assert row.actual_trees==1600
            with np.load(clone/row.prediction_file) as f:
                if ids is None: ids=f['row_id'].copy()
                assert np.array_equal(ids,f['row_id']); outputs.append(f['prediction'].copy())
    fresh=np.mean(outputs,axis=0)
    oldids,oldpred,_=frozen_baseline('short_recent',raw,raw.MONTHLY_RENT)
    assert np.array_equal(oldids,ids)
    result={'date':'2026-10-29','device':'RTX 4060 GPU','scope':'Independent local checkout and new venv, same physical device; full selected fit and recent validation, not every historical experiment.',
        'components':8,'trees_per_component':1600,'full_training_rows':metadata['raw_train_rows'],'test_rows':len(new),
        'fit_seconds':metadata['elapsed_seconds'],'submission_sha256':metadata['submission_sha256'],
        'production_submission_sha256':hashlib.sha256((ROOT/'submissions/final_submission.csv').read_bytes()).hexdigest(),
        'submission_mean_absolute_prediction_change':float(np.abs(new.Predicted-old.Predicted).mean()),
        'submission_max_absolute_prediction_change':float(np.abs(new.Predicted-old.Predicted).max()),
        'fresh_recent_metrics':regression_metrics(raw.MONTHLY_RENT.iloc[ids],fresh),
        'frozen_recent_metrics':regression_metrics(raw.MONTHLY_RENT.iloc[ids],oldpred),
        'recent_mean_absolute_prediction_change':float(np.abs(fresh-oldpred).mean()),
        'promoted':False}
    (ROOT/'reports/full_budget_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--train-transit',action='store_true'); parser.add_argument('--prepare-transit',action='store_true'); parser.add_argument('--summarize-reproduction',action='store_true'); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(ROOT/'data/raw/train.csv')
    analyze(raw)
    if args.train_transit or args.prepare_transit: train_transit(raw,args.prepare_transit)
    if args.summarize_reproduction: summarize_reproduction(raw)
    print('Follow-up evidence saved under reports/followup_*.csv',flush=True)
