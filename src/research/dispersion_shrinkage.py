"""Paired normal-history experiment: fixed dispersion-dependent finest-level shrinkage.

Production features/configuration are never edited. No frozen GPU predictions are
mixed into this same-device, seed-42, two-head comparison. No submission output.
"""
from pathlib import Path
import argparse
import json
import time
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.features import (build_block_lookup, make_core_features, add_spatial_features,
                          add_past_rent_features, add_adaptive_rent_features, prepare_for_catboost)
from src.research.history_reliability import reliability
from src.research.horizon_blend import digest, metrics

ROOT=Path(__file__).resolve().parents[2]
WINDOWS={'short_recent':('2025-01','2025-03'),'horizon16':('2023-12','2025-03')}
LEVEL='ADAPT_ADDRESS_FLAT_AREA_LEVEL'
PREMIUM='ADAPT_ADDRESS_FLAT_AREA_PREMIUM'


def build_features(reference,target,query,lookup,recipe):
    """Production local_trend recipe via unchanged feature functions, no XGBoost import."""
    if recipe != 'local_trend': raise ValueError('Only normal-history recipe is studied')
    reference=reference.reset_index(drop=True);query=query.reset_index(drop=True)
    result=add_spatial_features(make_core_features(query),lookup,'all_spatial')
    history=add_past_rent_features(reference,pd.Series(target),query,detailed=True,
                                   market_features=False,include_recent=False)
    cols=[c for c in history if c.startswith('HIST_')]
    result[cols]=history[cols]
    adaptive=add_adaptive_rent_features(reference,target,query,include_local_trend=True)
    result[adaptive.columns]=adaptive
    return prepare_for_catboost(result)


def fit_model(X,y,config,seed,device,verbose):
    model=CatBoostRegressor(loss_function='RMSE',iterations=config['iterations'],
        learning_rate=config['learning_rate'],depth=config['depth'],l2_leaf_reg=config['l2_leaf_reg'],
        random_seed=seed,thread_count=-1,task_type=device,devices='0' if device=='GPU' else None,
        allow_writing_files=False,verbose=verbose)
    model.fit(X,y,cat_features=X.select_dtypes(include=['object']).columns.tolist())
    return model


def past_dispersion(reference, target, query):
    """Training queries exclude their entire month; validation reference is frozen."""
    ref=reference.reset_index(drop=True).copy()
    ref['MONTHLY_RENT']=np.asarray(target,dtype=float)
    q=query.drop(columns='MONTHLY_RENT',errors='ignore').reset_index(drop=True)
    result=pd.DataFrame({'count':np.zeros(len(q),dtype=int),
                         'adjusted_relative_iqr':np.full(len(q),np.nan)})
    for month in sorted(q.RENT_APPROVAL_DATE.unique()):
        mask=q.RENT_APPROVAL_DATE.eq(month)
        cutoff=str(pd.Period(month,freq='M')-1)
        if not ref.RENT_APPROVAL_DATE.le(cutoff).any(): continue
        stats=reliability(ref,q.loc[mask],cutoff)
        result.loc[mask,['count','adjusted_relative_iqr']]=stats[['count','adjusted_relative_iqr']].to_numpy()
    return result


def shrink_features(original,stats):
    """Only two finest-level columns change; alpha=5 becomes 10 for high dispersion.

    From old=(total+5*parent)/(mass+5), new=(total+10*parent)/(mass+10).
    This is a fixed hypothesis, not a searched strength. Counts/age stay unchanged.
    """
    if len(original)!=len(stats): raise ValueError('Feature/statistic row mismatch')
    changed=original.copy()
    high=(stats['count'].ge(5)&stats.adjusted_relative_iqr.gt(.25)).to_numpy()
    mass=np.expm1(original.ADAPT_ADDRESS_FLAT_AREA_LOG_MASS.to_numpy(float))
    ratio=original[PREMIUM].to_numpy(float)
    parent=original.ADAPT_ADDRESS_FLAT_PREMIUM.to_numpy(float)
    updated=((mass+5)*ratio+5*parent)/(mass+10)
    changed.loc[high,PREMIUM]=updated[high]
    changed.loc[high,LEVEL]=(original.ADAPT_MARKET_LEVEL.to_numpy(float)*updated)[high]
    return changed,high


def fixed_offset(original,residual):
    return original[LEVEL].to_numpy(float).copy() if residual else np.zeros(len(original))


def fit_head(X,y,original,residual,iterations,device,seed):
    # Explicitly disable modeling.target_offset: manual targets retain OLD offsets
    # even when X contains the changed finest-level reference.
    config=dict(iterations=iterations,learning_rate=.04,depth=8,l2_leaf_reg=6.,residual_target=False)
    offset=fixed_offset(original,residual)
    return fit_model(X,np.asarray(y)-offset,config,seed,device,verbose=200)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/dispersion_shrinkage_cpu_v1')
    parser.add_argument('--task-type',choices=['CPU','GPU'],default='CPU')
    parser.add_argument('--windows',nargs='+',choices=WINDOWS,default=list(WINDOWS))
    parser.add_argument('--iterations',type=int,default=1600)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.iterations<1: raise ValueError('iterations must be positive')
    raw_path=ROOT/'data/raw/train.csv'; raw=pd.read_csv(raw_path)
    from src.check_setup import RAW_SHA256
    if digest(raw_path)!=RAW_SHA256['train.csv']: raise ValueError('Unexpected training CSV checksum')
    aux=ROOT/'data/raw/auxiliary'
    paths=[raw_path,*sorted(aux.glob('*.csv')),Path(__file__),ROOT/'src/features.py',ROOT/'src/modeling.py',
           ROOT/'src/research/history_reliability.py',ROOT/'src/research/horizon_blend.py']
    signature=dict(hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},
        device=args.task_type,seed=args.seed,iterations=args.iterations,windows=args.windows,
        versions={p:version(p) for p in ['catboost','numpy','pandas','scikit-learn']},python=platform.python_version(),
        alpha_original=5,alpha_high_dispersion=10,threshold=.25,min_history=5,
        scope='Normal-history direct/residual pair only; not full eight-model mainline',
        offset='Original finest-level offset retained for both recipes',
        protocol='Fixed budget; no outer-label early stopping; identical rows; no weight optimization',
        acceptance='Recent and long-window pair gains required for further confirmation; no automatic promotion')
    out=args.output
    if out.exists():
        if not args.resume: raise FileExistsError('Use a fresh output path or --resume')
        if not (out/'metadata.json').exists() or json.loads((out/'metadata.json').read_text())!=signature:
            raise ValueError('Resume signature differs or metadata missing')
    else:
        out.mkdir(parents=True)
        (out/'metadata.json').write_text(json.dumps(signature,indent=2)+'\n')
    lookup=None; records=[]; groups=[]; changes=[]
    for window in args.windows:
        start,end=WINDOWS[window]
        train=raw.loc[raw.RENT_APPROVAL_DATE.lt(start)].reset_index(drop=True)
        mask=raw.RENT_APPROVAL_DATE.between(start,end)
        ids=np.flatnonzero(mask); valid=raw.loc[mask].reset_index(drop=True)
        yt=train.MONTHLY_RENT.to_numpy(); yv=valid.MONTHLY_RENT.to_numpy()
        tr=train.drop(columns='MONTHLY_RENT'); va=valid.drop(columns='MONTHLY_RENT')
        feature_path=out/f'{window}_features.pkl'
        if feature_path.exists():
            # Only this locally created, signature-checked experiment cache is read.
            bundle=pd.read_pickle(feature_path)
        else:
            print('BUILD FEATURES',window,flush=True);clock=time.perf_counter()
            if lookup is None: lookup=build_block_lookup(aux)
            a=build_features(tr,yt,tr,lookup,'local_trend')
            b=build_features(tr,yt,va,lookup,'local_trend')
            sa=past_dispersion(tr,yt,tr); sb=past_dispersion(tr,yt,va)
            ac,ha=shrink_features(a,sa); bc,hb=shrink_features(b,sb)
            bundle=dict(original_train=a,original_valid=b,candidate_train=ac,candidate_valid=bc,
                        stats_train=sa,stats_valid=sb,high_train=ha,high_valid=hb,
                        seconds=time.perf_counter()-clock)
            pd.to_pickle(bundle,feature_path)
        changes.append(dict(window=window,train_rows=len(train),valid_rows=len(valid),
                            high_train=int(bundle['high_train'].sum()),high_valid=int(bundle['high_valid'].sum()),
                            feature_seconds=bundle['seconds']))
        estimates={}
        for recipe in ['original','candidate']:
            for mode in ['direct','residual']:
                stem=f'{window}_{recipe}_{mode}'; cache=out/f'{stem}.npz'
                if cache.exists():
                    with np.load(cache) as z:
                        np.testing.assert_array_equal(z['row_id'],ids);np.testing.assert_array_equal(z['y'],yv)
                        pred=z['prediction']; elapsed=float(z['seconds'])
                    print('REUSE',stem,flush=True)
                else:
                    print('FIT',stem,flush=True);clock=time.perf_counter()
                    model=fit_head(bundle[recipe+'_train'],yt,bundle['original_train'],mode=='residual',args.iterations,args.task_type,args.seed)
                    pred=model.predict(bundle[recipe+'_valid'])+fixed_offset(bundle['original_valid'],mode=='residual')
                    elapsed=time.perf_counter()-clock
                    if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
                    model.save_model(str(out/f'{stem}.cbm'))
                    np.savez_compressed(cache,row_id=ids,y=yv,prediction=pred,seconds=elapsed)
                estimates[recipe+'_'+mode]=pred
                records.append(dict(window=window,recipe=recipe,mode=mode,rows=len(valid),seconds=elapsed,**metrics(yv,pred)))
                pd.DataFrame(records).to_csv(out/'results.csv',index=False)
            estimates[recipe+'_pair']=(estimates[recipe+'_direct']+estimates[recipe+'_residual'])/2
        for recipe in ['original','candidate']:
            pred=estimates[recipe+'_pair']
            score=metrics(yv,pred);baseline=metrics(yv,estimates['original_pair'])
            records.append(dict(window=window,recipe=recipe,mode='pair',rows=len(valid),**score,
                                rmse_gain_vs_original=baseline['rmse']-score['rmse']))
            stat=bundle['stats_valid']; strata={'all':np.ones(len(valid),bool),
                'high':bundle['high_valid'],'low':(stat['count'].ge(5)&stat.adjusted_relative_iqr.le(.10)).to_numpy(),
                'other':~bundle['high_valid']}
            for name,selection in strata.items():
                if selection.any():groups.append(dict(window=window,recipe=recipe,group=name,rows=int(selection.sum()),
                    **metrics(yv[selection],pred[selection]),sse_share=float(np.sum((pred[selection]-yv[selection])**2)/np.sum((pred-yv)**2))))
        pd.DataFrame(records).to_csv(out/'results.csv',index=False)
        pd.DataFrame(groups).to_csv(out/'groups.csv',index=False)
        pd.DataFrame(changes).to_csv(out/'feature_changes.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print('Research only. Compare same-device pairs; do not compare these scores directly with frozen eight-model RMSE.')

if __name__=='__main__':main()
