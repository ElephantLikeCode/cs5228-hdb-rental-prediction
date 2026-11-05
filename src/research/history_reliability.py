"""Read-only frozen-prediction reliability diagnosis; no estimator fitting."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from src.features import normalize_text, _history_keys, add_adaptive_rent_features
from src.research.horizon_blend import ROOT, audit_sources, load_window, digest, metrics
from src.research.support_blend import support_counts

CORE = ['short_recent', 'long_recent', 'long_older', 'horizon16']
KEYS = ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE']


def reliability(reference, query, cutoff):
    """No query targets accepted/used. Origin-frozen records, query-relative ages."""
    a = reference.loc[reference.RENT_APPROVAL_DATE.le(cutoff)].copy().reset_index(drop=True)
    b = query.drop(columns=['MONTHLY_RENT'], errors='ignore').copy().reset_index(drop=True)
    if a.empty or not b.RENT_APPROVAL_DATE.gt(cutoff).all():
        raise ValueError('History must precede every query month')
    for c in KEYS:
        a[c] = normalize_text(a[c]); b[c] = normalize_text(b[c])
    a['m'] = pd.PeriodIndex(a.RENT_APPROVAL_DATE, freq='M').asi8
    qm = pd.PeriodIndex(b.RENT_APPROVAL_DATE, freq='M').asi8
    anchor = a.m.max()
    a['w'] = np.exp2((a.m-anchor)/12.)
    a['w2'] = a.w**2
    a['wm'] = a.w*a.m
    a['recent'] = (a.m >= anchor-5).astype(int)
    market = a.groupby(['FLAT_TYPE', 'm']).MONTHLY_RENT.transform('mean')
    a['ratio'] = a.MONTHLY_RENT/market
    valid = a[KEYS].notna().all(axis=1) & a[KEYS].fillna('').ne('').all(axis=1)
    a = a.loc[valid]
    gb = a.groupby(KEYS)
    stats = gb.agg(count=('MONTHLY_RENT','size'), rent_std=('MONTHLY_RENT','std'),
                   rent_mean=('MONTHLY_RENT','mean'), rent_median=('MONTHLY_RENT','median'),
                   ratio_std=('ratio','std'), ratio_median=('ratio','median'),
                   last=('m','max'), mean_month=('m','mean'), mass=('w','sum'),
                   mass2=('w2','sum'), wm=('wm','sum'), recent=('recent','sum'))
    for value in ['MONTHLY_RENT','ratio']:
        stats[value+'_iqr'] = gb[value].quantile(.75)-gb[value].quantile(.25)
    matched = stats.reindex(pd.MultiIndex.from_frame(b[KEYS])).reset_index(drop=True)
    good = b[KEYS].notna().all(axis=1) & b[KEYS].fillna('').ne('').all(axis=1)
    matched.loc[~good,:] = np.nan
    r = pd.DataFrame(index=b.index)
    r['count'] = matched['count'].fillna(0).astype(int)
    r['mass'] = matched.mass.fillna(0)
    r['kish_ess'] = (matched.mass**2/matched.mass2).fillna(0)
    r['mass_per_record'] = r.mass/r['count'].replace(0,np.nan)
    r['age'] = qm-matched['last']
    r['origin_gap'] = anchor-matched['last']
    r['mean_history_age_at_origin'] = anchor-matched.mean_month
    r['weighted_history_age_at_origin'] = anchor-matched.wm/matched.mass
    r['recent_share'] = matched.recent/r['count'].replace(0,np.nan)
    for c in ['rent_std','ratio_std']:
        r[c] = matched[c]
    r['rent_iqr'] = matched.MONTHLY_RENT_iqr
    r['raw_relative_iqr'] = matched.MONTHLY_RENT_iqr/matched.rent_median
    r['adjusted_relative_iqr'] = matched.ratio_iqr/matched.ratio_median
    r['support'] = pd.cut(r['count'],[-1,4,19,np.inf],labels=['0-4','5-19','20+']).astype(str)
    r['dispersion'] = pd.cut(r.adjusted_relative_iqr,[-np.inf,.10,.25,np.inf],labels=['low_le_0.10','mid_0.10_0.25','high_gt_0.25']).astype(str)
    r.loc[r['count']<5,'dispersion'] = 'insufficient_n_lt5'
    for c in ['age','origin_gap']:
        bins = [-np.inf,3,12,np.inf] if c=='age' else [-np.inf,2,6,np.inf]
        labels = ['1-3','4-12','13+'] if c=='age' else ['0-2','3-6','7+']
        r[c+'_group'] = pd.cut(r[c],bins,labels=labels).astype(str)
        r.loc[r['count']==0,c+'_group'] = 'no_history'
    r['mass_group'] = pd.cut(r.mass,[-np.inf,2,10,np.inf],labels=['0-2','2-10','10+']).astype(str)
    r['month'] = b.RENT_APPROVAL_DATE.to_numpy()
    r['flat_type'] = b.FLAT_TYPE.fillna('missing').to_numpy()
    r['town'] = b.TOWN.fillna('missing').to_numpy()
    r['area'] = pd.cut(pd.to_numeric(b.FLOOR_AREA_SQM),[-np.inf,70,100,np.inf],labels=['le70','70-100','gt100']).astype(str)
    return r


def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else np.nan


def group_scores(frame, columns, window):
    records=[]
    totals={model:float((frame[model]**2).sum()) for model in ['N','S','E']}
    for key,g in frame.groupby(columns,dropna=False,observed=True):
        if not isinstance(key,tuple): key=(key,)
        shared=dict(window=window, dimension='|'.join(columns),group='|'.join(map(str,key)),
                    rows=len(g),row_share=len(g)/len(frame), residual_correlation=corr(g.N,g.S),
                    mse_N_minus_S=float((g.N**2-g.S**2).mean()),
                    rmse_N_minus_S=float(np.sqrt((g.N**2).mean())-np.sqrt((g.S**2).mean())),
                    mean_count=float(g['count'].mean()),mean_mass=float(g.mass.mean()),
                    mean_age=float(g.age.mean()), small_group=len(g)<100)
        for model in ['N','S','E']:
            errors=g[model].to_numpy(); sse=float(np.sum(errors**2))
            records.append(dict(**shared,branch=model,**metrics(np.zeros(len(g)),errors),sse=sse,
                                sse_share=sse/totals[model]))
    return records


def stratified_contrast(f, variable, low, high, strata, window):
    """Descriptive pooled-stratum standardization, not a fitted prediction rule."""
    rows=[]
    for _,g in f.groupby(strata,observed=True):
        a=g.loc[g[variable].eq(low)]; b=g.loc[g[variable].eq(high)]
        if len(a)<20 or len(b)<20: continue
        rows.append((len(a),len(b),float((a.E**2).mean()),float((b.E**2).mean())))
    if not rows: return dict(window=window,variable=variable,strata='|'.join(strata),cells=0)
    v=np.array(rows); w=v[:,:2].sum(axis=1); w=w/w.sum()
    return dict(window=window,variable=variable,strata='|'.join(strata),low=low,high=high,cells=len(rows),
                low_rows=int(v[:,0].sum()),high_rows=int(v[:,1].sum()),
                low_coverage=float(v[:,0].sum()/max(1,f[variable].eq(low).sum())),
                high_coverage=float(v[:,1].sum()/max(1,f[variable].eq(high).sum())),
                standardized_low_rmse=float(np.sqrt(w@v[:,2])),standardized_high_rmse=float(np.sqrt(w@v[:,3])),
                standardized_mse_high_minus_low=float(w@(v[:,3]-v[:,2])))


def verify_sources(manifest,ledger,raw):
    sources,missing=audit_sources(manifest,ROOT,sorted(manifest.window.unique()))
    if missing: raise FileNotFoundError(str(missing))
    if digest(ROOT/'data/raw/train.csv')!=sources.data_sha256.iloc[0]: raise ValueError('Data hash mismatch')
    fields=['window','candidate','seed','train_end','validation_start','validation_end','iterations',
            'actual_trees','task_type','features','feature_count','data_sha256','code_sha256','protocol']
    for row in sources.itertuples():
        match=ledger.loc[ledger.run_id.eq(row.run_id)]
        if len(match)!=1: raise ValueError('Ledger run ID missing/duplicated')
        for c in fields:
            if str(match.iloc[0][c])!=str(getattr(row,c)): raise ValueError(f'Ledger mismatch {row.run_id}/{c}')
        cfg=json.loads(row.config_json)
        if cfg!=json.loads(match.iloc[0].config_json): raise ValueError('Config ledger mismatch')
        if str(match.iloc[0].prediction_file).replace('\\','/')!=row.prediction_file.replace('\\','/'):
            raise ValueError('Path ledger mismatch')
        normal=row.candidate.startswith('local_trend')
        required={'history':'local_trend' if normal else 'hierarchical','depth':8,'iterations':1600,
                  'learning_rate':.04,'l2_leaf_reg':6.,'residual_target':row.candidate.endswith('residual'),
                  'outlier_policy':'keep'}
        if any(cfg.get(k)!=v for k,v in required.items()) or cfg.get('augmentation_lag',0)!=(0 if normal else 6):
            raise ValueError('Unexpected branch configuration')
        if cfg.get('weight_half_life') is not None or cfg.get('normalize_target',False):
            raise ValueError('Unexpected training/target weighting')
        actual=raw.RENT_APPROVAL_DATE.lt(row.validation_start)
        if int(actual.sum())!=int(row.train_rows) or raw.loc[actual,'RENT_APPROVAL_DATE'].max()!=row.train_end:
            raise ValueError('Training cutoff/count mismatch')
    return sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/history_reliability_diagnostic_v1')
    args=parser.parse_args(); out=args.output
    if (out/'audit.json').exists(): raise FileExistsError('Use new output directory')
    raw=pd.read_csv(ROOT/'data/raw/train.csv'); test=pd.read_csv(ROOT/'data/raw/test.csv')
    assert len(raw)==150000 and len(test)==50000 and raw.MONTHLY_RENT.notna().all()
    sources=verify_sources(pd.read_csv(ROOT/'reports/stacking_sources.csv'),pd.read_csv(ROOT/'reports/temporal_comparison.csv'),raw)
    all_groups=[]; summaries=[]; correlations=[]; contrasts=[]; definitions=[]; existing=[]
    old=pd.read_csv(ROOT/'reports/followup_transfer.csv')
    support_old=pd.read_csv(ROOT/'artifacts/support_blend_v1/by_support.csv')
    for window in sources.window.unique():
        print('DIAGNOSE',window,flush=True)
        v=load_window(sources,window,ROOT,raw)
        query=raw.iloc[v['row_id']].drop(columns='MONTHLY_RENT').reset_index(drop=True)
        reference=raw.loc[raw.RENT_APPROVAL_DATE.le(v['cutoff'])].reset_index(drop=True)
        f=reliability(reference,query,v['cutoff'])
        np.testing.assert_array_equal(f['count'],support_counts(raw,v['row_id'],v['cutoff']))
        f['N']=v['normal']-v['y']; f['S']=v['stale']-v['y']; f['E']=(f.N+f.S)/2
        f['all']='all'
        dimensions=[['all'],['month'],['flat_type'],['town'],['area'],['support'],['dispersion'],['age_group'],
                    ['origin_gap_group'],['mass_group'],['support','dispersion'],['support','age_group']]
        for d in dimensions: all_groups.extend(group_scores(f,d,window))
        for var in ['count','mass','kish_ess','mass_per_record','age','origin_gap','mean_history_age_at_origin',
                    'weighted_history_age_at_origin','recent_share','rent_std','rent_iqr','raw_relative_iqr','adjusted_relative_iqr']:
            mask=f[var].notna()
            if 'std' in var or 'iqr' in var: mask &= f['count'].ge(5)
            for target in ['squared_error','abs_error']:
                err=f.E**2 if target=='squared_error' else f.E.abs()
                correlations.append(dict(window=window,variable=var,target=target,rows=int(mask.sum()),
                                          spearman=float(f.loc[mask,var].corr(err[mask],method='spearman'))))
        correlations.append(dict(window=window,variable='mass_vs_age',target='age',rows=int(f.age.notna().sum()),
                                  spearman=float(f.mass.corr(f.age,method='spearman'))))
        for variable,low,high,strata_sets in [
            ('dispersion','low_le_0.10','high_gt_0.25',[['support','flat_type','month'],['support','flat_type','month','area'],['support','flat_type','town']]),
            ('support','20+','0-4',[['flat_type','month'],['flat_type','month','area'],['flat_type','town']]),
            ('age_group','1-3','13+',[['support','flat_type'],['support','flat_type','month']]),
            ('origin_gap_group','0-2','7+',[['support','flat_type','month'],['support','flat_type','town']])]:
            for strata in strata_sets: contrasts.append(stratified_contrast(f,variable,low,high,strata,window))
        for method,model in [('selected','E'),('normal_only','N'),('stale_only','S')]:
            past=old.loc[old.window.eq(window)&old.method.eq(method)]
            score=metrics(np.zeros(len(f)),f[model].to_numpy())
            if len(past):
                np.testing.assert_allclose(score['rmse'],past.rmse.iloc[0],atol=1e-8)
                existing.append(dict(window=window,method=method,matched_existing=True))
        # Reuse prior support analysis as a cross-check, not new evidence.
        mapped={'short_recent':'quarter_2025','long_recent':'year_2024','long_older':'year_2023','horizon16':'horizon16'}
        if window in mapped:
            for label,old_label in [('0-4','sparse_0_4'),('5-19','medium_5_19'),('20+','dense_20_plus')]:
                previous=support_old.loc[support_old.plan.eq(mapped[window])&support_old.method.eq('equal_50_50')&support_old.group.eq(old_label)]
                if len(previous): np.testing.assert_allclose(np.sqrt((f.loc[f.support.eq(label),'E']**2).mean()),previous.rmse.iloc[0],atol=1e-8)
        if window in CORE:
            # Rebuild only deterministic historical features, no spatial features or model.
            adaptive=add_adaptive_rent_features(reference.drop(columns='MONTHLY_RENT'),reference.MONTHLY_RENT,query)
            keys=_history_keys(reference); qkeys=_history_keys(query)
            prod_count=keys.groupby(['_ADDRESS_KEY','_FLAT_TYPE_KEY']).size().reindex(pd.MultiIndex.from_frame(qkeys[['_ADDRESS_KEY','_FLAT_TYPE_KEY']])).fillna(0).to_numpy()
            area_count=keys.groupby(['_ADDRESS_KEY','_FLAT_TYPE_KEY','_AREA_KEY']).size().reindex(pd.MultiIndex.from_frame(qkeys[['_ADDRESS_KEY','_FLAT_TYPE_KEY','_AREA_KEY']])).fillna(0).to_numpy()
            mass=np.expm1(adaptive.ADAPT_ADDRESS_FLAT_LOG_MASS.to_numpy())
            seen=f['count'].gt(0)
            np.testing.assert_allclose(mass,f.mass,atol=1e-8)
            np.testing.assert_allclose(adaptive.loc[seen,'ADAPT_ADDRESS_FLAT_AGE'],f.loc[seen,'age'],atol=1e-8)
            definitions.append(dict(window=window,address_count_mismatches=int((prod_count!=f['count']).sum()),
                                    area_count_mismatches=int((area_count!=prod_count).sum()),
                                    production_mass_age_verified=True,no_history=int((~seen).sum())))
        for c in ['count','mass','kish_ess','age','origin_gap','recent_share','adjusted_relative_iqr']:
            summaries.append(dict(window=window,variable=c,missing=int(f[c].isna().sum()),
                                  p10=float(f[c].quantile(.1)),p50=float(f[c].median()),p90=float(f[c].quantile(.9))))
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('group_metrics',all_groups),('correlations',correlations),('stratified_contrasts',contrasts),
                      ('reliability_summary',summaries),('production_feature_checks',definitions),('existing_result_checks',existing)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    sources.to_csv(out/'sources.csv',index=False)
    audit=dict(train_rows=len(raw),test_rows=len(test),train_range=[raw.RENT_APPROVAL_DATE.min(),raw.RENT_APPROVAL_DATE.max()],
               test_range=[test.RENT_APPROVAL_DATE.min(),test.RENT_APPROVAL_DATE.max()],cache_files=len(sources),
               prediction_rows_aligned=True,ledger_and_configs_verified=True,
               source_code_hashes=sources.groupby('candidate').code_sha256.unique().apply(list).to_dict(),
               source_hash_note='Historical branch revisions differ in core windows; frozen manifest preserved. Same data/budget/device/seeds/recipe definitions verified, not a single-revision causal ablation.',
               definitions=dict(signed_error='prediction minus actual (positive=overprediction)',
                   sse_share='group SSE / same-window same-branch total SSE',
                   dispersion='IQR/median of prior rent divided by same-flat-type/month prior mean; n>=5; bins <=.10, .10-.25, >.25',
                   mass='sum 2**((record_month-origin_latest_month)/12); not raw count or Kish ESS',
                   kish_ess='sum(w)^2/sum(w^2)',age='query month minus last comparable month',
                   origin_gap='origin latest month minus last comparable month; removes forecast-horizon shift',
                   strata='at least 20 rows in each contrast arm per cell; pooled row weights; descriptive, no causal claim'),
               code_hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'src/features.py',ROOT/'src/modeling.py',ROOT/'src/research/horizon_blend.py',ROOT/'src/research/support_blend.py']},
               data_hash=digest(ROOT/'data/raw/train.csv'))
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Saved',out)


if __name__=='__main__': main()
