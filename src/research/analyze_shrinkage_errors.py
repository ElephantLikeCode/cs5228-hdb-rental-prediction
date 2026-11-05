"""Read saved shrinkage predictions; descriptive attribution, no fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.features import normalize_text
from src.research.horizon_blend import digest,metrics
ROOT=Path(__file__).resolve().parents[2]

def main():
    src=ROOT/'artifacts/dispersion_shrinkage_cpu_v1'
    out=ROOT/'artifacts/dispersion_shrinkage_attribution_v1'
    if out.exists():raise FileExistsError(out)
    meta=json.loads((src/'metadata.json').read_text())
    for name,h in meta['hashes'].items():
        if digest(ROOT/name)!=h:raise ValueError(f'Hash mismatch {name}')
    raw=pd.read_csv(ROOT/'data/raw/train.csv');rows=[];profiles=[];checks=[]
    for window,start,end in [('short_recent','2025-01','2025-03'),('horizon16','2023-12','2025-03')]:
        ids=np.flatnonzero(raw.RENT_APPROVAL_DATE.between(start,end));q=raw.iloc[ids].reset_index(drop=True)
        y=q.MONTHLY_RENT.to_numpy(); pred={}
        for recipe in ['original','candidate']:
            for mode in ['direct','residual']:
                with np.load(src/f'{window}_{recipe}_{mode}.npz') as z:
                    np.testing.assert_array_equal(ids,z['row_id']);np.testing.assert_array_equal(y,z['y'])
                    pred[recipe,mode]=z['prediction'].copy()
            pred[recipe,'pair']=(pred[recipe,'direct']+pred[recipe,'residual'])/2
        bundle=pd.read_pickle(src/f'{window}_features.pkl')
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
            for dims in [['all'],['month'],['period'],['flat'],['dispersion'],['month','dispersion'],['month','flat'],['period','dispersion']]:
                for key,indices in q.groupby(dims,observed=True).groups.items():
                    if not isinstance(key,tuple):key=(key,)
                    idx=np.asarray(indices);ma=metrics(y[idx],a[idx]);mb=metrics(y[idx],b[idx])
                    rows.append(dict(window=window,mode=mode,dimension='|'.join(dims),group='|'.join(map(str,key)),rows=len(idx),
                        original_rmse=ma['rmse'],candidate_rmse=mb['rmse'],rmse_change=mb['rmse']-ma['rmse'],
                        original_mae=ma['mae'],candidate_mae=mb['mae'],original_bias=ma['bias'],candidate_bias=mb['bias'],
                        delta_sse=float(d[idx].sum()),delta_mse=float(d[idx].mean()),
                        original_sse_share=float(np.sum((a[idx]-y[idx])**2)/np.sum((a-y)**2)),
                        positive_loss_share=float(np.maximum(d[idx],0).sum()/np.maximum(d,0).sum()),
                        row_mean_rent=float(y[idx].mean())))
            saved=pd.read_csv(src/'results.csv')
            for recipe,p in [('original',a),('candidate',b)]:
                r=saved[(saved.window==window)&(saved.recipe==recipe)&(saved['mode']==mode)].iloc[0]
                np.testing.assert_allclose(metrics(y,p)['rmse'],r.rmse,rtol=1e-10)
        checks.append(dict(window=window,rows=len(q),row_ids_labels_and_scores_verified=True))
    result=pd.DataFrame(rows)
    for (w,m),g in result.groupby(['window','mode']):
        total=g.loc[g.dimension.eq('all'),'delta_sse'].iloc[0]
        for d in g.dimension.unique():np.testing.assert_allclose(g.loc[g.dimension.eq(d),'delta_sse'].sum(),total,atol=1e-6)
    out.mkdir()
    result.to_csv(out/'error_changes.csv',index=False)
    pd.DataFrame(profiles).to_csv(out/'monthly_data_profile.csv',index=False)
    (out/'audit.json').write_text(json.dumps(dict(checks=checks,code_hash=digest(Path(__file__)),
        input_hashes={p.name:digest(p) for p in src.glob('*.npz')},
        sign='candidate minus original: positive means deterioration',
        limitations='One seed; descriptive reused overlapping windows; no causal attribution or parameter fitting'),indent=2)+'\n')
    print(result[(result['mode']=='pair')&result.dimension.isin(['month','period','flat','dispersion'])].to_string(index=False))
if __name__=='__main__':main()
