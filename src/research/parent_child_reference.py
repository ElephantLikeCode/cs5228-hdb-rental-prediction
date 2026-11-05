"""Paired diagnostic of existing market-adjusted references; no training/tuning."""
from pathlib import Path
from itertools import combinations
import argparse
import json
import numpy as np
import pandas as pd
from src.features import add_adaptive_rent_features, normalize_text
from src.research.history_reliability import CORE, reliability, verify_sources
from src.research.horizon_blend import ROOT, load_window, metrics, digest

LEVELS=['TOWN_FLAT','STREET_FLAT','ADDRESS_FLAT','ADDRESS_FLAT_AREA']


def paired(parent,child,y,clusters,bootstrap=1000):
    pe=(np.asarray(parent)-y)**2; ce=(np.asarray(child)-y)**2
    group,_=pd.factorize(clusters)
    n=np.bincount(group); ps=np.bincount(group,weights=pe); cs=np.bincount(group,weights=ce)
    rng=np.random.default_rng(42); gains=[]; deltas=[]
    for start in range(0,bootstrap,100):
        draw=rng.integers(len(n),size=(min(100,bootstrap-start),len(n)))
        count=n[draw].sum(axis=1)
        p=ps[draw].sum(axis=1)/count; c=cs[draw].sum(axis=1)/count
        gains.extend(np.sqrt(c)-np.sqrt(p)); deltas.extend(c-p)
    low,high=np.quantile(gains,[.025,.975]); ml,mh=np.quantile(deltas,[.025,.975])
    return dict(rows=len(y),clusters=len(n),parent_win_rate=float(np.mean(pe<ce)),
                tie_rate=float(np.mean(np.isclose(pe,ce,rtol=0,atol=1e-8))),
                mse_gain_child_minus_parent=float(np.mean(ce-pe)),
                rmse_gain_child_minus_parent=float(np.sqrt(ce.mean())-np.sqrt(pe.mean())),
                rmse_gain_ci_low=float(low),rmse_gain_ci_high=float(high),
                mse_gain_ci_low=float(ml),mse_gain_ci_high=float(mh),
                conclusion='parent_better' if low>0 else 'child_better' if high<0 else 'uncertain')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/parent_child_reference_v1')
    args=parser.parse_args();out=args.output
    if out.exists(): raise FileExistsError('Use a new output directory')
    # Capture prior artifacts and production files without changing them.
    protected={str(p):(p.stat().st_size,p.stat().st_mtime_ns)
               for root in ['data','artifacts','reports','configs','submissions','docs','src']
               for p in (ROOT/root).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    raw=pd.read_csv(ROOT/'data/raw/train.csv')
    sources=verify_sources(pd.read_csv(ROOT/'reports/stacking_sources.csv'),pd.read_csv(ROOT/'reports/temporal_comparison.csv'),raw)
    previous=pd.read_csv(ROOT/'artifacts/history_reliability_diagnostic_v1/group_metrics.csv')
    rows=[]; pairs=[]; monthly=[]; manifests=[]
    for window in CORE:
        print('CHECK',window,flush=True)
        v=load_window(sources,window,ROOT,raw)
        q=raw.iloc[v['row_id']].drop(columns='MONTHLY_RENT').reset_index(drop=True)
        ref=raw.loc[raw.RENT_APPROVAL_DATE.le(v['cutoff'])].reset_index(drop=True)
        diagnostic=reliability(ref,q,v['cutoff'])
        features=add_adaptive_rent_features(ref.drop(columns='MONTHLY_RENT'),ref.MONTHLY_RENT,q)
        predictions={k:features['ADAPT_'+k+'_LEVEL'].to_numpy() for k in LEVELS}
        for k,p in predictions.items():
            assert np.isfinite(p).all()
            np.testing.assert_allclose(p, features.ADAPT_MARKET_LEVEL*features['ADAPT_'+k+'_PREMIUM'])
        # Address cluster includes town/block/street, not room/type: repeated leases stay together.
        clusters=pd.MultiIndex.from_frame(q[['TOWN','BLOCK','STREET']].apply(normalize_text))
        totals={k:float(np.sum((p-v['y'])**2)) for k,p in predictions.items()}
        definitions={'all':np.ones(len(q),dtype=bool)}
        definitions.update({g:diagnostic.dispersion.eq(g).to_numpy() for g in diagnostic.dispersion.unique()})
        for label,mask in definitions.items():
            if label!='all':
                old=previous.loc[previous.window.eq(window)&previous.branch.eq('E')&previous.dimension.eq('dispersion')&previous.group.eq(label)]
                assert len(old)==1 and int(old.rows.iloc[0])==int(mask.sum())
                np.testing.assert_allclose(metrics(v['y'][mask],(v['normal'][mask]+v['stale'][mask])/2)['rmse'],old.rmse.iloc[0],atol=1e-8)
            for level,p in predictions.items():
                sse=float(np.sum((p[mask]-v['y'][mask])**2))
                rows.append(dict(window=window,group=label,level=level,rows=int(mask.sum()),row_share=float(mask.mean()),
                                 **metrics(v['y'][mask],p[mask]),sse=sse,sse_share=sse/totals[level]))
            for parent,child in combinations(LEVELS,2):
                pairs.append(dict(window=window,group=label,parent=parent,child=child,
                                  **paired(predictions[parent][mask],predictions[child][mask],v['y'][mask],clusters[mask])))
        # No extra fitting or bootstrap search: monthly point differences in the high-volatility group.
        for month in sorted(q.RENT_APPROVAL_DATE.unique()):
            mask=(q.RENT_APPROVAL_DATE.eq(month)&diagnostic.dispersion.eq('high_gt_0.25')).to_numpy()
            if not mask.any():continue
            for parent,child in zip(LEVELS[:-1],LEVELS[1:]):
                pe=(predictions[parent][mask]-v['y'][mask])**2;ce=(predictions[child][mask]-v['y'][mask])**2
                monthly.append(dict(window=window,month=month,parent=parent,child=child,rows=int(mask.sum()),
                                    rmse_gain_child_minus_parent=float(np.sqrt(ce.mean())-np.sqrt(pe.mean()))))
        manifests.append(dict(window=window,cutoff=v['cutoff'],rows=len(q),run_ids=v['run_ids']))
    changed=[p for p,s in protected.items() if not Path(p).exists() or (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=s]
    if changed:raise RuntimeError(f'Protected files changed: {changed}')
    out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(out/'reference_metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame(monthly).to_csv(out/'high_dispersion_monthly.csv',index=False)
    sources[sources.window.isin(CORE)].to_csv(out/'sources.csv',index=False)
    (out/'audit.json').write_text(json.dumps(dict(windows=manifests,protected_files_unchanged=len(protected),
        bootstrap={'unit':'normalized town/block/street','replicates':1000,'seed':42,'interval':'percentile 95%, exploratory, not multiplicity adjusted'},
        definitions={'bias':'prediction minus actual','gain':'child loss minus parent loss; positive favors parent',
                     'sse_share':'group SSE divided by same level/window total SSE',
                     'scale':'all ADAPT_*_LEVEL = same forecast-month ADAPT_MARKET_LEVEL * smoothed historical premium',
                     'references':'production smoothed child versus its production smoothed parent, not raw group means',
                     'dispersion':'unchanged origin-frozen IQR/median of market-normalized rents, n>=5; high >0.25, low <=0.10'},
        limitations=['overlapping development windows','reference accuracy is not final model accuracy','branch source revisions preserved',
                     'address clusters are not unique units','bootstrap reflects address resampling, not new market regimes'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'src/features.py',ROOT/'src/research/history_reliability.py',ROOT/'src/research/horizon_blend.py',ROOT/'data/raw/train.csv']}),indent=2)+'\n')
    print('Saved',out)

if __name__=='__main__':main()
