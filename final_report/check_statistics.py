"""Check raw/canonical address bootstrap sensitivity from frozen local evidence.

Run from the repository root: python final_report/check_statistics.py.
No training or changes to the selected prediction files are performed.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
from features import normalize_text
from feedback_experiments import frozen_baseline
raw = pd.read_csv(root/'data/raw/train.csv')
ledger = pd.read_csv(root/'reports/temporal_comparison.csv')
results=[]
for window in ['short_recent','long_recent','long_older','horizon16']:
    ids, pred, sources = frozen_baseline(window, raw, raw.MONTHLY_RENT)
    refs=[]
    for candidate in ['hierarchical','market_residual']:
        for seed in [42,2026]:
            row=ledger.loc[(ledger.window==window)&(ledger.candidate==candidate)&(ledger.seed==seed)&(ledger.iterations==1600)&(ledger.task_type=='GPU')].iloc[-1]
            with np.load(root/row.prediction_file) as saved:
                assert np.array_equal(ids,saved['row_id'])
                assert np.array_equal(raw.MONTHLY_RENT.iloc[ids],saved['y'])
                refs.append(saved['prediction'].copy())
    ref=np.mean(refs,axis=0); y=raw.MONTHLY_RENT.iloc[ids].to_numpy()
    item={'window':window,'rows':len(ids),'gain':float(np.sqrt(np.mean((ref-y)**2))-np.sqrt(np.mean((pred-y)**2))),'selected_source_runs':sources}
    for mode in ['raw','canonical']:
        keys=raw.iloc[ids][['TOWN','BLOCK','STREET']].copy()
        if mode=='canonical': keys=keys.apply(normalize_text)
        groups,_=pd.factorize(keys.astype(str).agg('|'.join,axis=1))
        count=np.bincount(groups); a=np.bincount(groups,weights=(ref-y)**2); b=np.bincount(groups,weights=(pred-y)**2)
        draws=np.random.default_rng(42).integers(len(count),size=(1000,len(count)))
        gains=np.sqrt(a[draws].sum(1)/count[draws].sum(1))-np.sqrt(b[draws].sum(1)/count[draws].sum(1))
        item[mode]={'clusters':len(count),'ci95':np.quantile(gains,[.025,.975]).tolist()}
    results.append(item)
output={'date':'2026-10-29','bootstrap_replicates':1000,'seed':42,'scope':'Paired development-sample sensitivity; no new model fit or promotion. Canonical keys use production normalize_text.','windows':results}
(root/'reports/pre_submission_checks.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
