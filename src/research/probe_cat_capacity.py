"""Two prespecified CatBoost capacity changes on unchanged official labels."""

import argparse
import hashlib
import json
from pathlib import Path
from src.research.cache_inputs import add_cache_arguments, reference_run, housing_run

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.research.hdb_attribute_features import course_keys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"artifacts/cat_capacity_20261102"
CACHES={"short_recent":"e925f89746af432b","long_recent":"17bad5d96e4b3667","long_older":"17bad5d96e4b3667"}
VARIANTS={
    "depth10_l2_20":dict(depth=10,l2_leaf_reg=20),
    "lossguide64_l2_20":dict(depth=10,l2_leaf_reg=20,grow_policy="Lossguide",max_leaves=64,min_data_in_leaf=100),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows',nargs='+',choices=list(CACHES),default=['short_recent'])
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    add_cache_arguments(parser)
    args=parser.parse_args()
    parents = {w: reference_run(args, ROOT/'artifacts/research_20261102'/CACHES[w], [w]) for w in args.windows}
    OUT = args.output_dir or globals()['OUT']
    OUT.mkdir(parents=True, exist_ok=True)
    protocol=dict(variants={k:VARIANTS[k] for k in args.variants},windows=args.windows,
                  iterations=1600,learning_rate=.04,seed=42,task_type="CPU",threads=8,
                  reference="Frozen four-head seed42 model; only local_trend direct head replaced",
                  information="Same 78 features, same official training labels; no outlier deletion, no external targets",
                  stopping="Two variants, no early stopping on outer validation. Require >=2 RMSE improvement and upper address-cluster delta CI <0 for a replacement or standalone candidate before long_recent and long_older confirmation.",
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    protocol['reference_cache_dirs'] = {w:str(p) for w,p in parents.items()}
    identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
    folder=OUT/identity;folder.mkdir(exist_ok=True)
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
    source=pd.read_csv(ROOT/'data/raw/train.csv')
    records=[]
    for window in args.windows:
        parent=parents[window]
        left,right,ytrain,weights,_=pd.read_pickle(parent/f'{window}_local_trend_0.pkl')
        scoring=pd.read_csv(parent/f'{window}_scoring_rows.csv');y=scoring.target.to_numpy()
        original=np.load(parent/f'{window}_reference_local_trend_0_direct_42.npy')
        old=[np.load(parent/f'{window}_reference_{v}_{m}_42.npy') for v,m in
             [('local_trend_0','residual'),('hierarchical_6','direct'),('hierarchical_6','residual')]]
        reference=np.mean(old+[original],axis=0)
        predictions={'reference':reference,'original_direct':original}
        for name in args.variants:
            file=folder/f'{window}_{name}.cbm'
            model=CatBoostRegressor(iterations=1600,learning_rate=.04,random_seed=42,
                                    loss_function='RMSE',thread_count=8,task_type='CPU',
                                    allow_writing_files=False,verbose=400,**VARIANTS[name])
            print(f'Fitting {window}/{name}',flush=True)
            if file.exists():model.load_model(str(file))
            else:
                model.fit(left,ytrain,sample_weight=weights,
                          cat_features=left.select_dtypes(include=['object','string','category']).columns.tolist())
                model.save_model(str(file))
            p=model.predict(right)
            predictions[name+'_standalone']=p
            predictions[name+'_replacement']=np.mean(old+[p],axis=0)
        query=source.iloc[scoring.source_row].reset_index(drop=True)
        keys=course_keys(query)
        groups,names=pd.factorize(keys.town+'|'+keys.block+'|'+keys.street)
        draws=np.random.default_rng(20261102).multinomial(len(names),np.full(len(names),1/len(names)),size=1000)
        mass=draws@np.bincount(groups)
        base=np.sqrt(np.einsum('ij,j->i',draws,np.bincount(groups,weights=(reference-y)**2),optimize=False)/mass)
        for name,p in predictions.items():
            assert p.shape==y.shape and np.isfinite(p).all()
            np.save(folder/f'{window}_{name}.npy',p)
            delta=np.sqrt(np.einsum('ij,j->i',draws,np.bincount(groups,weights=(p-y)**2),optimize=False)/mass)-base
            records.append(dict(window=window,candidate=name,rmse=float(np.sqrt(np.mean((p-y)**2))),
                                mae=float(np.abs(p-y).mean()),bias=float((p-y).mean()),
                                delta_rmse=float(np.sqrt(np.mean((p-y)**2))-np.sqrt(np.mean((reference-y)**2))),
                                delta_ci_low=float(np.quantile(delta,.025)),delta_ci_high=float(np.quantile(delta,.975))))
        result=pd.DataFrame(records);result.to_csv(folder/'metrics.csv',index=False)
        print(result.to_string(index=False),flush=True)
    print(f'Completed {folder}',flush=True)


if __name__=='__main__':main()
