"""Add only public historical rental records preceding the validation origin."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

try:
    from .features import build_block_lookup
    from .hdb_attribute_features_20261102 import course_keys, normal
    from .modeling import build_features, load_config
except ImportError:
    from features import build_block_lookup
    from hdb_attribute_features_20261102 import course_keys, normal
    from modeling import build_features, load_config

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "external_datagov/renting_out_of_flats_2021on.csv"
PARENT = ROOT / "artifacts/research_20261102/e925f89746af432b"
OUT = ROOT / "artifacts/historical_augmentation_20261102"


def historical_records(path, origin):
    records = defaultdict(Counter)
    with path.open() as handle:
        reader = csv.DictReader(handle)
        assert set(reader.fieldnames) == {"rent_approval_date", "town", "block", "street_name", "flat_type", "monthly_rent"}
        for row in reader:
            month = row["rent_approval_date"]
            if not "2021-01" <= month < origin:
                continue  # Do not inspect excluded periods' rental targets.
            key = (month, normal(row["town"]), normal(row["block"]),
                   normal(row["street_name"]), normal(row["flat_type"]))
            rent = float(row["monthly_rent"])
            if not np.isfinite(rent) or rent <= 0:
                raise ValueError("Invalid eligible source rent")
            records[key][rent] += 1
    return records


def eligible_additions(train, path, origin):
    given = defaultdict(Counter)
    templates = {}
    for row in train.to_dict(orient="records"):
        key = (row["RENT_APPROVAL_DATE"], normal(row["TOWN"]), normal(row["BLOCK"]),
               normal(row["STREET"]), normal(row["FLAT_TYPE"]))
        given[key][float(row["MONTHLY_RENT"])] += 1
        template_key = key[1:]
        if template_key in templates:
            for field in ["FLOOR_AREA_SQM", "FLAT_MODEL", "LEASE_COMMENCE_DATE"]:
                if templates[template_key][field] != row[field]:
                    raise ValueError("Nonunique course attribute template")
        templates[template_key] = row
    records = historical_records(path, origin)
    added, eligible = [], 0
    for key, rents in records.items():
        # Reject groups where source and course disagree on existing records.
        if given[key]-rents:
            continue
        for rent, count in (rents-given[key]).items():
            eligible += count
            if key[1:] not in templates:
                continue
            for _ in range(count):
                row = templates[key[1:]].copy()
                row.update(RENT_APPROVAL_DATE=key[0], MONTHLY_RENT=rent)
                added.append(row)
    frame = pd.DataFrame(added, columns=train.columns)
    assert frame.RENT_APPROVAL_DATE.lt(origin).all()
    return frame, dict(eligible_source_excess=eligible, matched_additions=len(frame),
                       course_rows=len(train), original_course_labels_unchanged=True)


def main():
    OUT.mkdir(exist_ok=True)
    source = pd.read_csv(ROOT / "data/raw/train.csv")
    train = source.loc[source.RENT_APPROVAL_DATE.lt("2025-01")].reset_index(drop=True)
    scoring = pd.read_csv(PARENT / "short_recent_scoring_rows.csv")
    query = source.iloc[scoring.source_row].drop(columns="MONTHLY_RENT").reset_index(drop=True)
    added, audit = eligible_additions(train, SOURCE, "2025-01")
    additions = OUT / "eligible_additions_before_202501.csv"
    added.to_csv(additions,index=False)
    protocol = dict(
        intervention="Append multiset-excess public rental records strictly before 2025-01 with unique course building/type attribute templates; retain all original labels",
        source_url="https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view",
        source_subset_sha256=hashlib.sha256(additions.read_bytes()).hexdigest(),
        caveat="Latest-vintage historical public records, not reconstructed historical availability. No validation/test-period rental targets used.",
        audit=audit, seed=42, iterations=1600, threads=8,
        comparison="Replace normal-history direct head only; other three reference heads frozen",
        gate="At least 2 RMSE improvement and address-cluster upper delta CI <0 before further confirmation; no automatic upload",
    )
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
    print(json.dumps(audit),flush=True)
    augmented = pd.concat([train, added],ignore_index=True)
    raw, target = augmented.drop(columns="MONTHLY_RENT"), augmented.MONTHLY_RENT
    feature_cache = OUT/"features.pkl"
    if feature_cache.exists():
        fit_x, query_x = pd.read_pickle(feature_cache)
    else:
        lookup = build_block_lookup(ROOT/"data/raw/auxiliary")
        print("Building augmented past-only history features",flush=True)
        fit_x = build_features(raw,target,raw,lookup,"local_trend")
        query_x = build_features(raw,target,query,lookup,"local_trend")
        pd.to_pickle((fit_x,query_x),feature_cache)
    assert len(fit_x)==len(target)
    config=load_config()
    model = CatBoostRegressor(iterations=1600,learning_rate=config["learning_rate"],
                              depth=config["depth"],l2_leaf_reg=config["l2_leaf_reg"],
                              loss_function="RMSE",random_seed=42,thread_count=8,
                              task_type="CPU",allow_writing_files=False,verbose=400)
    model.fit(fit_x,target,cat_features=fit_x.select_dtypes(include=["object","string","category"]).columns.tolist())
    model.save_model(str(OUT/"augmented_direct.cbm"))
    pred = model.predict(query_x)
    old_heads = [np.load(PARENT/f"short_recent_reference_{view}_{mode}_42.npy")
                 for view,mode in [("local_trend_0","residual"),("hierarchical_6","direct"),("hierarchical_6","residual")]]
    reference=np.load(PARENT/"short_recent_reference_ensemble.npy")
    candidates={"reference":reference,"augmented_standalone":pred,"augmented_replacement":np.mean(old_heads+[pred],axis=0)}
    y=scoring.target.to_numpy()
    keys=course_keys(query)
    groups,names=pd.factorize(keys.town+"|"+keys.block+"|"+keys.street)
    draws=np.random.default_rng(20261102).multinomial(len(names),np.full(len(names),1/len(names)),size=1000)
    mass=draws@np.bincount(groups)
    ref_sample=np.sqrt(np.einsum("ij,j->i",draws,np.bincount(groups,weights=(reference-y)**2),optimize=False)/mass)
    metrics=[]
    for name,p in candidates.items():
        assert np.isfinite(p).all()
        np.save(OUT/f"{name}.npy",p)
        delta=np.sqrt(np.einsum("ij,j->i",draws,np.bincount(groups,weights=(p-y)**2),optimize=False)/mass)-ref_sample
        metrics.append(dict(candidate=name,rmse=float(np.sqrt(np.mean((p-y)**2))),
                            delta_rmse=float(np.sqrt(np.mean((p-y)**2))-np.sqrt(np.mean((reference-y)**2))),
                            delta_ci_low=float(np.quantile(delta,.025)),delta_ci_high=float(np.quantile(delta,.975))))
    result=pd.DataFrame(metrics)
    result.to_csv(OUT/"metrics.csv",index=False)
    print(result.to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
