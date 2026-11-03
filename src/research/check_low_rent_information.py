"""Does the new housing information help identify low-rent events?"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.research.hdb_attribute_features import property_features
from src.research.compare_models import encode_lgbm

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "artifacts/research_20261102/e925f89746af432b"
HOUSING = ROOT / "artifacts/attribute_study_20261102/6829709328a10a35"
OUT = ROOT / "artifacts/attribute_study_20261102/low_rent_information"


def main():
    OUT.mkdir(exist_ok=True)
    fit_x, query_x, target, _, _ = pd.read_pickle(PARENT / "short_recent_local_trend_0.pkl")
    source = pd.read_csv(ROOT / "data/raw/train.csv")
    rows = pd.read_csv(PARENT / "short_recent_scoring_rows.csv")
    train = source.loc[source.RENT_APPROVAL_DATE.lt("2025-01")].reset_index(drop=True)
    query = source.iloc[rows.source_row].reset_index(drop=True)
    extra_a, extra_b = pd.read_pickle(HOUSING / "resale_features.pkl")
    stock = pd.read_csv(ROOT / "data/external/attribute_sources_20261102/property.csv")
    extra_a = extra_a.join(property_features(train, stock))
    extra_b = extra_b.join(property_features(query, stock))
    keep = train.RENT_APPROVAL_DATE.ge("2022-01").to_numpy()
    # Target describes an unusually low rent, not a verified bedroom rental.
    train_event = (np.asarray(target) < .65 * fit_x.ADAPT_ADDRESS_FLAT_AREA_LEVEL.to_numpy()).astype(int)
    query_event = (rows.target.to_numpy() < .65 * query_x.ADAPT_ADDRESS_FLAT_AREA_LEVEL.to_numpy()).astype(int)
    parameters = dict(n_estimators=400, learning_rate=.04, num_leaves=15,
                      min_child_samples=100, reg_lambda=10, random_state=42,
                      n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)
    protocol = dict(
        definition="actual rent < 0.65 * strictly-past adaptive address/type/area level; not a verified room/whole-flat label",
        training="2022-01..2024-12 (warmup excluded)", validation="2025-01..2025-03",
        candidates=["existing_inputs", "plus_housing_information"], parameters=parameters,
        information="Resale history origin-frozen; property stock is current snapshot, no historical-vintage claim",
        stopping="Two fixed diagnostic classifiers. No validation fitting, no use of probabilities to claim a tested rent-regression gain.",
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    predictions, results = {}, []
    for name, a, b in [("existing_inputs", fit_x, query_x),
                       ("plus_housing_information", fit_x.join(extra_a), query_x.join(extra_b))]:
        print(f"Fitting {name}", flush=True)
        left, right, vocabulary = encode_lgbm(a.loc[keep], b)
        model = LGBMClassifier(**parameters)
        model.fit(left, train_event[keep])
        probability = model.predict_proba(right)[:, 1]
        predictions[name] = probability
        results.append(dict(candidate=name, event_rate=float(query_event.mean()),
                            roc_auc=roc_auc_score(query_event, probability),
                            average_precision=average_precision_score(query_event, probability),
                            brier=brier_score_loss(query_event, probability)))
        model.booster_.save_model(str(OUT / f"{name}.txt"))
        (OUT / f"{name}_vocabulary.json").write_text(json.dumps(vocabulary))
    frame = pd.DataFrame(predictions)
    frame.insert(0, "event", query_event)
    frame.to_csv(OUT / "validation_probabilities.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "metrics.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
