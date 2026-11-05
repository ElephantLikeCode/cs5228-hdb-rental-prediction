"""A bounded relative-low-rent mixture probe, not a verified tenancy classifier."""

import argparse
import json
from pathlib import Path
from src.research.cache_inputs import add_cache_arguments, reference_run, housing_run, require_files

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score

from src.research.compare_models import encode_lgbm

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "artifacts/research_20261102/e925f89746af432b"
HOUSING = ROOT / "artifacts/attribute_study_20261102/6829709328a10a35"
CLASSIFIER = ROOT / "artifacts/attribute_study_20261102/low_rent_information"
OUT = ROOT / "artifacts/low_rent_soft_probe_20261102"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_cache_arguments(parser)
    parser.add_argument('--classifier-run-dir', type=Path)
    args = parser.parse_args()
    PARENT = reference_run(args, globals()['PARENT'])
    OUT = args.output_dir or globals()['OUT']
    HOUSING = housing_run(args, globals()['HOUSING'])
    CLASSIFIER = args.classifier_run_dir or globals()['CLASSIFIER']
    CLASSIFIER = require_files(CLASSIFIER, ['validation_probabilities.csv'], '--classifier-run-dir')
    OUT.mkdir(parents=True, exist_ok=True)
    fit_x, query_x, target, _, _ = pd.read_pickle(PARENT / "short_recent_local_trend_0.pkl")
    source = pd.read_csv(ROOT / "data/raw/train.csv")
    train = source.loc[source.RENT_APPROVAL_DATE.lt("2025-01")].reset_index(drop=True)
    scoring = pd.read_csv(PARENT / "short_recent_scoring_rows.csv")
    y = scoring.target.to_numpy()
    reference = np.load(PARENT / "short_recent_reference_ensemble.npy")
    train_level = fit_x.ADAPT_ADDRESS_FLAT_AREA_LEVEL.to_numpy(float)
    query_level = query_x.ADAPT_ADDRESS_FLAT_AREA_LEVEL.to_numpy(float)
    keep = train.RENT_APPROVAL_DATE.ge("2022-01").to_numpy()
    train_event = np.asarray(target) < .65 * train_level
    valid_event = y < .65 * query_level
    from src.research.hdb_attribute_features import property_features
    extra_a, extra_b = pd.read_pickle(HOUSING / "resale_features.pkl")
    prop = pd.read_csv(ROOT / "data/external/attribute_sources_20261102/property.csv")
    query = source.iloc[scoring.source_row].reset_index(drop=True)
    extra_a = extra_a.join(property_features(train, prop))
    extra_b = extra_b.join(property_features(query, prop))
    left, right, vocabulary = encode_lgbm(fit_x.join(extra_a).loc[keep], query_x.join(extra_b))
    parameters = dict(n_estimators=600, learning_rate=.04, num_leaves=15,
                      min_child_samples=100, reg_lambda=10, random_state=42,
                      n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
    protocol = dict(
        purpose="Test a relative-rent two-expert soft mixture with the already-fitted housing-information classifier",
        train="2022-01..2024-12; history features strictly past; same source as completed diagnostic classifiers",
        validation="2025-01..2025-03; reused development window, not independent confirmation",
        event="rent < 65% of strictly-past adaptive reference; does not identify room rentals",
        intervention="Two LightGBM ratio experts; squared reference-level weights preserve currency MSE objective. Fixed 25% blend with frozen four-head reference.",
        parameters=parameters, new_fits=2, no_external_rental_labels=True,
        gate="Advance only with >=2 RMSE improvement and address-cluster 95% upper delta <0; otherwise stop this probe.",
        exclusions="No validation routing by true event, no deleting scoring rows, no threshold sweep, no new Kaggle upload",
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    relative_target = np.asarray(target)[keep] / train_level[keep]
    sample_weight = (train_level[keep]/3000)**2
    event = train_event[keep]
    predictions = []
    for name, mask in [("regular", ~event), ("low", event)]:
        print(f"Fitting {name}: {mask.sum()} rows", flush=True)
        model = LGBMRegressor(**parameters)
        model.fit(left.loc[mask], relative_target[mask], sample_weight=sample_weight[mask])
        pred = model.predict(right) * query_level
        assert np.isfinite(pred).all()
        np.save(OUT / f"{name}.npy", pred)
        model.booster_.save_model(str(OUT / f"{name}.txt"))
        predictions.append(pred)
    probability = pd.read_csv(CLASSIFIER / "validation_probabilities.csv").plus_housing_information.to_numpy()
    mixed = (1-probability)*predictions[0]+probability*predictions[1]
    candidates = {"reference": reference, "relative_soft_mixture": mixed,
                  "mixture_blend25": .75*reference+.25*mixed}
    # Routing below is for error attribution only and never used for predictions.
    squared = (reference-y)**2
    attributions = []
    for name, mask in [("below_half_reference", y < .5*query_level),
                       ("below_65pct_reference", valid_event),
                       ("at_least_65pct_reference", ~valid_event),
                       ("below_1000", y < 1000)]:
        attributions.append(dict(subset=name, rows=int(mask.sum()), row_share=float(mask.mean()),
                                 squared_error_share=float(squared[mask].sum()/squared.sum()),
                                 rmse=float(np.sqrt(squared[mask].mean())),
                                 prediction_bias=float((reference[mask]-y[mask]).mean())))
    pd.DataFrame(attributions).to_csv(OUT / "reference_error_attribution.csv", index=False)
    from src.research.hdb_attribute_features import course_keys
    keys = course_keys(query)
    groups, names = pd.factorize(keys.town+"|"+keys.block+"|"+keys.street)
    bootstrap = np.random.default_rng(20261102).multinomial(
        len(names), np.full(len(names), 1/len(names)), size=1000)
    mass = bootstrap @ np.bincount(groups)
    reference_sample = np.sqrt(np.einsum("ij,j->i", bootstrap,
        np.bincount(groups, weights=squared), optimize=False)/mass)
    results = []
    for name, pred in candidates.items():
        np.save(OUT / f"{name}.npy", pred)
        delta = np.sqrt(np.einsum("ij,j->i", bootstrap,
            np.bincount(groups, weights=(pred-y)**2), optimize=False)/mass)-reference_sample
        results.append(dict(candidate=name, rmse=float(np.sqrt(np.mean((pred-y)**2))),
                            mae=float(np.abs(pred-y).mean()), bias=float((pred-y).mean()),
                            delta_rmse=float(np.sqrt(np.mean((pred-y)**2))-np.sqrt(squared.mean())),
                            delta_ci_low=float(np.quantile(delta,.025)), delta_ci_high=float(np.quantile(delta,.975))))
    metrics = pd.DataFrame(results)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    diagnostics = dict(event_auc=float(roc_auc_score(valid_event, probability)),
                       event_rate=float(valid_event.mean()),
                       top_decile_event_precision=float(valid_event[np.argsort(probability)[-int(np.ceil(.1*len(y))):]].mean()),
                       interpretation="Event probabilities are not verified room-rental probabilities")
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(diagnostics), flush=True)


if __name__ == "__main__":
    main()
