"""Export historical aggregate evidence; never train or look up targets.

The paper integrates selected findings into its original sections. The former
standalone chronological appendix and its three table fragments were retired.
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def save_csv(frame, name):
    frame.to_csv(REPORTS / name, index=False, lineterminator="\n")


def main():
    simplified_path = ROOT / "artifacts/legit_ceiling_20261102/results.csv"
    simplified = pd.read_csv(simplified_path)
    simplified["label_visibility"] = np.where(simplified.candidate.str.startswith("ORACLE"),
                                               "validation-target-visible diagnostic", "training labels only")
    simplified["source"] = str(simplified_path.relative_to(ROOT))
    save_csv(simplified, "november_simplified_20261102.csv")

    log_parent = ROOT / "artifacts/research_20261102/17bad5d96e4b3667"
    log_rows = []
    windows = {"short_recent": "Jan--Mar 2025", "long_recent": "Apr 2024--Mar 2025",
               "long_older": "Apr 2023--Mar 2024"}
    for window in windows:
        score = pd.read_csv(log_parent / f"{window}_scoring_rows.csv")
        y = score.target.to_numpy()
        ref = np.load(log_parent / f"{window}_reference_ensemble.npy")
        log = np.load(log_parent / f"{window}_cat_log_ensemble.npy")
        for method, pred in [("reference4", ref), ("log2", log),
                             ("equal6", (4*ref+2*log)/6), ("blend25", .75*ref+.25*log)]:
            log_rows.append(dict(window=window, candidate=method, rows=len(y),
                                 rmse=float(np.sqrt(np.mean((pred-y)**2))),
                                 bias=float((pred-y).mean()), source=str(log_parent.relative_to(ROOT))))
    log_frame = pd.DataFrame(log_rows)
    save_csv(log_frame, "november_log_integration_20261102.csv")

    initial = pd.read_csv(ROOT / "artifacts/research_20261102/e925f89746af432b/paired_comparison.csv")
    initial = initial.rename(columns={"delta_rmse_vs_reference": "delta_rmse"})
    initial["study"] = "initial_cpu"
    attributes = pd.read_csv(ROOT / "artifacts/attribute_study_20261102/paired_comparison.csv")
    attributes["study"] = "external_attributes"
    mixture = pd.read_csv(ROOT / "artifacts/low_rent_soft_probe_20261102/metrics.csv")
    mixture["study"] = "relative_low_rent_mixture"
    followup = pd.read_csv(ROOT / "reports/rmse_followup_20261102.csv")
    followup["study"] = "rmse_followup"
    combined = pd.concat([initial, attributes, mixture, followup], ignore_index=True)
    combined["window"] = "2025-01..2025-03"
    save_csv(combined, "november_cpu_comparisons_20261102.csv")


    detector = pd.read_csv(ROOT / "artifacts/attribute_study_20261102/low_rent_information/metrics.csv")
    save_csv(detector, "november_low_rent_detection_20261102.csv")
    attribution = pd.read_csv(ROOT / "artifacts/low_rent_soft_probe_20261102/reference_error_attribution.csv")
    save_csv(attribution, "november_error_attribution_20261102.csv")
    simplified_feature_count = 91
    metadata = dict(
        simplified_feature_count=simplified_feature_count,
        simplified_windows={w: int(simplified.loc[simplified.window.eq(w), "n"].iloc[0])
                            for w in ["jan_mar25", "gap16", "long_older"]},
        log_target="natural log(y), training-set Duan smearing; not log1p",
        original_reference="Selected eight-head GPU 440.7629; distinct from November four-head CPU 441.8429",
        no_established_irreducible_error_floor=True,
        no_other_team_method_inference=True,
        stopped_eight_head_stock_prototype="No submission produced; distinct from the completed four-head stock probe",
        submission_records=[json.loads((ROOT / f"artifacts/{name}/kaggle_result.json").read_text())
                            for name in ["stock_probe_full_20261102", "market_normalized_full_20261102"]],
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [simplified_path, log_parent / "metrics.csv", ROOT / "reports/kaggle_submissions.csv"]},
        exported_from_environment={name: version(name) for name in
                                   ["numpy", "pandas", "scikit-learn", "scipy", "catboost", "lightgbm", "xgboost", "pytest"]},
        validation_reuse="All windows are reused development samples; bootstrap intervals do not remove selection bias",
        withheld="No raw rows, targets, per-row predictions, source-matching submission, credentials or fitted model files are exported",
    )
    (REPORTS / "november_evidence_20261102.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("Exported historical aggregate evidence; no model training or appendix generation.")


if __name__ == "__main__":
    main()
