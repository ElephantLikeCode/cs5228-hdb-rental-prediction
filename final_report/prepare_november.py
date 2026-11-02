"""Export November aggregate evidence and appendix tables; never train or look up targets."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
GENERATED = ROOT / "final_report/generated"


def save_csv(frame, name):
    frame.to_csv(REPORTS / name, index=False, lineterminator="\n")


def table(name, header, rows, alignment):
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (GENERATED / name).write_text("\n".join(lines))


def main():
    simplified_path = ROOT / "artifacts/legit_ceiling_20261102/results.csv"
    simplified = pd.read_csv(simplified_path)
    simplified["label_visibility"] = np.where(simplified.candidate.str.startswith("ORACLE"),
                                               "validation-target-visible diagnostic", "training labels only")
    simplified["source"] = str(simplified_path.relative_to(ROOT))
    save_csv(simplified, "november_simplified_20261102.csv")
    names = {
        "ORACLE_same_month_town_ft_model_area10_median": "Target-visible group median$^{*}$",
        "base_town_ft_recent_median": "Town/type median",
        "base_comparable_address_flat": "Address/type comparables",
        "cat_direct": "CatBoost direct", "cat_log": "CatBoost log target",
        "cat_recent_w": "CatBoost recency weights", "cat_clean": "CatBoost low-rent removal",
        "cat_resid": "CatBoost median residual", "lgbm_direct": "LightGBM", "xgb_direct": "XGBoost",
        "blend_direct_resid": "Direct/residual mean", "blend_cat3_lgbm": "Three CatBoost + LightGBM",
        "blend_all_trees": "Six-tree-model mean",
    }
    pivot = simplified.pivot(index="candidate", columns="window", values="rmse")
    table("november_simplified.tex", ["Candidate", "Recent", "16 mo.", "Older"],
          [[label, *[f"{pivot.loc[key, window]:.2f}" for window in ["jan_mar25", "gap16", "long_older"]]]
           for key, label in names.items()], "lrrr")

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
    p = log_frame.pivot(index="window", columns="candidate", values="rmse")
    table("november_log.tex", ["Window", "Ref. 4", "Log 2", "Mean 6", "Blend"],
          [[label, *[f"{p.loc[key, m]:.2f}" for m in ["reference4", "log2", "equal6", "blend25"]]]
           for key, label in windows.items()], "lrrrr")

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

    selected = [
        ("initial_cpu", "cat_raw", "Raw category/area signatures"),
        ("initial_cpu", "lgbm", "Same-input LightGBM ensemble"),
        ("initial_cpu", "lgbm_relative", "Relative-target LightGBM"),
        ("initial_cpu", "embedding_mlp", "Embedding MLP"),
        ("external_attributes", "resale", "17 lagged resale features"),
        ("external_attributes", "stock_snapshot", "16 building-stock features"),
        ("external_attributes", "given_macro", "19 current supplied macro features"),
        ("relative_low_rent_mixture", "mixture_blend25", "Relative-low-rent mixture (25\\%)"),
        ("rmse_followup", "duplicate_mean_native_replacement", "Duplicate-mean replacement"),
        ("rmse_followup", "market_normalized_damped6_replacement", "Market-normalized replacement"),
        ("rmse_followup", "augmented_replacement", "4,357 additional historical records"),
        ("rmse_followup", "depth10_l2_20_replacement", "Depth 10, regularization 20"),
        ("rmse_followup", "lossguide64_l2_20_replacement", "Asymmetric-tree replacement"),
    ]
    rows = []
    for study, candidate, label in selected:
        r = combined.loc[combined.study.eq(study) & combined.candidate.eq(candidate)].iloc[0]
        rows.append([label, f"{r.rmse:.3f}", f"{r.delta_rmse:+.3f}"])
    table("november_cpu.tex", ["Candidate/composition", "RMSE", "$\\Delta$"], rows, "lrr")

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
    print("Exported November aggregate evidence and three report tables; no model training.")


if __name__ == "__main__":
    main()
