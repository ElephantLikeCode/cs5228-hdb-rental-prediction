"""Paired error analysis for the completed isolated validation experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(y, prediction):
    error = prediction - y
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(np.abs(error))), bias=float(error.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw")
    args = parser.parse_args()
    folder = args.run_folder
    rows = pd.read_csv(folder / "short_recent_scoring_rows.csv")
    source = pd.read_csv(args.raw_dir / "train.csv").iloc[rows.source_row].reset_index(drop=True)
    y = rows.target.to_numpy()
    assert np.array_equal(source.MONTHLY_RENT.to_numpy(), y)
    prediction = {}
    for candidate in ("reference", "lgbm", "lgbm_relative", "cat_raw"):
        path = folder / f"short_recent_{candidate}_ensemble.npy"
        if path.exists():
            prediction[candidate] = np.load(path)
    for candidate, directory in (("lgbm_sparse_categories", "sparse_categories_followup"),
                                 ("embedding_mlp", "embedding_mlp_followup")):
        path = folder / directory / "ensemble.npy"
        if path.exists():
            prediction[candidate] = np.load(path)
    reference = prediction["reference"]
    # One exploratory blend, fixed at 25%, not a weight search.
    for candidate, estimate in list(prediction.items()):
        if candidate != "reference":
            prediction[f"blend25_{candidate}"] = 0.75 * reference + 0.25 * estimate
    keys = (source.TOWN.astype(str) + "|" + source.BLOCK.astype(str)
            + "|" + source.STREET.str.lower().str.strip())
    group_id, unique = pd.factorize(keys)
    counts = np.bincount(group_id)
    rng = np.random.default_rng(20261102)
    bootstrap = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=1000)
    masses = bootstrap @ counts
    reference_square = np.bincount(group_id, weights=(reference-y)**2)
    assert (masses > 0).all() and np.isfinite(reference_square).all()
    # Avoid platform BLAS floating-status warnings for mixed int/float matmul.
    reference_rmse = np.sqrt(np.einsum("ij,j->i", bootstrap, reference_square,
                                      optimize=False) / masses)
    summary, bands = [], []
    labels = pd.cut(y, [-np.inf, 2500, 3000, 3500, 4000, np.inf]).astype(str)
    for candidate, estimate in prediction.items():
        assert estimate.shape == y.shape and np.isfinite(estimate).all()
        square = np.bincount(group_id, weights=(estimate-y)**2)
        differences = np.sqrt(np.einsum("ij,j->i", bootstrap, square,
                                        optimize=False) / masses) - reference_rmse
        assert np.isfinite(differences).all()
        low, high = np.quantile(differences, [0.025, 0.975])
        result = dict(candidate=candidate, **metrics(y, estimate),
                      delta_rmse_vs_reference=metrics(y, estimate)["rmse"] - metrics(y, reference)["rmse"],
                      delta_ci_low=low, delta_ci_high=high)
        summary.append(result)
        for label in sorted(set(labels)):
            mask = labels == label
            bands.append(dict(candidate=candidate, band=label, rows=int(mask.sum()),
                              **metrics(y[mask], estimate[mask]),
                              sse_share=float(np.sum((estimate[mask]-y[mask])**2) / np.sum((estimate-y)**2))))
    summary_frame = pd.DataFrame(summary).sort_values("rmse")
    summary_frame.to_csv(folder / "paired_comparison.csv", index=False)
    pd.DataFrame(bands).to_csv(folder / "rent_band_comparison.csv", index=False)
    (folder / "analysis_caveats.json").write_text(json.dumps(dict(
        window="2025-01 to 2025-03", rows=len(y), clusters=len(unique),
        bootstrap="1000 address-cluster resamples; delta positive means candidate worse",
        status="Exploratory reused development window. CI does not account for model selection, temporal shift, or leaderboard reuse.",
        blends="Single 25% candidate / 75% reference exploratory blend; not tuned or submitted",
    ), indent=2))
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
