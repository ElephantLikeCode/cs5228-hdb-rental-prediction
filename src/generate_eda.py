"""Generate focused EDA figures and summary tables for the project reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "reports" / "figures"
TARGET = "MONTHLY_RENT"


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    train = pd.read_csv(RAW / "train.csv")
    train["DATE"] = pd.to_datetime(train["RENT_APPROVAL_DATE"], format="%Y-%m")
    train["LEASE_AGE"] = train["DATE"].dt.year - train["LEASE_COMMENCE_DATE"]

    plt.figure(figsize=(8, 4.5))
    sns.histplot(train[TARGET], bins=45, color="#176B87")
    plt.axvline(train[TARGET].median(), color="#C0392B", linestyle="--", label="Median")
    plt.xlabel("Monthly rent (SGD)")
    plt.title("Distribution of monthly HDB rent")
    plt.legend()
    save("eda_target_distribution.png")

    monthly = train.groupby("DATE")[TARGET].agg(["median", "mean", "count"]).reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(monthly["DATE"], monthly["median"], label="Median rent", color="#176B87", linewidth=2)
    ax1.plot(monthly["DATE"], monthly["mean"], label="Mean rent", color="#E67E22", linewidth=1.5)
    ax1.set_ylabel("Monthly rent (SGD)")
    ax1.set_title("Rent trend and monthly sample size")
    ax2 = ax1.twinx()
    ax2.fill_between(monthly["DATE"], monthly["count"], alpha=0.15, color="#5D6D7E")
    ax2.set_ylabel("Records")
    ax1.legend(loc="upper left")
    save("eda_monthly_trend.png")

    order = train.groupby("FLAT_TYPE")[TARGET].median().sort_values().index
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=train, x="FLAT_TYPE", y=TARGET, order=order, showfliers=False, color="#72A0C1")
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Flat type")
    plt.ylabel("Monthly rent (SGD)")
    plt.title("Rent distribution by flat type")
    save("eda_rent_by_flat_type.png")

    town = train.groupby("TOWN")[TARGET].agg(median="median", count="size").sort_values("median")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(town.index, town["median"], color="#4C956C")
    ax.set_xlabel("Median monthly rent (SGD)")
    ax.set_title("Median rent by town")
    for y, (_, row) in enumerate(town.iterrows()):
        ax.text(row["median"] + 15, y, f"n={int(row['count']):,}", va="center", fontsize=7)
    save("eda_rent_by_town.png")

    sample = train.sample(min(25000, len(train)), random_state=42)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=sample,
        x="FLOOR_AREA_SQM",
        y=TARGET,
        hue="FLAT_TYPE",
        alpha=0.25,
        s=12,
        linewidth=0,
    )
    plt.xlabel("Floor area (sqm)")
    plt.ylabel("Monthly rent (SGD)")
    plt.title("Rent, floor area, and flat type (sample of 25,000)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Flat type")
    save("eda_area_flat_type.png")

    summary = pd.DataFrame(
        {
            "metric": [
                "rows", "months", "target_mean", "target_median", "target_std",
                "exact_duplicate_rows", "constant_columns",
            ],
            "value": [
                len(train), train["DATE"].nunique(), train[TARGET].mean(),
                train[TARGET].median(), train[TARGET].std(), train.duplicated().sum(),
                ", ".join(c for c in train.columns if train[c].nunique(dropna=False) == 1),
            ],
        }
    )
    summary.to_csv(ROOT / "reports" / "eda_summary.csv", index=False)
    monthly.to_csv(ROOT / "reports" / "eda_monthly_summary.csv", index=False)
    town.reset_index().to_csv(ROOT / "reports" / "eda_town_summary.csv", index=False)
    print(f"Wrote {len(list(FIGURES.glob('eda_*.png')))} EDA figures to {FIGURES}")


if __name__ == "__main__":
    main()
