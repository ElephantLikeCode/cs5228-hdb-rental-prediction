"""Non-rental-label enrichment and evidence about the course attribute join."""

import re

import numpy as np
import pandas as pd

ABBREVIATIONS = {
    "ave": "avenue", "rd": "road", "st": "street", "dr": "drive",
    "cres": "crescent", "ctrl": "central", "nth": "north", "sth": "south",
    "pl": "place", "cl": "close", "lor": "lorong", "bt": "bukit",
    "jln": "jalan", "kg": "kampong", "upp": "upper", "ter": "terrace",
    "gdns": "gardens", "hts": "heights", "pk": "park", "mkt": "market",
    "c'wealth": "commonwealth", "tg": "tanjong", "ctr": "centre",
    "sq": "square", "blvd": "boulevard", "wk": "walk", "ln": "lane",
    "ind": "industrial", "est": "estate",
}


def normal(value):
    words = re.sub(r"[^a-z0-9']+", " ", str(value).lower()).split()
    return " ".join(ABBREVIATIONS.get(word, word) for word in words)


def course_keys(frame):
    result = pd.DataFrame(index=np.arange(len(frame)))
    result["town"] = frame.TOWN.map(normal).to_numpy()
    result["block"] = frame.BLOCK.map(normal).to_numpy()
    result["street"] = frame.STREET.map(normal).to_numpy()
    result["type"] = frame.FLAT_TYPE.map(normal).to_numpy()
    return result


def prepare_resale(source):
    result = source.copy()
    for column in ("town", "block", "street_name", "flat_type", "flat_model"):
        result[column] = result[column].map(normal)
    result = result.rename(columns={"street_name": "street", "flat_type": "type"})
    for column in ("floor_area_sqm", "lease_commence_date", "resale_price"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    result["psm"] = result.resale_price / result.floor_area_sqm
    numbers = result.storey_range.str.findall(r"\d+")
    result["storey"] = numbers.map(lambda values: np.mean([float(v) for v in values]))
    result["high_floor"] = result.storey.ge(10).astype(float)
    result["period"] = pd.PeriodIndex(result.month, freq="M")
    return result


def provenance_audit(raw, sales):
    keys = ["town", "block", "street", "type"]
    listings = course_keys(raw)
    listings["area"] = raw.FLOOR_AREA_SQM.to_numpy()
    listings["model"] = raw.FLAT_MODEL.map(normal).to_numpy()
    listings["lease"] = raw.LEASE_COMMENCE_DATE.to_numpy()
    grouped = listings.groupby(keys).agg(rows=("area", "size"), areas=("area", "nunique"),
                                        models=("model", "nunique"), leases=("lease", "nunique"))
    templates = listings.drop_duplicates(keys)
    renamed = sales.rename(columns={"floor_area_sqm": "area", "flat_model": "model",
                                    "lease_commence_date": "lease"})
    tuples = renamed[keys+["area", "model", "lease"]].drop_duplicates()
    matched = templates.merge(tuples.assign(tuple_found=1), how="left",
                              on=keys+["area", "model", "lease"], validate="one_to_one")
    support = sales.groupby(keys).size().rename("resale_count").reset_index()
    matched = matched.merge(support, on=keys, how="left", validate="one_to_one")
    summary = dict(
        groups=len(grouped), repeated_groups=int(grouped.rows.ge(2).sum()),
        repeated_groups_with_constant_area_model_lease=int(
            (grouped.rows.ge(2)&grouped.areas.eq(1)&grouped.models.eq(1)&grouped.leases.eq(1)).sum()),
        groups_with_2017_2024_resales=int(matched.resale_count.notna().sum()),
        template_tuple_present_in_resales=int(matched.tuple_found.eq(1).sum()),
        warning="A matching tuple is evidence of compatibility, not proof of the instructor's source or aggregation rule. Source lacks a rental-unit identifier.",
    )
    return summary, matched


def resale_features(raw, sales, origin=None):
    """24-month trailing sales, with two-month lag and optional origin freeze."""
    keys = course_keys(raw)
    periods = pd.PeriodIndex(raw.RENT_APPROVAL_DATE, freq="M")
    output = pd.DataFrame(index=np.arange(len(raw)), dtype=float)
    names = ["log_count", "area_mean", "area_std", "area_q25", "area_q75",
             "psm_median", "psm_std", "storey_mean", "storey_std", "high_floor_share",
             "lease_min", "lease_max", "model_count", "template_area_z", "value_proxy",
             "sale_age_months", "missing"]
    for name in names:
        output["EXT_SALE_"+name] = np.nan
    columns = ["town", "block", "street", "type"]
    for month in sorted(periods.unique()):
        ids = np.flatnonzero(periods == month)
        cutoff = month-2
        if origin is not None:
            cutoff = min(cutoff, pd.Period(origin, freq="M")-2)
        reference = sales.loc[sales.period.le(cutoff)&sales.period.ge(cutoff-23)]
        stats = reference.groupby(columns).agg(
            count=("psm", "size"), area_mean=("floor_area_sqm", "mean"),
            area_std=("floor_area_sqm", "std"),
            psm_median=("psm", "median"), psm_std=("psm", "std"),
            storey_mean=("storey", "mean"), storey_std=("storey", "std"),
            high_floor_share=("high_floor", "mean"),
            lease_min=("lease_commence_date", "min"), lease_max=("lease_commence_date", "max"),
            model_count=("flat_model", "nunique"), last=("period", "max"))
        quantiles = reference.groupby(columns).floor_area_sqm.quantile([.25, .75]).unstack()
        quantiles.columns = ["area_q25", "area_q75"]
        stats = stats.join(quantiles)
        joined = keys.iloc[ids].join(stats, on=columns)
        for name in names:
            if name in joined:
                output.loc[ids, "EXT_SALE_"+name] = joined[name].to_numpy()
        output.loc[ids, "EXT_SALE_log_count"] = np.log1p(joined["count"].fillna(0)).to_numpy()
        area = raw.iloc[ids].FLOOR_AREA_SQM.to_numpy(float)
        output.loc[ids, "EXT_SALE_template_area_z"] = (
            (area-joined.area_mean.to_numpy())/joined.area_std.clip(lower=1).to_numpy())
        output.loc[ids, "EXT_SALE_value_proxy"] = joined.psm_median.to_numpy()*area
        output.loc[ids, "EXT_SALE_missing"] = joined["count"].isna().to_numpy(float)
        output.loc[ids, "EXT_SALE_sale_age_months"] = joined["last"].map(
            lambda value: month.ordinal-value.ordinal if pd.notna(value) else np.nan).to_numpy()
    return output


def property_features(raw, source):
    """Current building-stock snapshot; no claim of historical-vintage availability."""
    table = source.copy()
    table["block"] = table.blk_no.map(normal)
    table["street"] = table.street.map(normal)
    unique = ~table.duplicated(["block", "street"], keep=False)
    numeric = ["total_dwelling_units", "1room_sold", "2room_sold", "3room_sold", "4room_sold",
               "5room_sold", "exec_sold", "multigen_sold", "studio_apartment_sold",
               "1room_rental", "2room_rental", "3room_rental", "other_room_rental"]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    fit = table.loc[unique].set_index(["block", "street"])
    keys = course_keys(raw)
    joined = keys.join(fit[numeric+["commercial", "market_hawker"]], on=["block", "street"])
    result = pd.DataFrame(index=keys.index)
    denominator = joined.total_dwelling_units.clip(lower=1)
    result["EXT_STOCK_LOG_UNITS"] = np.log1p(joined.total_dwelling_units)
    for column in numeric[1:]:
        result["EXT_STOCK_SHARE_"+column] = joined[column]/denominator
    result["EXT_STOCK_COMMERCIAL"] = joined.commercial.map({"Y": 1., "N": 0.})
    result["EXT_STOCK_HAWKER"] = joined.market_hawker.map({"Y": 1., "N": 0.})
    result["EXT_STOCK_MISSING"] = joined.total_dwelling_units.isna().astype(float)
    return result


def supplied_current_macro(table, raw, origin=None):
    """Contemporaneous supplied covariates, explicitly not a month-start forecast."""
    months = pd.PeriodIndex(raw.RENT_APPROVAL_DATE, freq="M")
    if origin is not None:
        last = pd.Period(origin, freq="M")-1
        months = pd.PeriodIndex([min(m, last) for m in months])
    result = table.reindex(months).reset_index(drop=True)
    result = result.rename(columns=lambda name: "GIVEN_"+name)
    result["GIVEN_MISSING"] = result.isna().mean(axis=1)
    return result.fillna(0)
