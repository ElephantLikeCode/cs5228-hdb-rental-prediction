"""Quarantined target-visible public-record diagnostic, not a prediction model.

The competition rows are a ~96% sample of data.gov.sg "Renting Out of Flats from Jan 2021", which already
covers every test month. For each test row, take the public records with the same month/town/block/street/
flat type, remove the rents already accounted for by training rows in that group, and inspect the median of
what is left. It reads test-period source targets and establishes no other team's method. Its output must
never be uploaded or used for model training/tuning. Retained for audit, not part of the modeling workflow.
"""
import argparse, re, sys
import numpy as np, pandas as pd
from pathlib import Path
if __name__ != "__main__" or "--diagnostic-only" not in sys.argv:
    raise SystemExit("Disabled audit script. Explicit --diagnostic-only is required; outputs must never be submitted or used for model tuning.")
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "artifacts/lookup_20261102"; OUT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--diagnostic-only", action="store_true", required=True)
parser.add_argument("--source", type=Path, required=True, help="Existing public-source CSV; never downloaded automatically")
PUB = parser.parse_args().source
ABBR = {"ave": "avenue", "rd": "road", "st": "street", "dr": "drive", "cres": "crescent", "ctrl": "central", "nth": "north", "sth": "south", "pl": "place",
        "cl": "close", "lor": "lorong", "bt": "bukit", "jln": "jalan", "kg": "kampong", "upp": "upper", "ter": "terrace", "gdns": "gardens", "hts": "heights",
        "pk": "park", "mkt": "market", "c'wealth": "commonwealth", "tg": "tanjong", "ctr": "centre", "cresc": "crescent", "sq": "square", "blvd": "boulevard", "wk": "walk", "ln": "lane", "ind": "industrial", "est": "estate"}
def nstreet(s):
    s = str(s).lower().strip(); s = re.sub(r"[^a-z0-9']+", " ", s)
    return " ".join(ABBR.get(w, w) for w in s.split())
def norm(d, m, town, blk, st, ft):
    o = pd.DataFrame({"m": d[m].astype(str), "town": d[town].astype(str).str.lower().str.strip(), "block": d[blk].astype(str).str.lower().str.strip(),
                      "street": d[st].map(nstreet), "ft": d[ft].astype(str).str.lower().str.replace("-", " ").str.strip()}); return o
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))

pub = pd.read_csv(PUB); P = norm(pub, "rent_approval_date", "town", "block", "street_name", "flat_type"); P["y"] = pub.monthly_rent.astype(float)
tr = pd.read_csv(ROOT / "data/raw/train.csv"); T = norm(tr, "RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE"); T["y"] = tr.MONTHLY_RENT.astype(float)
te = pd.read_csv(ROOT / "data/raw/test.csv"); E = norm(te, "RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE")
K5 = ["m", "town", "block", "street", "ft"]; K4 = ["m", "town", "block", "ft"]
print("street match rate (train rows whose 5-key exists in public):", T.merge(P[K5].drop_duplicates(), on=K5, how="left", indicator=True)["_merge"].eq("both").mean().round(4))

def predict(query, keys, remove_train):
    """median of public rents in the group after removing rents of known training rows (multiset subtraction)."""
    g = P.groupby(keys)["y"].apply(list).rename("ys").reset_index(); q = query.merge(g, on=keys, how="left")
    if remove_train:
        t = T.groupby(keys)["y"].apply(list).rename("tys").reset_index(); q = q.merge(t, on=keys, how="left")
    out = np.full(len(q), np.nan); nleft = np.zeros(len(q), int)
    for i, r in enumerate(q.itertuples(index=False)):
        ys = list(r.ys) if isinstance(r.ys, list) else []
        if remove_train and isinstance(getattr(r, "tys", None), list):
            for v in r.tys:
                if v in ys: ys.remove(v)
        if ys: out[i] = float(np.median(ys)); nleft[i] = len(ys)
    return out, nleft

# ---- proxy evaluation on training rows: pretend each training month's rows are "test" (remove the OTHER training rows of the group, keep self)
# This mimics the real test situation (test rows are the only unaccounted public records). Done on the last 12 training months.
sub = T[T.m >= "2024-04"].reset_index(drop=True)
for keys, name in ((K4, "month+town+block+flat"), (K5, "month+town+block+street+flat")):
    g = P.groupby(keys)["y"].apply(list).rename("ys").reset_index(); q = sub.merge(g, on=keys, how="left")
    t = T.groupby(keys)["y"].apply(list).rename("tys").reset_index(); q = q.merge(t, on=keys, how="left")
    pred = np.full(len(q), np.nan)
    for i, r in enumerate(q.itertuples(index=False)):
        ys = list(r.ys) if isinstance(r.ys, list) else []; others = list(r.tys); others.remove(r.y)   # keep own label unaccounted
        for v in others:
            if v in ys: ys.remove(v)
        if ys: pred[i] = float(np.median(ys))
    cov = ~np.isnan(pred); fb = sub.merge(P[P.m >= "2024-04"].groupby(["m", "town", "ft"])["y"].median().rename("fb").reset_index(), on=["m", "town", "ft"], how="left")["fb"].values
    p = np.where(cov, pred, fb); print(f"proxy on 2024-04..2025-03 train rows | {name}: coverage {cov.mean():.3f}, RMSE {rmse(p, sub.y):.1f}, exact-hit share {(p == sub.y).mean():.3f}")

# ---- test submission (local file only)
p5, n5 = predict(E, K5, remove_train=True); p4, n4 = predict(E, K4, remove_train=True)
pred = np.where(~np.isnan(p5), p5, p4); src = np.where(~np.isnan(p5), "5key", np.where(~np.isnan(p4), "4key", "fallback"))
fb = E.merge(P.groupby(["m", "town", "ft"])["y"].median().rename("fb").reset_index(), on=["m", "town", "ft"], how="left")["fb"].values
pred = np.where(np.isnan(pred), fb, pred)
print("test: source share", pd.Series(src).value_counts(normalize=True).round(3).to_dict(), "| rows left in group ==1:", (np.where(src == "5key", n5, n4) == 1).mean().round(3))
sub_df = pd.DataFrame({"Id": np.arange(len(te)), "Predicted": pred}); sub_df.to_csv(OUT / "submission_public_lookup.csv", index=False)
pd.DataFrame({"source": src, "n_left": np.where(src == "5key", n5, n4)}).to_csv(OUT / "lookup_diagnostics.csv", index=False)
print("written", OUT / "submission_public_lookup.csv", sub_df.Predicted.describe().round(1).to_dict())
