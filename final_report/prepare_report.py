"""Regenerate report figures and tables from existing evidence; never fit models.

Run from the project root: python final_report/prepare_report.py
The full path requires course data and the retained temporal prediction caches.
The committed figures/tables let readers compile the paper without those caches.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager, ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import normalize_text
from feedback_experiments import frozen_baseline

MAIN = "local_trend_history_diversity_blend"
COLORS = ["#356AA0", "#D58A36", "#258E83", "#8A6BA8", "#667785"]


def model_overview():
    """Vector schematic; counts and views follow the frozen selected recipe."""
    config = json.loads((ROOT/'configs/model.json').read_text())
    assert config['seeds'] == [42,2026] and set(config['target_modes']) == {False,True}
    assert config['history_views'] == [dict(history='local_trend',lag=0),dict(history='hierarchical',lag=6)]
    metadata = json.loads((ROOT/'reports/final_model_metadata.json').read_text())
    assert metadata['feature_counts'] == {'local_trend:0':78,'hierarchical:6':74}
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.005)
    ax.set(xlim=(0, 10), ylim=(0, 3.1))
    ax.axis('off')
    ink = '#263746'

    def box(x, y, w, h, text, fill, edge=ink, size=8, bold=False):
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.025,rounding_size=0.055',
                                   facecolor=fill, edgecolor=edge, linewidth=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,
                color=ink,weight='bold' if bold else 'normal',linespacing=1.22)

    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,
                                    linewidth=.8,color='#667785',zorder=0))

    for x, title in [(0.1,'1  INPUTS'),(2.12,'2  HISTORY VIEWS'),(4.78,'3  CATBOOST COMPONENTS'),(8.15,'4  OUTPUT')]:
        ax.text(x,2.93,title,fontsize=8,color=ink,weight='bold',va='center')
    box(.1,1.62,1.63,.81,'Flat attributes\n+ spatial context', '#EEF2F6')
    box(.1,.62,1.63,.81,'Permitted earlier\ntraining rents', '#EEF2F6')
    box(2.12,1.62,2.13,.81,'Normal history\n+ local trends\n78 features', '#EAF1FA', COLORS[0])
    box(2.12,.62,2.13,.81,'Original + 6-month\nstale-history augmentation\n74 features', '#E7F3F0',COLORS[2])
    for y, color, fill in [(1.62,COLORS[0],'#EAF1FA'),(.62,COLORS[2],'#E7F3F0')]:
        box(4.78,y,1.38,.81,'Direct target\nseeds 42 / 2026\n2 models',fill,color,7.6)
        box(6.34,y,1.38,.81,'Residual target\nseeds 42 / 2026\n2 models',fill,color,7.6)
        arrow((4.27,y+.405),(4.75,y+.405))
        ax.plot([4.49,4.49,7.03],[y+.405,y+.92,y+.92],color='#667785',linewidth=.8,zorder=0)
        arrow((7.03,y+.92),(7.03,y+.83))
        ax.plot([5.47,5.47,7.92],[y,y-.12,y-.12],color='#667785',linewidth=.8,zorder=0)
        arrow((7.92,y-.12),(8.13,1.53))
        arrow((7.74,y+.405),(8.13,1.53))
    box(8.16,1.08,1.7,.9,'Equal mean\n8 predictions\nweight = 1/8', '#FFF1DF',COLORS[1],8,True)
    ax.text(9.01,.74,'Monthly rent (SGD)',ha='center',fontsize=8,color=ink)
    for sy in [2.02,1.02]:
        for ty in [2.02,1.02]:
            arrow((1.76,sy),(2.09,ty))
    ax.text(5,.18,'TRAIN: original transaction weight is preserved.   PREDICT: both views use all permitted earlier rents.',
            fontsize=7.5,ha='center',color='#52616E')
    fig.savefig(HERE/'figures/model_overview.pdf')
    plt.close(fig)


def validation_overview():
    fig, ax = plt.subplots(figsize=(3.4,1.65),constrained_layout=True)
    intervals = [('Recent quarter','2024-12','2025-03'),
                 ('Recent year','2024-03','2025-03'),
                 ('Older year','2023-03','2024-03'),
                 ('16 months','2023-11','2025-03')]
    start = pd.Timestamp('2021-01-01')
    for row,(label,cut,end) in enumerate(intervals):
        cutoff = pd.Timestamp(cut)+pd.offsets.MonthBegin(1)
        finish = pd.Timestamp(end)+pd.offsets.MonthBegin(1)
        ax.barh(row,(cutoff-start).days,left=mdates.date2num(start),height=.49,color=COLORS[0],
                label='Fit on earlier months' if row==0 else None)
        ax.barh(row,(finish-cutoff).days,left=mdates.date2num(cutoff),height=.49,color=COLORS[1],
                label='Frozen validation' if row==0 else None,hatch='///',edgecolor='white',linewidth=.3)
    ax.set_yticks(range(4),[r[0] for r in intervals])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(start,pd.Timestamp('2025-04-01'))
    ax.legend(loc='lower left',bbox_to_anchor=(-.32,1.01),frameon=False,ncol=2,fontsize=7)
    ax.tick_params(axis='both',length=0)
    ax.spines[['left','bottom']].set_visible(False)
    fig.savefig(HERE/'figures/validation_windows.pdf')
    plt.close(fig)


def tex_table(name, header, rows, alignment):
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (HERE / "generated" / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def signed_bias(value):
    """Format report bias to one decimal without a misleading signed zero."""
    return "0.0" if round(float(value), 1) == 0 else f"{value:+.1f}"


def main():
    for directory in ("figures", "generated"):
        (HERE / directory).mkdir(exist_ok=True)
    # Follow the course template at final publication size; never silently fall back.
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42})
    model_overview()
    validation_overview()
    raw = pd.read_csv(ROOT / "data/raw/train.csv")
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    summary = pd.read_csv(ROOT / "reports/temporal_summary.csv")
    comparisons = pd.read_csv(ROOT / "reports/temporal_comparison.csv")
    feedback = pd.read_csv(ROOT / "reports/feedback_results.csv")
    y = raw.MONTHLY_RENT
    ids, prediction, _, _ = frozen_baseline("short_recent", raw, y, return_components=True)
    q = raw.iloc[ids].reset_index(drop=True).copy()
    q["prediction"] = prediction
    q["error"] = prediction - q.MONTHLY_RENT
    q["squared_error"] = q.error**2
    q["flat"] = normalize_text(q.FLAT_TYPE)
    q["band"] = pd.cut(q.MONTHLY_RENT, [0, 2500, 3000, 3500, 4000, np.inf],
        labels=["<=2500", "2501-3000", "3001-3500", "3501-4000", ">4000"])
    groups = q.groupby("band", observed=True).agg(rows=("error", "size"),
        mse=("squared_error", "mean"), sse=("squared_error", "sum"), bias=("error", "mean"))
    groups["rmse"] = np.sqrt(groups.mse)
    groups["row_percent"] = groups.rows / len(q) * 100
    groups["sse_percent"] = groups.sse / q.squared_error.sum() * 100
    groups.reset_index().to_csv(HERE / "generated/rent_band_evidence.csv", index=False)
    tex_table("rent_bands", ["Actual rent", "$n$", "RMSE", "Bias", "SSE (\\%)"],
        [[str(k).replace("<=", r"$\leq$").replace(">", r"$>$"), f"{int(v.rows):,}", f"{v.rmse:.1f}",
          signed_bias(v.bias), f"{v.sse_percent:.1f}"] for k,v in groups.iterrows()], "lrrrr")

    # Observable groups complement the outcome-defined bands; no label-based routing.
    flat_groups = q.groupby("flat").agg(rows=("error", "size"),
        mse=("squared_error", "mean"), sse=("squared_error", "sum"), bias=("error", "mean"))
    flat_groups["rmse"] = np.sqrt(flat_groups.mse)
    flat_groups["sse_percent"] = flat_groups.sse / q.squared_error.sum() * 100
    flat_groups.reset_index().to_csv(HERE / "generated/flat_type_evidence.csv", index=False)
    tex_table("flat_types", ["Observed flat type", "$n$", "RMSE", "Bias", "SSE (\\%)"],
        [[k.replace(" ", "-"), f"{int(v.rows):,}", f"{v.rmse:.1f}", signed_bias(v.bias),
          f"{v.sse_percent:.1f}"] for k,v in flat_groups.iterrows()], "lrrrr")

    # Descriptive training data, not an estimate of the effect of calendar time.
    a = raw.assign(month=pd.to_datetime(raw.RENT_APPROVAL_DATE), flat=normalize_text(raw.FLAT_TYPE))
    trend = a.groupby(["month", "flat"]).MONTHLY_RENT.median().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), constrained_layout=True)
    for i, flat in enumerate(["2 room", "3 room", "4 room", "5 room", "executive"]):
        axes[0].plot(trend.index, trend[flat], label=flat.replace(" ", "-"),
                     color=COLORS[i], linewidth=1.35, linestyle=["-", "--", "-.", ":", "-"][i])
    axes[0].set(xlabel="Approval month", ylabel="Median monthly rent (SGD)")
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=5, loc="outside upper center", frameon=False)
    axes[1].hist(y, bins=np.arange(250, 8001, 250), color=COLORS[0], edgecolor="white", linewidth=.35)
    axes[1].set(xlabel="Monthly rent (SGD)", ylabel="Training transactions")
    axes[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    for ax, label in zip(axes, ("(a)", "(b)")):
        ax.set_title(label, loc="left", fontsize=8)
    fig.savefig(HERE / "figures/data_profile.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.15), constrained_layout=True)
    x = np.arange(len(groups)); width=.34
    ax.bar(x-width/2, groups.row_percent, width, label="Share of rows", color=COLORS[0])
    bars = ax.bar(x+width/2, groups.sse_percent, width, label="Share of squared error", color=COLORS[1], hatch="//",edgecolor='white',linewidth=.4)
    ax.annotate(f'{groups.sse_percent.iloc[0]:.1f}%', (bars[0].get_x()+width/2,groups.sse_percent.iloc[0]),
                xytext=(0,4),textcoords='offset points',ha='center',fontsize=8,color=COLORS[1],weight='bold')
    ax.set_xticks(x, ["\u2264 2500", "2501-3000", "3001-3500", "3501-4000", ">4000"], rotation=20)
    ax.set(xlabel="Actual monthly rent band (SGD)", ylabel="Share (%)")
    ax.legend(frameon=False,loc='upper right',fontsize=7)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True,color='#E7ECF1',linewidth=.5)
    ax.set_ylim(0,62)
    fig.savefig(HERE / "figures/error_concentration.pdf")
    plt.close(fig)

    candidates = ["hierarchical_market_blend", "history_diversity_blend", MAIN]
    labels = ["Adaptive market (4)", "History diversity (8)", "Local trend (8)"]
    windows = ["short_recent", "long_recent", "long_older", "horizon16"]
    window_labels = ["Jan-Mar 2025", "Apr 2024-Mar 2025", "Apr 2023-Mar 2024", "Dec 2023-Mar 2025"]
    rows = []
    for window, label in zip(windows, window_labels):
        z = summary[summary.window.eq(window)].set_index("candidate")
        # Obtain cutoffs from the executed fits rather than infer them from labels.
        cutoffs = comparisons.loc[comparisons.window.eq(window), "train_end"].unique()
        if len(cutoffs) != 1:
            raise ValueError(f"Ambiguous training cutoff for {window}: {cutoffs}")
        rows.append([label, cutoffs[0], f"{int(z.loc[MAIN,'rows']):,}",
                     *[f"{z.loc[c,'rmse']:.2f}" for c in candidates]])
    tex_table("temporal", ["Validation interval", "Train through", "$n$", "Adaptive (4)",
                           "Diversity (8)", "Local trend (8)"], rows, "llrrrr")
    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)
    for i, (candidate,label) in enumerate(zip(candidates[1:],labels[1:])):
        gains = []
        for window in windows:
            z=summary[summary.window.eq(window)].set_index("candidate")
            gains.append(z.loc[candidates[0], "rmse"]-z.loc[candidate,"rmse"])
        bars = ax.bar(np.arange(4) + (i-.5)*.34, gains, width=.34, label=label,
               color=COLORS[i], hatch=["", "//"][i],edgecolor='white',linewidth=.4)
        if candidate == MAIN:
            ax.bar_label(bars,labels=[f'{g:+.2f}' for g in gains],fontsize=7,padding=3,color=COLORS[1])
    ax.axhline(0, color="0.6", linewidth=.7)
    ax.set_xticks(range(4), ["Recent\nquarter", "Recent\nyear", "Older\nyear", "16-month\nwindow"])
    ax.set(ylabel="RMSE improvement (SGD)", ylim=(-12, 8))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True,color='#E7ECF1',linewidth=.5)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(HERE / "figures/temporal_tradeoff.pdf")
    plt.close(fig)

    baseline = pd.read_csv(ROOT / "reports/experiment_results.csv")
    base = baseline[baseline.fold.eq("2025-01_to_2025-03") &
                    (baseline.feature_set.eq("core") | baseline.feature_set.eq("raw"))]
    model_labels={"global_median":"Global median", "town_flat_type_median":"Town/type median",
        "linear_regression":"Linear regression", "ridge":"Ridge", "decision_tree":"Decision tree",
        "random_forest":"Random forest", "hist_gradient_boosting":"Histogram boosting", "catboost":"CatBoost (core)"}
    selected=[]
    for model,label in model_labels.items():
        z=base[base.model.eq(model)]
        if model=="catboost": z=z[z.protocol.eq("latest_spatial_ablation")]
        if len(z)!=1: raise ValueError(f"Ambiguous historical baseline: {model}")
        v=z.iloc[0]; selected.append([label, f"{v.rmse:.2f}", f"{v.mae:.2f}"])
    tex_table("baselines",["Historical core model", "RMSE", "MAE"],selected,"lrr")

    variants=[("d0e607697ec39679","nearest_flat_blend25","Nearest-comparable blend"),
        ("d0e607697ec39679","flat_specialists_blend25","Flat-type specialist blend"),
        ("d0e607697ec39679","rent_mixture_blend25","Rent-band mixture blend"),
        ("02e104f0ffdf57c5","crossfit_distribution_replace_seed","Cross-fitted statistics"),
        ("02e104f0ffdf57c5","crossfit_past_distribution_replace_seed","Strict-past distribution"),
        ("8050c7a295047db8","market_coe_replace_seed","COE indicators"),
        ("8050c7a295047db8","market_stock_replace_seed","Stock indicators"),
        ("8050c7a295047db8","learner_xgb_original_replace_seed","Matched-input XGBoost"),
        ("8050c7a295047db8","correction_residual_shrink25","OOF residual correction"),
        ("e3bf7c874aee98da","learner_ridge_guarded_origin_only_blend25","Guarded Ridge blend"),
        ("878fed7edebe94c5","public_cpi_replace_seed","CPI"),
        ("878fed7edebe94c5","public_ura_replace_seed","URA rental index"),
        ("878fed7edebe94c5","public_rail_static_replace_seed","Corrected static MRT"),
        ("878fed7edebe94c5","public_rail_events_replace_seed","MRT opening exposure"),
        ("f0d15ed2879e3f3b","public_macro_both_monthly_replace_seed","Monthly-available CPI+URA")]
    reference = summary[summary.candidate.eq(MAIN)].set_index("window")
    rows=[["Selected model", "--", *[f"{reference.loc[w,'rmse']:.3f}" for w in ("short_recent","horizon16")]]]
    for run,method,label in variants:
        z=feedback[feedback.run_id.eq(run)&feedback.method.eq(method)].set_index("window")
        mode = "B" if "blend25" in method else "C" if "shrink25" in method else "R"
        rows.append([label,mode,*[f"{z.loc[w,'rmse']:.3f}" for w in ("short_recent","horizon16")]])
    tex_table("research",["Candidate", "Use", "Recent", "16-month"],rows,"llrr")

    facts={"raw_sha256":{n:hashlib.sha256((ROOT/'data/raw'/f'{n}.csv').read_bytes()).hexdigest() for n in ("train","test")},
        "train_rows":len(raw),"test_rows":len(test),"train_start":raw.RENT_APPROVAL_DATE.min(),
        "train_end":raw.RENT_APPROVAL_DATE.max(),"test_start":test.RENT_APPROVAL_DATE.min(),
        "test_end":test.RENT_APPROVAL_DATE.max(),"recent_rmse":float(np.sqrt(np.mean(q.error**2))),
        "recent_mae":float(np.mean(abs(q.error))),"recent_bias":float(q.error.mean()),
        "recent_n":len(q),"observed_low_band_sse_percent":float(groups.iloc[0].sse_percent),
        "paper_source_tables":["experiment_results.csv","temporal_summary.csv","temporal_comparison.csv",
                               "feedback_results.csv","kaggle_submissions.csv"],
        "mainline_prediction_sources":pd.read_csv(ROOT/'reports/stacking_sources.csv').query("window == 'short_recent'").run_id.tolist(),
        "selected_vs_adaptive_gain_intervals":summary.loc[
            summary.candidate.eq(MAIN) & summary.window.isin(windows),
            ["window", "baseline_candidate", "rmse_improvement", "address_bootstrap_replicates",
             "gain_ci_lower", "gain_ci_upper"]].to_dict("records")}
    if not np.isclose(facts["recent_rmse"],440.7628634143297,atol=1e-7): raise ValueError("Mainline changed")
    (HERE / "generated/evidence.json").write_text(json.dumps(facts,indent=2)+'\n',encoding="utf-8")
    print(json.dumps(facts,indent=2))


if __name__ == "__main__":
    main()
