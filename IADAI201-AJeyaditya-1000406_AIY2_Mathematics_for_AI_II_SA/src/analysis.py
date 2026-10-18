"""
Exploratory data analysis helpers (groupby / agg / pivot_table / statistics).

Every function takes the *clean* DataFrame produced by src.processing and
returns a small DataFrame or dict - no plotting and no Streamlit here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .processing import LONG_STAY_DAYS


# --------------------------------------------------------------------------- #
# Headline numbers
# --------------------------------------------------------------------------- #
def kpis(df: pd.DataFrame) -> dict:
    """Headline KPIs for the overview strip."""
    if df.empty:
        return {}
    out = {
        "discharges": len(df),
        "avg_los": df["los"].mean(),
        "median_los": df["los"].median(),
        "avg_charges": df["charges"].mean(),
        "median_charges": df["charges"].median(),
        "total_charges": df["charges"].sum(),
        "long_stay_pct": df["long_stay"].mean() * 100,
        "bed_days": df["los"].sum(),
    }
    if "costs" in df:
        out["avg_costs"] = df["costs"].mean()
    return out


# --------------------------------------------------------------------------- #
# Group summaries
# --------------------------------------------------------------------------- #
def dx_summary(df: pd.DataFrame, min_cases: int = 100) -> pd.DataFrame:
    """Per diagnosis: volume, average / median stay, long-stay share, charges."""
    g = df.groupby("dx_label", observed=True)
    out = g.agg(
        cases=("los", "size"),
        avg_los=("los", "mean"),
        median_los=("los", "median"),
        long_stay_pct=("long_stay", lambda s: s.mean() * 100),
        avg_charges=("charges", "mean"),
        category=("dx_category", "first"),
    ).reset_index()
    out = out[out["cases"] >= min_cases]
    return out.sort_values("avg_los", ascending=False).reset_index(drop=True)


def age_gender_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Total / mean / median charges by age group (and gender when available)."""
    keys = ["age_group"] + (["gender"] if "gender" in df else [])
    out = df.groupby(keys, observed=True).agg(
        patients=("charges", "size"),
        total_charges=("charges", "sum"),
        avg_charges=("charges", "mean"),
        median_charges=("charges", "median"),
    ).reset_index()
    return out


def severity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Cost and stay by severity of illness, with a multiplier vs. 'Minor'."""
    agg = dict(cases=("los", "size"), avg_los=("los", "mean"),
               avg_charges=("charges", "mean"), median_charges=("charges", "median"))
    if "costs" in df:
        agg["avg_costs"] = ("costs", "mean")
    out = df.groupby("severity", observed=True).agg(**agg).reset_index()
    out = out[out["cases"] > 0].reset_index(drop=True)
    base = out.loc[out["severity"].astype(str) == "Minor", "avg_charges"]
    if len(base) and base.iloc[0] > 0:
        out["charge_multiple_vs_minor"] = out["avg_charges"] / base.iloc[0]
    return out


def county_summary(df: pd.DataFrame, min_cases: int = 100) -> pd.DataFrame:
    """Average stay per hospital county."""
    out = df.groupby("county", observed=True).agg(
        cases=("los", "size"), avg_los=("los", "mean"), median_los=("los", "median"),
        avg_charges=("charges", "mean"),
    ).reset_index()
    out = out[out["cases"] >= min_cases]
    return out.sort_values("avg_los", ascending=False).reset_index(drop=True)


def payment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Volume share and billing level per primary payment type."""
    agg = dict(patients=("charges", "size"), avg_charges=("charges", "mean"),
               median_charges=("charges", "median"), avg_los=("los", "mean"))
    if "cost_to_charge" in df:
        agg["median_cost_to_charge"] = ("cost_to_charge", "median")
    out = df.groupby("payment", observed=True).agg(**agg).reset_index()
    out["share_pct"] = out["patients"] / out["patients"].sum() * 100
    return out.sort_values("patients", ascending=False).reset_index(drop=True)


def facility_utilization(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """High-utilisation facilities ranked by bed-days (sum of stay length)."""
    out = df.groupby("facility", observed=True).agg(
        discharges=("los", "size"), bed_days=("los", "sum"), avg_los=("los", "mean"),
        total_charges=("charges", "sum"), county=("county", "first"),
    ).reset_index()
    return out.sort_values("bed_days", ascending=False).head(top).reset_index(drop=True)


def yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Discharges, average stay and average charges per discharge year."""
    if "year" not in df or df["year"].dropna().nunique() < 1:
        return pd.DataFrame()
    return df.dropna(subset=["year"]).groupby("year").agg(
        discharges=("los", "size"), avg_los=("los", "mean"), avg_charges=("charges", "mean"),
    ).reset_index()


# --------------------------------------------------------------------------- #
# Pivot tables
# --------------------------------------------------------------------------- #
def heatmap_matrix(df: pd.DataFrame, rows: str, cols: str, value: str = "los",
                   top_rows: int = 25, min_cases: int = 20) -> pd.DataFrame:
    """Pivot (mean of `value`) for heat-maps. Rows limited to the busiest `top_rows`."""
    keep = df[rows].value_counts().head(top_rows).index
    sub = df[df[rows].isin(keep)]
    pv = pd.pivot_table(sub, index=rows, columns=cols, values=value, aggfunc="mean", observed=True)
    counts = pd.pivot_table(sub, index=rows, columns=cols, values=value, aggfunc="size", observed=True)
    pv = pv.where(counts >= min_cases)
    pv = pv.dropna(how="all").dropna(axis=1, how="all")
    return pv


def region_payment_pivot(df: pd.DataFrame, value: str = "charges", agg: str = "mean") -> pd.DataFrame:
    """pivot_table(region x payment type) of the chosen metric."""
    idx = "region" if "region" in df else "county"
    return pd.pivot_table(df, index=idx, columns="payment", values=value, aggfunc=agg, observed=True)


def severity_age_pivot(df: pd.DataFrame, value: str = "los") -> pd.DataFrame:
    """Average stay (or charges) by age group x severity."""
    return pd.pivot_table(df, index="age_group", columns="severity", values=value,
                          aggfunc="mean", observed=True)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def distribution_stats(s: pd.Series) -> dict:
    """Mean, median, std, skew, kurtosis and key percentiles."""
    q = s.quantile([0.25, 0.75, 0.9, 0.95, 0.99])
    return {
        "mean": s.mean(), "median": s.median(), "std": s.std(),
        "skew": s.skew(), "kurtosis": s.kurt(),
        "p25": q[0.25], "p75": q[0.75], "p90": q[0.9], "p95": q[0.95], "p99": q[0.99],
    }


def spearman_los_charges(df: pd.DataFrame, max_rows: int = 200_000) -> float:
    """Rank correlation between stay length and charges (robust to skew)."""
    sub = df[["los", "charges"]]
    if len(sub) > max_rows:
        sub = sub.sample(max_rows, random_state=0)
    return float(sub["los"].corr(sub["charges"], method="spearman"))


def kruskal_by_group(df: pd.DataFrame, value: str, group: str, max_rows: int = 200_000):
    """Kruskal-Wallis H-test: do the groups differ in `value`? -> (H, p) or None."""
    try:
        from scipy import stats
    except ImportError:  # scipy is optional for the dashboard to run
        return None
    sub = df[[value, group]]
    if len(sub) > max_rows:
        sub = sub.sample(max_rows, random_state=0)
    samples = [g[value].to_numpy() for _, g in sub.groupby(group, observed=True) if len(g) > 5]
    if len(samples) < 2:
        return None
    h, p = stats.kruskal(*samples)
    return float(h), float(p)


def charge_concentration(df: pd.DataFrame, top_share: float = 0.10) -> float:
    """Share (%) of total charges generated by the most expensive `top_share` of discharges."""
    s = df["charges"].sort_values(ascending=False)
    k = max(1, int(len(s) * top_share))
    return float(s.iloc[:k].sum() / s.sum() * 100)


# --------------------------------------------------------------------------- #
# Plain-English answers to the research questions (computed from the data)
# --------------------------------------------------------------------------- #
def _money(x: float) -> str:
    return f"${x:,.0f}"


def answer_questions(df: pd.DataFrame, min_cases: int = 100) -> dict[str, str]:
    """Return {question id: finding text} computed from the (filtered) data."""
    ans: dict[str, str] = {}
    if df.empty:
        return {k: "No records match the current filters." for k in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]}

    dx = dx_summary(df, min_cases)
    if len(dx):
        top, low = dx.iloc[0], dx.iloc[-1]
        ans["Q1"] = (f"**{top['dx_label']}** has the longest average stay at **{top['avg_los']:.1f} days** "
                     f"({top['long_stay_pct']:.0f}% of its stays exceed {LONG_STAY_DAYS} days), "
                     f"versus {low['avg_los']:.1f} days for {low['dx_label']}. "
                     f"Only diagnoses with at least {min_cases:,} cases are ranked.")
    else:
        ans["Q1"] = f"No diagnosis has at least {min_cases:,} cases under the current filters - lower the threshold."

    ag = df.groupby("age_group", observed=True).agg(
        total=("charges", "sum"), avg=("charges", "mean"))
    ag = ag[ag["total"] > 0]
    if len(ag):
        a_tot, a_avg = ag["total"].idxmax(), ag["avg"].idxmax()
        share = ag.loc[a_tot, "total"] / ag["total"].sum() * 100
        ans["Q2"] = (f"Patients aged **{a_tot}** generate the most total charges "
                     f"({_money(ag.loc[a_tot, 'total'])}, {share:.0f}% of the total), while **{a_avg}** "
                     f"have the highest average bill per stay ({_money(ag.loc[a_avg, 'avg'])}).")

    sev = severity_summary(df)
    sev = sev[sev["severity"].astype(str) != "Unknown"]
    if len(sev) >= 2:
        lo, hi = sev.iloc[0], sev.iloc[-1]
        mult = hi["avg_charges"] / lo["avg_charges"] if lo["avg_charges"] else np.nan
        rho = spearman_los_charges(df)
        kw = kruskal_by_group(df, "charges", "severity")
        kw_txt = f" A Kruskal-Wallis test confirms the severity groups differ (H = {kw[0]:,.0f}, p < 0.001)." \
            if kw and kw[1] < 0.001 else ""
        ans["Q3"] = (f"Average charges rise from {_money(lo['avg_charges'])} for **{lo['severity']}** cases to "
                     f"{_money(hi['avg_charges'])} for **{hi['severity']}** cases - about **{mult:.1f}x** higher. "
                     f"Stay length and charges are strongly linked (Spearman rho = {rho:.2f}).{kw_txt}")

    cs = county_summary(df, min_cases)
    if len(cs) >= 2:
        top, low = cs.iloc[0], cs.iloc[-1]
        ans["Q4"] = (f"**{top['county']}** County has the longest average stay (**{top['avg_los']:.1f} days**) "
                     f"and **{low['county']}** the shortest ({low['avg_los']:.1f} days) - a gap of "
                     f"{top['avg_los'] - low['avg_los']:.1f} days across {len(cs)} counties.")
    else:
        ans["Q4"] = "Not enough counties with sufficient cases under the current filters."

    pay = payment_summary(df)
    pay_big = pay[pay["patients"] >= max(30, min_cases // 5)]
    if len(pay_big) >= 2:
        hi, lo = pay_big.sort_values("avg_charges").iloc[-1], pay_big.sort_values("avg_charges").iloc[0]
        lead = pay.iloc[0]
        ans["Q5"] = (f"**{lead['payment']}** covers the largest share of patients ({lead['share_pct']:.0f}%). "
                     f"Average charges are highest for **{hi['payment']}** ({_money(hi['avg_charges'])}) and lowest for "
                     f"**{lo['payment']}** ({_money(lo['avg_charges'])}). Part of the gap reflects different "
                     f"case mix (age, severity, stay length), not the payer alone.")

    fu = facility_utilization(df, top=10)
    if len(fu):
        total_bd = df["los"].sum()
        top3 = fu.head(3)["bed_days"].sum() / total_bd * 100
        ans["Q6"] = (f"**{fu.iloc[0]['facility']}** consumes the most bed-days "
                     f"({fu.iloc[0]['bed_days']:,.0f}); the top 3 facilities account for **{top3:.0f}%** of all bed-days. "
                     f"The most expensive 10% of discharges generate **{charge_concentration(df):.0f}%** of total charges.")
    for k in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
        ans.setdefault(k, "Not enough data under the current filters.")
    return ans
