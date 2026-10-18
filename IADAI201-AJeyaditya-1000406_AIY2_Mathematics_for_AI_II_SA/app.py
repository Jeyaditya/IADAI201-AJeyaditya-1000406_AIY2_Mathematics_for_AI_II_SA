"""
MediScope Health Analytics - Hospital Inpatient Discharges Dashboard
====================================================================
Streamlit app for hospital administrators, clinicians and operations managers.

Run locally:   streamlit run app.py
Data:          data/inpatient_clean.csv.gz  (created by prepare_data.py)
               -> falls back to an optional upload, then to clearly-labelled demo data.
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analysis as A
from src import processing as P
from src.demo_data import make_raw

st.set_page_config(
    page_title="MediScope | Inpatient Discharges",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
INK, TEAL, AMBER, CRIMSON, SLATE = "#10242F", "#0B7A75", "#D99A2B", "#B23A2B", "#5B7083"
COLORWAY = [TEAL, AMBER, CRIMSON, SLATE, "#6C9BB8", "#8FBFAE", "#B9A27E", "#7C6A9C", "#C7D3DA"]
SEV_COLORS = {"Minor": "#7FB7A4", "Moderate": "#E8B04A", "Major": "#D9703A",
              "Extreme": "#A8281C", "Unknown": "#9AA7B0"}
SCALE_TEAL = [[0, "#EAF4F2"], [0.5, "#4FA89F"], [1, "#0A4A57"]]
SCALE_RISK = [[0, "#8FBFAE"], [0.5, "#E3A33B"], [1, "#B23A2B"]]

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stTabs button {{ font-family: 'Public Sans', 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 1.4rem; max-width: 1320px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.mx-title {{ font-size: 2.05rem; font-weight: 700; color: {INK}; line-height: 1.15; margin: 0; }}
.mx-sub {{ color: {SLATE}; font-size: 1rem; margin: 0.35rem 0 1.1rem 0; max-width: 62rem; }}
.kpi {{ background: #fff; border: 1px solid #DCE4E8; border-left: 5px solid {TEAL};
        border-radius: 6px; padding: 0.7rem 0.95rem; height: 100%; }}
.kpi .v {{ font-size: 1.65rem; font-weight: 700; color: {INK}; line-height: 1.2; }}
.kpi .l {{ font-size: 0.82rem; color: {SLATE}; font-weight: 500; }}
.kpi .s {{ font-size: 0.74rem; color: #7F909C; }}
.kpi.warn {{ border-left-color: {CRIMSON}; }}
.kpi.gold {{ border-left-color: {AMBER}; }}
.finding {{ background: #fff; border: 1px solid #DCE4E8; border-radius: 6px;
            padding: 0.8rem 1rem; margin-bottom: 0.6rem; }}
.chart-q {{ color: {SLATE}; font-size: 0.9rem; margin: -0.4rem 0 0.4rem 0; }}
.badge {{ display:inline-block; padding:2px 9px; border-radius: 999px; font-size: .72rem;
          font-weight:600; background:#E4F1EF; color:{TEAL}; margin-right:6px; }}
.badge.demo {{ background:#FBEAD0; color:#8A5A00; }}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CANDIDATES = [os.path.join(DATA_DIR, n) for n in
                  ("raw_sparcs.csv", "raw_sparcs.csv.gz", "raw_sparcs.txt")]
# The raw file can have millions of rows. The app cleans it ONCE, keeps a uniform random sample of about
# MAX_ROWS valid rows (fast + memory-safe, averages are statistically unchanged) and caches the result in
# data/. Set MAX_ROWS=0 (environment variable) to keep every row when running locally.
MAX_ROWS = int(os.environ.get("MAX_ROWS", 500_000)) or None


def money(x: float, short: bool = False) -> str:
    if x is None or pd.isna(x):
        return "-"
    if short:
        for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(x) >= div:
                return f"${x / div:,.2f}{suf}"
    return f"${x:,.0f}"


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def style(fig: go.Figure, height: int = 430, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, colorway=COLORWAY, font=dict(family="Public Sans, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10), showlegend=legend,
        hoverlabel=dict(font_size=13), legend=dict(title_text=""),
    )
    fig.update_xaxes(gridcolor="#E6ECEF", zeroline=False)
    fig.update_yaxes(gridcolor="#E6ECEF", zeroline=False)
    return fig


def chart_header(num: int, title: str, question: str) -> None:
    st.markdown(f"#### {num}. {title}")
    st.markdown(f'<div class="chart-q">{question}</div>', unsafe_allow_html=True)


def short_labels(labels: pd.Series, n: int = 52) -> pd.Series:
    cut = labels.astype(str).where(labels.astype(str).str.len() <= n, labels.astype(str).str.slice(0, n - 1) + "…")
    return cut if not cut.duplicated().any() else labels.astype(str)


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_dataset(raw_path: str | None, raw_size: int, max_rows: int | None):
    """Clean data/raw_sparcs.csv (or reuse the cached clean file). Shared across sessions, never mutated."""
    return P.ensure_clean(raw_path, DATA_DIR, max_rows=max_rows)


@st.cache_data(show_spinner="Cleaning uploaded file ...")
def clean_upload(file_bytes: bytes, overrides: tuple):
    head = file_bytes[:20_000].decode("utf-8-sig", "replace").splitlines()[0]
    sep = max([",", "\t", "|", ";"], key=head.count)
    raw = P.read_raw(io.BytesIO(file_bytes), sep=sep)
    return P.clean_data(raw, dict(overrides))


@st.cache_data(show_spinner="Generating demo data ...")
def load_demo():
    return P.clean_data(make_raw(60_000))


def find_raw() -> str | None:
    for path in RAW_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def get_data():
    """Return (df, cleaning_report, source_kind, source_label)."""
    with st.sidebar.expander("Data source", expanded=False):
        up = st.file_uploader("Upload a raw discharge CSV (optional)", type=["csv"],
                              help="Same layout as the NY SPARCS Hospital Inpatient Discharges file.")
    if up is not None:
        data = up.getvalue()
        overrides: dict[str, str] = {}
        while True:
            try:
                df, rep = clean_upload(data, tuple(sorted(overrides.items())))
                return df, rep, "upload", up.name
            except P.MissingColumnsError as e:
                st.error("I could not recognise every required column in the uploaded file. "
                         "Map the missing fields below and the dashboard will rebuild.")
                for m in e.missing:
                    overrides[m] = st.selectbox(f"Which column holds **{m}**?", [""] + e.available, key=f"map_{m}")
                if not all(overrides.get(m) for m in e.missing):
                    st.stop()

    raw = find_raw()
    cache_exists = os.path.exists(os.path.join(DATA_DIR, P.CLEAN_NAME))
    if raw or cache_exists:
        size = os.path.getsize(raw) if raw else 0
        try:
            with st.spinner("First run: cleaning the raw discharge file. This is a one-off step "
                            "(a few minutes for a very large file) - the result is cached for next time ..."):
                df, rep, status = load_dataset(raw, size, MAX_ROWS)
        except P.MissingColumnsError as e:
            st.error(f"The raw file is missing columns the dashboard needs: {', '.join(e.missing)}.")
            st.write("Columns found in your file:", e.available)
            st.info("Add the matching column names to `COLUMN_ALIASES` in `src/processing.py`.")
            st.stop()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        label = os.path.basename(raw) if raw else P.CLEAN_NAME
        n_raw = next((r["rows_before"] for r in (rep or []) if r["step"] == "Raw rows read"), None)
        if n_raw and n_raw > len(df):
            label += f" ({len(df):,} of {n_raw:,} raw rows)"
        return df, rep, "raw", label
    df, rep = load_demo()
    return df, rep, "demo", "synthetic demo data"


df_all, clean_report, src_kind, src_label = get_data()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<p class="mx-title">Hospital Inpatient Discharges</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="mx-sub">Where do patients stay longest, which cases drive cost, and where should hospital '
    "administrators focus? Explore stay length, charges, severity and payers across counties and facilities.</p>",
    unsafe_allow_html=True,
)
if src_kind == "demo":
    st.warning("**Demo mode** - no data file was found, so this dashboard is showing *synthetic* data. "
               "Place the raw dataset at `data/raw_sparcs.csv` and restart the app: it will be cleaned "
               "automatically on first run.", icon="⚠️")

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.markdown("### MediScope Health Analytics")
st.sidebar.caption("Filters apply to every chart, table and finding.")


def cat_filter(label: str, col: str, mask: pd.Series) -> pd.Series:
    if col not in df_all:
        return mask
    s = df_all[col]
    options = list(s.cat.categories) if hasattr(s, "cat") else sorted(s.dropna().unique())
    chosen = st.sidebar.multiselect(label, options, placeholder="All", key=f"f_{col}")
    return mask & s.isin(chosen) if chosen else mask


mask = pd.Series(True, index=df_all.index)
if "year" in df_all and df_all["year"].nunique() > 1:
    years = sorted(df_all["year"].dropna().astype(int).unique())
    sel_years = st.sidebar.multiselect("Discharge year", years, placeholder="All", key="f_year")
    if sel_years:
        mask &= df_all["year"].isin(sel_years)
if "region" in df_all:
    mask = cat_filter("Hospital service area", "region", mask)
mask = cat_filter("County", "county", mask)
mask = cat_filter("Facility", "facility", mask)
mask = cat_filter("Age group", "age_group", mask)
mask = cat_filter("Gender", "gender", mask)
mask = cat_filter("Severity of illness", "severity", mask)
mask = cat_filter("Primary payment type", "payment", mask)
mask = cat_filter("Type of admission", "admission_type", mask)
los_max = int(df_all["los"].max())
los_lo, los_hi = st.sidebar.slider("Length of stay (days)", 1, los_max, (1, los_max), key="f_los")
mask &= df_all["los"].between(los_lo, los_hi)
excl_out = st.sidebar.checkbox("Exclude extreme outliers", value=True, key="f_out",
                               help="Removes records flagged during cleaning as far outliers (3×IQR on the "
                                    "log scale) in stay length or charges.")
if excl_out:
    mask &= ~df_all["is_outlier"]

fdf = df_all.loc[mask]
st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(fdf):,}** of **{len(df_all):,}** discharges")
if len(fdf) == 0:
    st.warning("No discharges match the current filters. Widen or clear a filter in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- #
# KPI strip
# --------------------------------------------------------------------------- #
K = A.kpis(fdf)
cols = st.columns(5)
cols[0].markdown(kpi("Discharges", f"{K['discharges']:,}", f"{K['bed_days']:,.0f} bed-days"), unsafe_allow_html=True)
cols[1].markdown(kpi("Average stay", f"{K['avg_los']:.1f} days", f"median {K['median_los']:.0f} days"), unsafe_allow_html=True)
cols[2].markdown(kpi("Average charges", money(K["avg_charges"]), f"median {money(K['median_charges'])}", "gold"), unsafe_allow_html=True)
cols[3].markdown(kpi("Total charges", money(K["total_charges"], short=True), "billed, before payer discounts", "gold"), unsafe_allow_html=True)
cols[4].markdown(kpi("Long stays", f"{K['long_stay_pct']:.1f}%", f"stays over {P.LONG_STAY_DAYS} days", "warn"), unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------- #
# Research-question content
# --------------------------------------------------------------------------- #
QUESTIONS = {
    "Q1": ("Which diagnosis codes are linked to long stays?",
           "Length of stay drives bed occupancy and staffing. Diagnoses with persistently long stays are the best "
           "candidates for care pathways and early discharge planning.",
           "groupby diagnosis -> mean/median stay, share of stays over 7 days (minimum case threshold)."),
    "Q2": ("Which age groups incur the highest charges?",
           "Age shapes both how many patients arrive and how intensive their care is, so planners use age bands "
           "to forecast demand and budget.",
           "groupby age group (x gender) -> total, mean and median charges."),
    "Q3": ("How does patient severity affect total cost?",
           "Severity of illness (APR-DRG) is the standard risk-adjustment lens: comparing costs between hospitals or "
           "departments is misleading without it.",
           "groupby severity -> charge multiple vs. 'Minor'; Spearman correlation; Kruskal-Wallis test."),
    "Q4": ("Are there counties with longer average stays?",
           "Regional differences in stay length can point to case-mix differences, limited post-acute capacity "
           "(nursing homes, home care) or discharge inefficiencies.",
           "groupby county -> average stay (minimum case threshold), ranked."),
    "Q5": ("How do different payment types affect the final billing?",
           "The payer determines what a hospital is actually reimbursed, and billed charges are not the same as "
           "the cost of care, so billing patterns by payer matter for revenue planning.",
           "groupby payment type -> patient share, mean/median charges, cost-to-charge ratio."),
    "Q6": ("Which facilities show the highest utilisation?",
           "Resource use is often concentrated in a few facilities and a small share of expensive cases, so "
           "improvement effort aimed there pays off most.",
           "groupby facility -> discharges and bed-days (sum of stay length); charge concentration (top 10%)."),
}
min_cases_default = 100 if len(fdf) > 20_000 else 30
answers = A.answer_questions(fdf, min_cases_default)

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_dash, tab_rq, tab_eda, tab_data, tab_about = st.tabs(
    ["Dashboard", "Research questions", "Deep dive (EDA)", "Data & cleaning", "About"]
)

# =========================== TAB 1: DASHBOARD =============================== #
with tab_dash:
    with st.expander("Headline findings (update with your filters)", expanded=True):
        for q in ("Q1", "Q3", "Q6"):
            st.markdown(f"- {answers[q]}")

    # ---- Chart 1: bar - average stay per diagnosis ------------------------- #
    chart_header(1, "Average hospital stay per diagnosis", QUESTIONS["Q1"][0])
    c1, c2 = st.columns(2)
    top_n = c1.slider("Diagnoses to show", 5, 30, 15, key="dx_top")
    min_cases = c2.slider("Minimum cases per diagnosis", 10, 1000, min_cases_default, step=10, key="dx_min")
    dx = A.dx_summary(fdf, min_cases).head(top_n).iloc[::-1].copy()
    if dx.empty:
        st.info("No diagnosis meets the minimum-case threshold - lower it above.")
    else:
        dx["label_short"] = short_labels(dx["dx_label"])
        fig = px.bar(dx, x="avg_los", y="label_short", orientation="h", color="long_stay_pct",
                     color_continuous_scale=SCALE_RISK,
                     custom_data=["dx_label", "cases", "median_los", "long_stay_pct", "avg_charges"])
        fig.update_traces(
            texttemplate="%{x:.1f}", textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{customdata[0]}</b><br>Average stay: %{x:.1f} days<br>Median stay: %{customdata[2]:.0f} days"
                          "<br>Cases: %{customdata[1]:,}<br>Stays > 7 days: %{customdata[3]:.0f}%"
                          "<br>Average charges: $%{customdata[4]:,.0f}<extra></extra>")
        fig.update_coloraxes(colorbar_title="% stays<br>> 7 days")
        fig.update_xaxes(title="Average length of stay (days)")
        fig.update_yaxes(title="", automargin=True)
        st.plotly_chart(style(fig, height=max(380, 28 * len(dx) + 90), legend=False))
        st.caption("Bars are ranked by average stay; colour shows the share of stays longer than a week.")

    st.divider()

    # ---- Chart 2 & 4: box plot + pie --------------------------------------- #
    left, right = st.columns([3, 2])
    with left:
        chart_header(2, "Total charges by severity of illness", QUESTIONS["Q3"][0])
        log_y = st.toggle("Log scale", value=True, key="box_log",
                          help="Charges are heavily right-skewed; a log axis makes the boxes comparable.")
        sample = fdf.sample(min(len(fdf), 80_000), random_state=1)
        order = [s for s in P.SEVERITY_ORDER + ["Unknown"] if s in set(sample["severity"].astype(str))]
        fig = px.box(sample, x="severity", y="charges", color="severity", color_discrete_map=SEV_COLORS,
                     category_orders={"severity": order}, points=False, log_y=log_y)
        fig.update_traces(boxmean=True)
        fig.update_xaxes(title="Severity of illness (APR-DRG)")
        fig.update_yaxes(title="Total charges (US$)" + (" - log scale" if log_y else ""))
        st.plotly_chart(style(fig, legend=False))
        st.caption("Box = middle 50% of patients, line = median, dashed = mean. "
                   + (f"Drawn from a random sample of {len(sample):,} discharges for speed."
                      if len(fdf) > len(sample) else "All filtered discharges are drawn."))
    with right:
        chart_header(4, "Patients by payment type", QUESTIONS["Q5"][0])
        basis = st.radio("Share of", ["Patients", "Total charges"], horizontal=True, key="pie_basis")
        if basis == "Patients":
            pie = fdf["payment"].astype(str).value_counts().rename_axis("payment").reset_index(name="value")
        else:
            pie = fdf.groupby("payment", observed=True)["charges"].sum().reset_index(name="value")
            pie["payment"] = pie["payment"].astype(str)
        pie = pie.sort_values("value", ascending=False)
        small = pie["value"] / pie["value"].sum() < 0.02
        if small.sum() > 1:
            pie = pd.concat([pie[~small], pd.DataFrame({"payment": ["Other payers (<2% each)"],
                                                         "value": [pie.loc[small, "value"].sum()]})])
        fig = px.pie(pie, names="payment", values="value", hole=0.5, color_discrete_sequence=COLORWAY)
        fig.update_traces(textposition="inside", textinfo="percent", sort=False,
                          hovertemplate="<b>%{label}</b><br>%{percent}<br>%{value:,.0f}<extra></extra>")
        fig.update_layout(legend=dict(orientation="h", y=-0.15, font=dict(size=11)))
        st.plotly_chart(style(fig, height=470))

    st.divider()

    # ---- Chart 3: heatmap ---------------------------------------------------- #
    chart_header(3, "Average stay duration by facility and county", QUESTIONS["Q4"][0])
    views = {"Facility × County": ("facility", "county"), "County × Severity": ("county", "severity"),
             "County × Age group": ("county", "age_group")}
    if "region" in fdf:
        views["Service area × Payment type"] = ("region", "payment")
    h1, h2, h3 = st.columns(3)
    view = h1.selectbox("Heat-map view", list(views), key="hm_view")
    metric = h2.selectbox("Metric", ["Average stay (days)", "Average charges (US$)"], key="hm_metric")
    top_rows = h3.slider("Rows (busiest first)", 8, 40, 20, key="hm_rows")
    rows_col, cols_col = views[view]
    value_col = "los" if metric.startswith("Average stay") else "charges"
    pv = A.heatmap_matrix(fdf, rows_col, cols_col, value_col, top_rows=top_rows, min_cases=10)
    if pv.empty:
        st.info("Not enough data for this view under the current filters.")
    else:
        if view == "Facility × County":  # group facilities under their county for readability
            pv = pv.loc[pv.notna().idxmax(axis=1).sort_values(kind="stable").index]
        pv.index, pv.columns = pv.index.astype(str), pv.columns.astype(str)
        fmt = ".1f" if value_col == "los" else ",.0f"
        fig = px.imshow(pv, aspect="auto", color_continuous_scale=SCALE_TEAL,
                        text_auto=fmt if pv.shape[1] <= 10 else False,
                        labels=dict(x=cols_col.replace("_", " ").title(), y=rows_col.replace("_", " ").title(),
                                    color=metric))
        fig.update_xaxes(side="top", tickangle=-35)
        fig.update_yaxes(automargin=True)
        fig.update_traces(hovertemplate="%{y}<br>%{x}<br>" + metric + ": %{z:,.1f}<extra></extra>")
        st.plotly_chart(style(fig, height=max(380, 24 * len(pv) + 140), legend=False))
        st.caption("Blank cells have fewer than 10 discharges (or the facility does not operate in that county). "
                   "Darker = longer stay / higher charges.")

    st.divider()

    # ---- Chart 5: histogram --------------------------------------------------- #
    chart_header(5, "Distribution of length of stay", "How long do most patients stay - and how heavy is the tail?")
    g1, g2 = st.columns(2)
    cap = g1.slider("Show stays up to (days; longer stays are grouped in the last bar)", 10, 120, 30, key="hist_cap")
    log_counts = g2.toggle("Log scale (counts)", value=False, key="hist_log")
    capped = np.minimum(fdf["los"].round().astype(int), cap)
    hist = capped.value_counts().sort_index().rename_axis("days").reset_index(name="discharges")
    hist["share"] = hist["discharges"] / hist["discharges"].sum() * 100
    fig = px.bar(hist, x="days", y="discharges", custom_data=["share"], log_y=log_counts)
    fig.update_traces(marker_color=TEAL,
                      hovertemplate="%{x} days<br>%{customdata[0]:.1f}% of discharges"
                                    "<br>%{y:,} discharges<extra></extra>")
    fig.add_vline(x=K["median_los"], line_dash="dot", line_color=INK,
                  annotation_text=f"median {K['median_los']:.0f} d", annotation_position="top right")
    fig.add_vline(x=K["avg_los"], line_dash="dash", line_color=CRIMSON,
                  annotation_text=f"mean {K['avg_los']:.1f} d", annotation_position="top")
    fig.update_xaxes(title=f"Length of stay (days; last bar = {cap}+)")
    fig.update_yaxes(title="Discharges")
    st.plotly_chart(style(fig, height=400, legend=False))
    ds = A.distribution_stats(fdf["los"])
    st.caption(f"Mean ({ds['mean']:.1f}) sits above the median ({ds['median']:.0f}) and skewness is {ds['skew']:.1f}: "
               f"a minority of very long stays pulls the average up. 95% of stays end within {ds['p95']:.0f} days.")

# ======================= TAB 2: RESEARCH QUESTIONS ========================== #
with tab_rq:
    st.markdown("Six questions frame the analysis. Each finding below is **computed live from the current filters**, "
                "so it will change as you slice the data.")
    rq_min = st.slider("Minimum cases for ranking diagnoses and counties", 10, 1000, min_cases_default, step=10, key="rq_min")
    ans = A.answer_questions(fdf, rq_min)
    for qid, (question, why, method) in QUESTIONS.items():
        with st.expander(f"{qid}. {question}", expanded=(qid == "Q1")):
            st.markdown(f"**Finding.** {ans[qid]}")
            st.markdown(f"**Why it matters.** {why}")
            st.caption(f"Method: {method}")
            if qid == "Q1":
                t = A.dx_summary(fdf, rq_min).head(8)
                st.dataframe(t[["dx_label", "cases", "avg_los", "median_los", "long_stay_pct", "avg_charges"]],
                             hide_index=True, column_config={
                                 "dx_label": "Diagnosis", "cases": st.column_config.NumberColumn("Cases", format="%d"),
                                 "avg_los": st.column_config.NumberColumn("Avg stay (d)", format="%.1f"),
                                 "median_los": st.column_config.NumberColumn("Median (d)", format="%.0f"),
                                 "long_stay_pct": st.column_config.NumberColumn("% stays > 7 d", format="%.0f"),
                                 "avg_charges": st.column_config.NumberColumn("Avg charges", format="$%.0f")})
            elif qid == "Q2":
                t = fdf.groupby("age_group", observed=True).agg(
                    patients=("charges", "size"), total_charges=("charges", "sum"),
                    avg_charges=("charges", "mean")).reset_index()
                st.dataframe(t, hide_index=True, column_config={
                    "age_group": "Age group", "patients": st.column_config.NumberColumn("Patients", format="%d"),
                    "total_charges": st.column_config.NumberColumn("Total charges", format="$%.0f"),
                    "avg_charges": st.column_config.NumberColumn("Avg charges", format="$%.0f")})
            elif qid == "Q3":
                t = A.severity_summary(fdf)
                st.dataframe(t, hide_index=True, column_config={
                    "severity": "Severity", "cases": st.column_config.NumberColumn("Cases", format="%d"),
                    "avg_los": st.column_config.NumberColumn("Avg stay (d)", format="%.1f"),
                    "avg_charges": st.column_config.NumberColumn("Avg charges", format="$%.0f"),
                    "median_charges": st.column_config.NumberColumn("Median charges", format="$%.0f"),
                    "avg_costs": st.column_config.NumberColumn("Avg costs", format="$%.0f"),
                    "charge_multiple_vs_minor": st.column_config.NumberColumn("× vs Minor", format="%.2f")})
            elif qid == "Q4":
                t = A.county_summary(fdf, rq_min)
                if len(t) > 10:
                    t = pd.concat([t.head(5), t.tail(5)])
                st.dataframe(t, hide_index=True, column_config={
                    "county": "County", "cases": st.column_config.NumberColumn("Cases", format="%d"),
                    "avg_los": st.column_config.NumberColumn("Avg stay (d)", format="%.1f"),
                    "median_los": st.column_config.NumberColumn("Median (d)", format="%.0f"),
                    "avg_charges": st.column_config.NumberColumn("Avg charges", format="$%.0f")})
                st.caption("Top 5 and bottom 5 counties by average stay.")
            elif qid == "Q5":
                t = A.payment_summary(fdf)
                st.dataframe(t, hide_index=True, column_config={
                    "payment": "Payment type", "patients": st.column_config.NumberColumn("Patients", format="%d"),
                    "share_pct": st.column_config.NumberColumn("Share (%)", format="%.1f"),
                    "avg_charges": st.column_config.NumberColumn("Avg charges", format="$%.0f"),
                    "median_charges": st.column_config.NumberColumn("Median charges", format="$%.0f"),
                    "avg_los": st.column_config.NumberColumn("Avg stay (d)", format="%.1f"),
                    "median_cost_to_charge": st.column_config.NumberColumn("Cost-to-charge", format="%.2f")})
            elif qid == "Q6":
                t = A.facility_utilization(fdf, 8)
                st.dataframe(t, hide_index=True, column_config={
                    "facility": "Facility", "county": "County",
                    "discharges": st.column_config.NumberColumn("Discharges", format="%d"),
                    "bed_days": st.column_config.NumberColumn("Bed-days", format="%d"),
                    "avg_los": st.column_config.NumberColumn("Avg stay (d)", format="%.1f"),
                    "total_charges": st.column_config.NumberColumn("Total charges", format="$%.0f")})

    st.markdown("### What this means for administrators")
    st.markdown(
        "- **Discharge planning:** start planning on admission for the long-stay diagnoses in Q1; each avoided "
        "day frees a bed for the next patient.\n"
        "- **Resource allocation:** severity (Q3) and age (Q2) are the strongest levers on cost - budget and "
        "staff by case-mix, not by raw volume.\n"
        "- **Regional review:** investigate counties with unusually long stays (Q4) for post-acute capacity gaps "
        "before judging hospital performance.\n"
        "- **Payer strategy:** use payer-level billing patterns (Q5) alongside cost-to-charge ratios when forecasting "
        "revenue.\n"
        "- **Focus effort:** a few facilities and cases carry a large share of bed-days and charges (Q6), so "
        "target improvement programmes there first."
    )

# ============================ TAB 3: DEEP DIVE ============================== #
with tab_eda:
    st.markdown("Exploratory analysis behind the dashboard: distributions, group comparisons and pivot tables.")

    st.markdown("#### Distribution statistics")
    stats_df = pd.DataFrame({"Length of stay (days)": A.distribution_stats(fdf["los"]),
                             "Total charges (US$)": A.distribution_stats(fdf["charges"])}).rename(
        index={"p25": "25th pct", "p75": "75th pct", "p90": "90th pct", "p95": "95th pct", "p99": "99th pct",
               "std": "std. dev.", "skew": "skewness", "kurtosis": "kurtosis"})
    st.dataframe(stats_df.round(2))
    st.caption("Both measures are strongly right-skewed (skewness well above 0), which is why medians, "
               "log scales and rank-based tests are used throughout.")

    st.divider()
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("#### Charges by age group and gender")
        basis = st.radio("Metric", ["Total charges", "Average charges"], horizontal=True, key="age_metric")
        ag = A.age_gender_charges(fdf)
        ycol = "total_charges" if basis == "Total charges" else "avg_charges"
        fig = px.bar(ag, x="age_group", y=ycol, color="gender" if "gender" in ag else None, barmode="group",
                     color_discrete_sequence=[TEAL, AMBER, SLATE],
                     labels={"age_group": "Age group", ycol: basis + " (US$)", "gender": "Gender"})
        fig.update_traces(hovertemplate="%{x}<br>%{y:$,.0f}<extra></extra>")
        st.plotly_chart(style(fig, height=380))
    with e2:
        st.markdown("#### Severity: stay and charges")
        sev = A.severity_summary(fdf)
        sev["severity"] = sev["severity"].astype(str)
        fig = go.Figure()
        fig.add_bar(x=sev["severity"], y=sev["avg_charges"], name="Avg charges (US$)",
                    marker_color=[SEV_COLORS.get(s, SLATE) for s in sev["severity"]],
                    hovertemplate="%{x}<br>%{y:$,.0f}<extra></extra>")
        fig.add_scatter(x=sev["severity"], y=sev["avg_los"], name="Avg stay (days)", yaxis="y2",
                        mode="lines+markers", line=dict(color=INK, width=2.5),
                        hovertemplate="%{x}<br>%{y:.1f} days<extra></extra>")
        fig.update_layout(yaxis=dict(title="Average charges (US$)"),
                          yaxis2=dict(title="Average stay (days)", overlaying="y", side="right", showgrid=False),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(style(fig, height=380))

    st.divider()
    st.markdown("#### Does a longer stay mean a bigger bill?")
    sc = fdf.sample(min(len(fdf), 15_000), random_state=2)
    fig = px.scatter(sc, x="los", y="charges", color="severity", color_discrete_map=SEV_COLORS,
                     log_x=True, log_y=True, opacity=0.35,
                     category_orders={"severity": [s for s in P.SEVERITY_ORDER + ["Unknown"]]},
                     labels={"los": "Length of stay (days, log)", "charges": "Total charges (US$, log)",
                             "severity": "Severity"},
                     hover_data={"dx_label": True, "payment": True})
    lx, ly = np.log10(sc["los"]), np.log10(sc["charges"])
    slope, intercept = np.polyfit(lx, ly, 1)
    xs = np.linspace(lx.min(), lx.max(), 50)
    fig.add_scatter(x=10 ** xs, y=10 ** (intercept + slope * xs), mode="lines", name=f"log-log fit (slope {slope:.2f})",
                    line=dict(color=INK, width=2.5, dash="dash"))
    st.plotly_chart(style(fig, height=470))
    rho = A.spearman_los_charges(fdf)
    st.caption(f"Spearman rank correlation = **{rho:.2f}**. In log-log terms a 10% longer stay goes with roughly "
               f"**{slope * 10:.1f}%** higher charges. {len(sc):,} randomly sampled discharges are plotted.")

    st.divider()
    e3, e4 = st.columns(2)
    with e3:
        st.markdown("#### Counties with the longest average stay")
        cmin = st.slider("Minimum cases per county", 10, 1000, min_cases_default, step=10, key="cty_min")
        cs = A.county_summary(fdf, cmin).head(15).iloc[::-1]
        if cs.empty:
            st.info("No county meets the threshold.")
        else:
            fig = px.bar(cs, x="avg_los", y="county", orientation="h", color="avg_los", color_continuous_scale=SCALE_TEAL,
                         custom_data=["cases", "median_los", "avg_charges"],
                         labels={"avg_los": "Average stay (days)", "county": ""})
            fig.update_traces(hovertemplate="<b>%{y}</b><br>Avg stay %{x:.1f} d<br>Median %{customdata[1]:.0f} d"
                                            "<br>Cases %{customdata[0]:,}<extra></extra>")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(style(fig, height=max(340, 26 * len(cs) + 80), legend=False))
    with e4:
        st.markdown("#### High-utilisation facilities (bed-days)")
        nfac = st.slider("Facilities to show", 5, 25, 12, key="fac_n")
        fu = A.facility_utilization(fdf, nfac).iloc[::-1]
        fig = px.bar(fu, x="bed_days", y="facility", orientation="h", color="avg_los", color_continuous_scale=SCALE_RISK,
                     custom_data=["discharges", "avg_los", "county", "total_charges"],
                     labels={"bed_days": "Bed-days (sum of stay length)", "facility": "", "avg_los": "Avg stay (d)"})
        fig.update_traces(hovertemplate="<b>%{y}</b> (%{customdata[2]})<br>Bed-days %{x:,.0f}"
                                        "<br>Discharges %{customdata[0]:,}<br>Avg stay %{customdata[1]:.1f} d"
                                        "<br>Charges $%{customdata[3]:,.0f}<extra></extra>")
        fig.update_yaxes(automargin=True)
        st.plotly_chart(style(fig, height=max(340, 26 * len(fu) + 80), legend=False))

    st.divider()
    st.markdown("#### Pivot tables")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Region × payment type**")
        pm = st.selectbox("Value", ["Average charges (US$)", "Average stay (days)", "Number of discharges"], key="pv_metric")
        val, agg, fmt = {"Average charges (US$)": ("charges", "mean", ",.0f"),
                         "Average stay (days)": ("los", "mean", ".1f"),
                         "Number of discharges": ("los", "size", ",d")}[pm]
        pv = A.region_payment_pivot(fdf, val, agg)
        pv.index, pv.columns = pv.index.astype(str), pv.columns.astype(str)
        fig = px.imshow(pv, aspect="auto", color_continuous_scale=SCALE_TEAL, text_auto=fmt if pv.shape[1] <= 9 else False,
                        labels=dict(color=pm, x="Payment type", y=""))
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(style(fig, height=max(340, 38 * len(pv) + 160), legend=False))
    with p2:
        st.markdown("**Age group × severity (average stay, days)**")
        sp = A.severity_age_pivot(fdf, "los")
        sp.index, sp.columns = sp.index.astype(str), sp.columns.astype(str)
        fig = px.imshow(sp, aspect="auto", color_continuous_scale=SCALE_RISK, text_auto=".1f",
                        labels=dict(x="Severity", y="Age group", color="Avg stay (d)"))
        st.plotly_chart(style(fig, height=max(340, 38 * len(sp) + 160), legend=False))

    yt = A.yearly_trend(fdf)
    if len(yt) >= 2:
        st.divider()
        st.markdown("#### Trend by discharge year")
        t1, t2 = st.columns(2)
        fig = px.line(yt, x="year", y="discharges", markers=True, labels={"year": "Year", "discharges": "Discharges"})
        fig.update_xaxes(dtick=1)
        t1.plotly_chart(style(fig, height=300, legend=False))
        fig = px.line(yt, x="year", y="avg_los", markers=True, labels={"year": "Year", "avg_los": "Average stay (days)"})
        fig.update_xaxes(dtick=1)
        t2.plotly_chart(style(fig, height=300, legend=False))

# ========================= TAB 4: DATA & CLEANING =========================== #
with tab_data:
    badge = ('<span class="badge demo">DEMO DATA</span>' if src_kind == "demo"
             else f'<span class="badge">{src_kind.upper()}</span>')
    st.markdown(f"{badge} Source: **{src_label}** - {len(df_all):,} cleaned records, {df_all.shape[1]} fields.",
                unsafe_allow_html=True)

    st.markdown("#### Cleaning pipeline")
    if clean_report:
        rep = pd.DataFrame(clean_report)[["step", "rows_before", "rows_after", "rows_removed", "note"]]
        st.dataframe(rep, hide_index=True, column_config={
            "step": "Step", "rows_before": st.column_config.NumberColumn("Rows in", format="%d"),
            "rows_after": st.column_config.NumberColumn("Rows out", format="%d"),
            "rows_removed": st.column_config.NumberColumn("Removed", format="%d"), "note": "What happened"})
    st.markdown(
        "1. **Load** with pandas as text, reading only the columns the dashboard needs.\n"
        "2. **Convert** currency strings (`$12,345.60`) to floats and stays like `120 +` to numbers; "
        "`Discharge Year` to numeric.\n"
        "3. **Remove invalid records:** missing, zero or negative stay; missing or non-positive charges. "
        "Identical rows are *not* de-duplicated because the data are de-identified and genuine repeats exist.\n"
        "4. **Standardise categories:** trim whitespace, unify `M/F/U` to Male/Female/Unknown, order age groups "
        "and severity levels, replace `N/A`/blank with *Unknown*.\n"
        "5. **Drop irrelevant or null-heavy columns** such as Birth Weight (almost entirely zero/blank).\n"
        "6. **Map diagnosis codes** to a readable `code - description` label and a broad clinical category.\n"
        "7. **Engineer features:** charge per day, cost per day, cost-to-charge ratio, length-of-stay band, "
        "long-stay flag (> 7 days).\n"
        "8. **Flag outliers** with far-out Tukey fences (3×IQR) on the log scale for stay and charges; flagged "
        "rows are excluded by default (sidebar toggle)."
    )

    o1, o2, o3 = st.columns(3)
    n_out = int(df_all["is_outlier"].sum())
    o1.metric("Flagged outliers", f"{n_out:,}", f"{n_out / len(df_all) * 100:.2f}% of records", delta_color="off")
    o2.metric("Mean charges (with outliers)", money(df_all["charges"].mean()))
    o3.metric("Mean charges (without)", money(df_all.loc[~df_all["is_outlier"], "charges"].mean()))

    st.markdown("#### Preview of the filtered data")
    preview_cols = [c for c in ["facility", "county", "year", "age_group", "gender", "dx_label", "severity",
                                "payment", "los", "charges", "costs", "charge_per_day", "cost_to_charge",
                                "los_band", "long_stay"] if c in fdf]
    st.dataframe(fdf[preview_cols].head(200), hide_index=True)
    if st.checkbox("Prepare a CSV download of the filtered data (max 200,000 rows)", key="prep_dl"):
        st.download_button("Download CSV", fdf.head(200_000).to_csv(index=False).encode("utf-8"),
                           file_name="filtered_inpatient_discharges.csv", mime="text/csv")

    st.markdown("#### Data dictionary")
    dd = pd.DataFrame([
        ("los", "Length of stay in days (numeric; top-coded '120 +' read as 120)"),
        ("charges / costs", "Total hospital charges / costs in US$ (converted from currency strings)"),
        ("severity", "APR-DRG severity of illness: Minor < Moderate < Major < Extreme"),
        ("age_group / gender", "Patient demographics (standardised, ordered)"),
        ("dx_label / dx_category", "CCSR diagnosis as 'code - description' and its broad clinical group (body system)"),
        ("payment", "Primary payment typology (payer)"),
        ("county / region / facility", "Where the care was delivered"),
        ("charge_per_day", "charges ÷ los (engineered)"),
        ("cost_to_charge", "costs ÷ charges (engineered)"),
        ("los_band / long_stay", "Stay band (1-2, 3-5, 6-10, 11-20, 21+ d) and flag for stays > 7 days (engineered)"),
        ("is_outlier", "True if stay or charges are far outliers (3×IQR, log scale)"),
    ], columns=["Field", "Meaning"])
    st.dataframe(dd, hide_index=True)

# ============================== TAB 5: ABOUT ================================ #
with tab_about:
    st.markdown(
        """
### About this dashboard
**Audience:** hospital administrators, clinicians and operations managers at partner hospitals of
MediScope Health Analytics.

**Purpose:** turn inpatient discharge records into decisions - which diagnoses keep patients longest,
what drives cost, which counties and facilities need attention.

**How to use it**
1. Narrow the data with the sidebar filters (year, area, county, facility, age, gender, severity, payer, stay).
2. Read the five core charts on **Dashboard**, then check the plain-English answers on **Research questions**.
3. Use **Deep dive** for distributions, correlations and pivot tables, and **Data & cleaning** to see how the
   data were prepared.

**Method notes & limitations**
- *Charges are not costs.* Charges are what hospitals bill; costs are estimated resource use. Both are shown.
- *Association is not causation.* Payer or county differences partly reflect case mix (age, severity).
- *Skewed data.* Stays and charges have long right tails, so medians, log scales and rank-based statistics
  (Spearman, Kruskal-Wallis) are used alongside means.
- *Outliers.* Extreme records are excluded by default; untick the sidebar box to include them.
- Records are de-identified; no patient-level information is displayed.

**Built with** Python, pandas, NumPy, SciPy, Plotly and Streamlit.
"""
    )
