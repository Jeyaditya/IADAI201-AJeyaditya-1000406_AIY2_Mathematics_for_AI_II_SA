# 🏥 Hospital Inpatient Discharges Dashboard

**MediScope Health Analytics · Healthcare AI & Data Insights Team**
Course: *Mathematics for AI-II* · CRS: *Artificial Intelligence* · Summative Assessment (Scenario 2)

| | |
| --- | --- |
| **Live web app** | 👉 **[PASTE YOUR STREAMLIT CLOUD LINK HERE]** |
| **Student** | [Full name] · [Student ID] · [Candidate registration number] |
| **School** | [School name] |

![Dashboard overview](screenshots/01_dashboard.png)

---

## 1. Project overview

Hospital administrators need to know **where patients stay longest, which cases drive cost, and where to focus
limited beds, staff and budget**. This project turns a hospital inpatient discharge dataset into an interactive
Streamlit dashboard that answers those questions for non-technical decision makers.

**What the dashboard analyses**

- How long patients stay, by diagnosis, county, facility, age group and severity
- What hospitals bill (total charges) and how that varies by severity, age, gender and payer
- Which facilities use the most bed-days (high-utilisation facilities) and how concentrated resource use is
- How strongly stay length and cost are linked (correlation and statistical tests)

**Target users:** hospital administrators, clinicians and operations managers.

### Key features

- ✅ **5 required interactive Plotly charts** (bar, box plot, heat-map, pie, histogram) plus 7+ deep-dive charts
- ✅ **Global sidebar filters** (year, service area, county, facility, age, gender, severity, payer, admission type, stay range, outlier toggle) that update every chart, KPI and finding
- ✅ **Live KPI strip:** discharges, bed-days, average stay, average/total charges, long-stay share
- ✅ **Research-question tab** with plain-English answers **computed from the current filters**
- ✅ **Deep-dive EDA:** distribution statistics, log-log scatter with fit, Spearman correlation, Kruskal-Wallis test, pivot tables
- ✅ **Transparent data cleaning:** a step-by-step audit table (rows in / out) is shown inside the app
- ✅ **Robust loading:** schema-aware column detection, a column-mapping fallback for uploads, chunked cleaning for very large files
- ✅ **Automated tests** for the cleaning and analysis code

### Screenshots

| Dashboard | Research questions | Deep dive |
| --- | --- | --- |
| ![](screenshots/01_dashboard.png) | ![](screenshots/02_research.png) | ![](screenshots/03_deep_dive.png) |

| Heat-map | Data & cleaning |
| --- | --- |
| ![](screenshots/04_heatmap.png) | ![](screenshots/05_data_cleaning.png) |

---

## 2. Step 1 - Understanding the problem & research

### Dataset
The project uses the *Hospital Inpatient Discharges* dataset provided in the assessment (`data/raw_sparcs.csv`; the layout follows the
New York State **SPARCS** de-identified discharge file). Each row is one hospital discharge with:

| Group | Fields |
| --- | --- |
| Demographics | Age Group, Gender, Race, Ethnicity |
| Clinical | CCSR Diagnosis Code & Description, APR DRG, APR MDC, **APR Severity of Illness**, Risk of Mortality, Medical/Surgical |
| Stay | **Length of Stay**, Type of Admission, Patient Disposition, Discharge Year |
| Financial | **Total Charges**, Total Costs, Payment Typology (payer) |
| Facility | Hospital Service Area, Hospital County, Facility Name |

### Background research (what administrators look at)
- **Length of stay (LOS)** is the standard operational efficiency measure: it determines bed occupancy, staffing
  and the cost of each episode. LOS distributions are right-skewed, so medians and percentiles matter as much as means.
- **Severity of illness (APR-DRG)** is the accepted risk-adjustment scale (Minor → Extreme). Comparing hospitals,
  departments or payers without adjusting for severity and age is misleading.
- **Charges ≠ costs.** Charges are list prices billed; costs estimate resources used. The *cost-to-charge ratio* is
  therefore reported by payer.
- **CCSR (Clinical Classifications Software Refined)** from AHRQ's HCUP collapses thousands of ICD-10 codes into clinically
  meaningful diagnosis categories (codes like `CIR019`; the 3-letter prefix is the body system) - used here to build
  readable labels and broad clinical groups.
- Healthcare resource use is typically **concentrated** (a few facilities and a small share of expensive cases
  account for a large share of cost), so the dashboard explicitly measures concentration.

### Five+ business questions

| # | Question | Where answered |
| --- | --- | --- |
| Q1 | Which diagnosis codes are linked to long stays? | Chart 1, Research tab |
| Q2 | Which age groups incur the highest charges? | Deep dive (age × gender), Research tab |
| Q3 | How does patient severity affect total cost? | Chart 2, Deep dive, Research tab |
| Q4 | Are there counties with longer average stays? | Chart 3, Deep dive, Research tab |
| Q5 | How do different payment types affect the final billing? | Chart 4, Pivot tables, Research tab |
| Q6 | Which facilities show the highest utilisation (bed-days)? | Deep dive, Research tab |

---

## 3. Step 2 - Data preprocessing & cleaning

All logic lives in [`src/processing.py`](src/processing.py) (modular, commented, Streamlit-free) and is run once on
the full file by [`prepare_data.py`](prepare_data.py).

| # | Step | Detail |
| --- | --- | --- |
| 1 | Load with pandas | Auto-detects delimiter/encoding; reads as text, only the needed columns, in chunks; keeps a uniform random sample of ~500k valid rows (`MAX_ROWS`) |
| 2 | Convert types | `"$12,345.60"` → `12345.6`; `"120 +"` → `120`; year → numeric |
| 3 | Remove invalid records | Missing / zero / **negative stay duration**; missing / non-positive charges |
| 4 | Standardise categories | Trim whitespace; `M/F/U` → Male/Female/Unknown; ordered age groups and severity; `N/A`/blank → *Unknown* |
| 5 | Drop irrelevant / null-heavy columns | e.g. **Birth Weight** (almost entirely 0/blank) |
| 6 | Map diagnosis codes | `dx_label` = "CCSR code - description"; `dx_category` = body system from the code prefix |
| 7 | Feature engineering | `charge_per_day`, `cost_per_day`, `cost_to_charge`, `los_band`, `long_stay` (> 7 days) |
| 8 | Outliers | Far-out Tukey fences (3×IQR) on log-scale stay and charges → `is_outlier` flag (excluded by default, toggle in sidebar) |

Row counts for every step are written to `data/cleaning_report.json` and displayed in the app
(*Data & cleaning* tab). Identical rows are intentionally **not** de-duplicated: the data are de-identified and
genuine repeats exist.

---

## 4. Step 3 - Exploratory data analysis

Implemented in [`src/analysis.py`](src/analysis.py) using `groupby().agg()` and `pivot_table()`:

- Average / median stay and long-stay share per diagnosis (minimum-case threshold)
- Total, mean and median charges by age group × gender
- Charge multiple by severity vs. *Minor*; Spearman ρ between stay and charges; Kruskal-Wallis test
- Average stay per county; high-utilisation facilities ranked by bed-days
- Payer mix, billing level and cost-to-charge ratio
- Pivot tables: region × payment type; age group × severity; facility × county
- Distribution statistics (skewness, kurtosis, percentiles) and charge concentration (top-10% share)

### Key findings

> ⚠️ **Fill this table after the app is running on your real data.** Open the **Research questions** tab with no
> filters, and copy the numbers - they are computed live, so they are guaranteed to match the dashboard.

| Question | Finding (from your data) |
| --- | --- |
| Q1 Long-stay diagnoses | [e.g. "Schizophrenia … averages X days, Y% of stays exceed 7 days"] |
| Q2 Age & charges | [..] |
| Q3 Severity & cost | [e.g. "Extreme cases cost X× Minor cases; ρ = …"] |
| Q4 Counties | [..] |
| Q5 Payment types | [..] |
| Q6 Utilisation | [..] |

---

## 5. Step 4 - Interactive visualisations

| # | Chart (as required) | Type | Interactivity |
| --- | --- | --- | --- |
| 1 | Avg. hospital stay per diagnosis code | Horizontal bar | Top-N and minimum-case sliders, rich hover, colour = % long stays |
| 2 | Total charges segmented by severity | Box plot | Log-scale toggle, mean marker |
| 3 | Avg. stay by facility and county | Heat-map | 4 views, 2 metrics, row slider |
| 4 | Distribution of patients by payment type | Donut / pie | Switch between patients and total charges |
| 5 | Overall distribution of length of stay | Histogram | Range slider, log toggle, mean/median markers |

Extras in **Deep dive:** age × gender grouped bars, severity dual-axis chart, LOS-vs-charges log-log scatter with fit,
county ranking, facility utilisation, two pivot-table heat-maps, yearly trend lines.

**Design choices:** one colour story (teal = volume/length, amber = money, crimson = risk), severity colours that
run from calm to alarming, KPI strip always visible, filters in one place, every chart carries a plain-English
caption.

---

## 6. Integration details

```
 data/raw_sparcs.csv ──► src/processing.py (first run: chunked cleaning) ──► data/inpatient_clean.csv.gz (cache)
                                  ▲                                                   │
                  prepare_data.py (optional CLI)                                      ▼
                                                 app.py (filters, layout, UI) ──► src/analysis.py ──► Plotly charts
```

| Component | Role |
| --- | --- |
| `src/processing.py` | Column detection, cleaning, feature engineering (shared by CLI, app and tests) |
| `src/analysis.py` | Pure functions returning DataFrames / dicts for each analysis |
| `src/demo_data.py` | Synthetic raw-format data for tests and a *clearly labelled* demo fallback |
| `prepare_data.py` | Optional CLI to pre-build the cache with progress output |
| `app.py` | Streamlit UI: data loading (`st.cache_data`), sidebar filters, KPI strip, tabs, Plotly charts |
| `.streamlit/config.toml` | Theme and upload limit |

Data loading order: sidebar upload → `data/raw_sparcs.csv` (cleaned once, cached) → cached `data/inpatient_clean.csv.gz`
alone → demo data (with a visible warning), so the deployment never shows a blank page.

---

## 7. Deployment instructions

### Run locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) put the dataset at data/raw_sparcs.csv  (the app cleans it automatically on first run)
# 2) launch the dashboard
streamlit run app.py

# optional: pre-build the cache with progress output, or change the sample size
python prepare_data.py --max-rows 300000
MAX_ROWS=0 streamlit run app.py     # use every row (local only)
```

### Run the tests
```bash
python -m pytest -q        # or: python tests/test_pipeline.py
```

### Deploy on Streamlit Community Cloud
1. Run the app locally once so `data/inpatient_clean.csv.gz`, `cleaning_report.json` and `clean_meta.json` exist, then push
   everything (including `data/raw_sparcs.csv`) to GitHub. The cached files are the safety net: if the server cannot read
   the big raw file (e.g. a Git-LFS stub), the app uses the cache automatically.
2. Go to <https://streamlit.io/cloud> → **Deploy an app** → connect GitHub.
3. Select the repository, branch `main`, main file path `app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` automatically.
5. Copy the live URL into this README (top of page) and into the submission document.

---

## 8. Repository structure

```
.
├── app.py                      # Streamlit dashboard (main file)
├── prepare_data.py             # optional raw → clean CLI
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── src/
│   ├── processing.py           # cleaning & feature engineering
│   ├── analysis.py             # EDA / statistics
│   └── demo_data.py            # synthetic fallback + test data
├── data/
│   ├── raw_sparcs.csv          # raw dataset (input)
│   ├── inpatient_clean.csv.gz  # auto-generated clean cache used by the app
│   ├── cleaning_report.json
│   ├── clean_meta.json
│   └── README.md
├── tests/test_pipeline.py
├── screenshots/                # images used in this README
└── docs/
```

---

## 9. Limitations & ethics
- Records are de-identified; no patient-level information is displayed.
- Charges are billed amounts, not reimbursement or true cost.
- Differences between payers, counties or facilities partly reflect case mix; they are associations, not causes.
- Extreme outliers are excluded by default and can be re-included from the sidebar.

## 10. References
- NYS Department of Health - SPARCS (Statewide Planning and Research Cooperative System): <https://www.health.ny.gov/statistics/sparcs/>
- AHRQ HCUP - Clinical Classifications Software Refined (CCSR): <https://hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp>
- Streamlit blog - *Improving healthcare management with Streamlit*: <https://blog.streamlit.io/improving-healthcare-management-with-streamlit/>
- Plotly Express documentation: <https://plotly.com/python/plotly-express/>
- Streamlit documentation (layouts, caching, state): <https://docs.streamlit.io/>
- pandas documentation (`groupby`, `pivot_table`): <https://pandas.pydata.org/docs/>

---
*Built by [Full name] for Mathematics for AI-II · Summative Assessment.*
