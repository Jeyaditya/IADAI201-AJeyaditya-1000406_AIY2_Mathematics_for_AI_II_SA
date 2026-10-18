"""
Data loading, cleaning and feature engineering for the
Hospital Inpatient Discharges dashboard (MediScope Health Analytics).

The module is deliberately free of any Streamlit code so that the exact same
pipeline can be used by:
    * prepare_data.py  - one-off cleaning of the full raw file
    * app.py           - live cleaning of an uploaded file / demo data
    * tests/           - automated checks

Pipeline overview
-----------------
1. resolve_columns()    map the raw header onto canonical field names
2. basic_clean()        types, invalid rows, standardised categories
3. engineer_features()  outlier flags + derived analytical columns
4. restore_dtypes()     re-apply ordered categoricals to a saved clean file
"""
from __future__ import annotations

import gzip
import json
import os
import re
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
AGE_ORDER = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"]
SEVERITY_ORDER = ["Minor", "Moderate", "Major", "Extreme"]
LOS_BAND_LABELS = ["1-2 days", "3-5 days", "6-10 days", "11-20 days", "21+ days"]
LONG_STAY_DAYS = 7  # a stay longer than one week is treated as a "long stay"

# canonical name -> acceptable (normalised) raw column names, best match first
COLUMN_ALIASES: dict[str, list[str]] = {
    "region": ["hospital_service_area", "health_service_area", "service_area"],
    "county": ["hospital_county", "county"],
    "facility": ["facility_name", "hospital_name", "facility"],
    "age_group": ["age_group", "age"],
    "gender": ["gender", "sex"],
    "race": ["race"],
    "ethnicity": ["ethnicity"],
    "los": ["length_of_stay", "los", "length_of_stay_days"],
    "admission_type": ["type_of_admission", "admission_type"],
    "disposition": ["patient_disposition", "disposition"],
    "year": ["discharge_year", "year"],
    "dx_code": ["ccsr_diagnosis_code", "ccs_diagnosis_code", "diagnosis_code", "dx_code"],
    "dx_desc": ["ccsr_diagnosis_description", "ccs_diagnosis_description", "diagnosis_description", "diagnosis"],
    "mdc": ["apr_mdc_description", "mdc_description"],
    "severity": ["apr_severity_of_illness_description", "severity_of_illness", "severity"],
    "mortality_risk": ["apr_risk_of_mortality", "risk_of_mortality"],
    "surgical": ["apr_medical_surgical_description", "medical_surgical"],
    "payment": ["payment_typology_1", "payment_type", "primary_payer", "payment_typology"],
    "ed_indicator": ["emergency_department_indicator"],
    "charges": ["total_charges", "charges"],
    "costs": ["total_costs", "costs"],
    "birth_weight": ["birth_weight"],
}
REQUIRED = ["los", "charges", "severity", "age_group", "county", "facility", "payment"]
CATEGORICAL_COLUMNS = [
    "region", "county", "facility", "age_group", "gender", "race", "ethnicity",
    "admission_type", "disposition", "dx_label", "dx_desc", "dx_category", "mdc",
    "severity", "mortality_risk", "surgical", "payment", "ed_indicator", "los_band",
]

# Broad clinical groupings for single-level AHRQ CCS diagnosis codes
CCS_RANGES = [
    (1, 10, "Infectious diseases"),
    (11, 47, "Cancer (neoplasms)"),
    (48, 58, "Endocrine & metabolic"),
    (59, 64, "Blood disorders"),
    (76, 95, "Nervous system"),
    (96, 121, "Circulatory (heart & vascular)"),
    (122, 134, "Respiratory"),
    (135, 155, "Digestive"),
    (156, 175, "Genitourinary"),
    (176, 196, "Pregnancy & childbirth"),
    (197, 200, "Skin"),
    (201, 212, "Musculoskeletal"),
    (213, 217, "Congenital anomalies"),
    (218, 224, "Perinatal conditions"),
    (225, 244, "Injury & poisoning"),
    (245, 259, "Symptoms & other conditions"),
    (650, 670, "Mental health & substance use"),
]

# CCSR (refined) diagnosis codes are alphanumeric: the 3-letter prefix is the body system
CCSR_PREFIXES = {
    "BLD": "Blood disorders", "CIR": "Circulatory (heart & vascular)", "DIG": "Digestive",
    "EAR": "Ear & mastoid", "END": "Endocrine & metabolic", "EXT": "External causes of injury",
    "EYE": "Eye", "FAC": "Health status & services factors", "GEN": "Genitourinary",
    "INF": "Infectious diseases", "INJ": "Injury & poisoning", "MAL": "Congenital anomalies",
    "MBD": "Mental health & substance use", "MUS": "Musculoskeletal", "NEO": "Cancer (neoplasms)",
    "NVS": "Nervous system", "PNL": "Perinatal conditions", "PRG": "Pregnancy & childbirth",
    "RSP": "Respiratory", "SKN": "Skin", "SYM": "Symptoms & other conditions",
}

CLEAN_NAME, REPORT_NAME, META_NAME = "inpatient_clean.csv.gz", "cleaning_report.json", "clean_meta.json"


class MissingColumnsError(ValueError):
    """Raised when required fields cannot be located in the raw file."""

    def __init__(self, missing: list[str], available: list[str]):
        self.missing = missing
        self.available = available
        super().__init__(f"Could not find columns for: {', '.join(missing)}")


# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #
def normalise_name(col: str) -> str:
    """'Total Charges' -> 'total_charges'."""
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def resolve_columns(columns: Iterable[str], overrides: dict[str, str] | None = None):
    """Return (mapping canonical->raw column, list of missing required fields)."""
    columns = list(columns)
    norm = {normalise_name(c): c for c in columns}
    mapping: dict[str, str] = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for alias in aliases:  # exact match first
            if alias in norm:
                mapping[canon] = norm[alias]
                break
        else:  # fall back to "header contains the primary alias"
            hit = next((raw for n, raw in norm.items() if aliases[0] in n), None)
            if hit is not None:
                mapping[canon] = hit
    if overrides:
        mapping.update({k: v for k, v in overrides.items() if v})
    missing = [c for c in REQUIRED if c not in mapping]
    if "dx_code" not in mapping and "dx_desc" not in mapping:
        missing.append("dx_code")
    return mapping, missing


def wanted_column(col: str) -> bool:
    """usecols callback: read only the fields the dashboard can use."""
    n = normalise_name(col)
    for canon, aliases in COLUMN_ALIASES.items():
        if canon == "birth_weight":
            continue
        if n in aliases or aliases[0] in n:
            return True
    return False


def detect_sep(path) -> str:
    """Guess the delimiter (comma, tab, pipe, semicolon) from the header line."""
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace") as fh:
        head = fh.readline()
    counts = {sep: head.count(sep) for sep in [",", "\t", "|", ";"]}
    return max(counts, key=counts.get)


def read_raw(source, nrows: int | None = None, chunksize: int | None = None, sep: str | None = None):
    """Read a raw SPARCS-style file as text, keeping only the useful columns."""
    if sep is None:
        sep = detect_sep(source) if isinstance(source, (str, os.PathLike)) else ","
    return pd.read_csv(
        source, sep=sep, dtype=str, usecols=wanted_column, nrows=nrows, chunksize=chunksize,
        low_memory=True, keep_default_na=False, encoding="utf-8-sig", encoding_errors="replace",
    )


def count_rows(path) -> int:
    """Fast newline count (minus the header) so we can size a random sample."""
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    n = 0
    with opener(path, "rb") as fh:
        while block := fh.read(1 << 24):
            n += block.count(b"\n")
    return max(n - 1, 0)


def is_lfs_pointer(path) -> bool:
    """True if the file is a Git-LFS pointer stub rather than the real data."""
    try:
        if os.path.getsize(path) > 2048:
            return False
        with open(path, "rb") as fh:
            return fh.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def _clean_text(s: pd.Series, unknown: str = "Unknown") -> pd.Series:
    """Trim, collapse spaces and replace blanks / N/A markers with a label."""
    s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    bad = s.isna() | s.isin(["", "N/A", "NA", "nan", "None", "NULL", "null", "Not Available"])
    return s.mask(bad, unknown)


def _to_float(s: pd.Series) -> pd.Series:
    """'$12,345.60' -> 12345.6 ; '(100)' -> -100 ; junk -> NaN."""
    s = s.astype("string").str.strip()
    negative = (s.str.startswith("(") & s.str.endswith(")")).fillna(False).astype(bool)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    out = pd.to_numeric(s, errors="coerce").astype("float64")
    return out.where(~negative, -out.abs())


def _std_age(s: pd.Series) -> pd.Series:
    s = _clean_text(s)
    key = s.str.lower().str.replace(" ", "", regex=False)
    lookup = {"0to17": "0 to 17", "18to29": "18 to 29", "30to49": "30 to 49",
              "50to69": "50 to 69", "70orolder": "70 or Older", "70+": "70 or Older"}
    return key.map(lookup).fillna(s).astype("string")


def _std_gender(s: pd.Series) -> pd.Series:
    s = _clean_text(s)
    lookup = {"m": "Male", "male": "Male", "f": "Female", "female": "Female",
              "u": "Unknown", "unknown": "Unknown"}
    return s.str.lower().map(lookup).fillna("Unknown").astype("string")


def _std_severity(s: pd.Series) -> pd.Series:
    s = _clean_text(s)
    lookup = {k.lower(): k for k in SEVERITY_ORDER}
    return s.str.lower().map(lookup).fillna("Unknown").astype("string")


def _ccs_category(code: pd.Series) -> pd.Series:
    out = pd.Series("Other / unclassified", index=code.index, dtype="object")
    for lo, hi, label in CCS_RANGES:
        out[(code >= lo) & (code <= hi)] = label
    return out.astype("string")


def _dx_category(code: pd.Series) -> pd.Series:
    """Broad clinical group from a CCSR code ('CIR019') or legacy numeric CCS code ('108')."""
    code = code.astype("string")
    ccsr = code.str.extract(r"^([A-Za-z]{3})", expand=False).str.upper().map(CCSR_PREFIXES)
    legacy = _ccs_category(pd.to_numeric(code, errors="coerce"))
    return ccsr.fillna(legacy).astype("string")


def _log(report: list, step: str, before: int, after: int, note: str = "") -> None:
    report.append({"step": step, "rows_before": int(before), "rows_after": int(after),
                   "rows_removed": int(before - after), "note": note})


def merge_reports(reports: list[list[dict]]) -> list[dict]:
    """Combine per-chunk reports (same steps, summed counts)."""
    merged: dict[str, dict] = {}
    for rep in reports:
        for r in rep:
            m = merged.setdefault(r["step"], {**r, "rows_before": 0, "rows_after": 0, "rows_removed": 0})
            m["rows_before"] += r["rows_before"]
            m["rows_after"] += r["rows_after"]
            m["rows_removed"] += r["rows_removed"]
    return list(merged.values())


# --------------------------------------------------------------------------- #
# Stage 1: basic cleaning
# --------------------------------------------------------------------------- #
def basic_clean(raw: pd.DataFrame, mapping: dict[str, str], report: list | None = None,
                null_heavy_threshold: float = 0.60) -> pd.DataFrame:
    """Select, type-convert, standardise and validate the raw records."""
    report = report if report is not None else []
    n0 = len(raw)

    # 1. keep only mapped fields (this also drops Birth Weight & any unmapped column)
    df = pd.DataFrame({c: raw[col] for c, col in mapping.items() if c != "birth_weight"})
    n_dropped = raw.shape[1] - df.shape[1]
    _log(report, "Select relevant columns", n0, len(df),
         f"Kept {df.shape[1]} fields; dropped {n_dropped} unused/irrelevant (e.g. Birth Weight).")

    # 2. numeric conversion: length of stay ("120 +" -> 120) and currency strings
    df["los"] = _to_float(df["los"])
    df["charges"] = _to_float(df["charges"])
    if "costs" in df:
        df["costs"] = _to_float(df["costs"])
    if "year" in df:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # 3. invalid records
    before = len(df)
    df = df[df["los"].notna() & (df["los"] >= 1)]
    _log(report, "Remove invalid length of stay", before, len(df),
         "Missing, zero or negative stay durations are impossible and were removed.")
    before = len(df)
    df = df[df["charges"].notna() & (df["charges"] > 0)]
    _log(report, "Remove invalid charges", before, len(df),
         "Missing, zero or negative total charges were removed.")
    df = df.copy()
    if "costs" in df:
        df["costs"] = df["costs"].where(df["costs"] > 0)  # keep the record, discard the bad cost

    # 4. standardise categorical fields
    df["age_group"] = _std_age(df["age_group"])
    df["severity"] = _std_severity(df["severity"])
    if "gender" in df:
        df["gender"] = _std_gender(df["gender"])
    for c in ["region", "county", "facility", "payment", "race", "ethnicity", "admission_type",
              "disposition", "mdc", "mortality_risk", "surgical", "ed_indicator", "dx_desc"]:
        if c in df:
            df[c] = _clean_text(df[c])

    # 5. drop optional text columns that are mostly empty
    core = set(REQUIRED) | {"dx_code", "dx_desc"}
    heavy = [c for c in df.columns
             if c not in core and df[c].dtype.kind not in "fi"
             and (df[c].astype(str) == "Unknown").mean() > null_heavy_threshold]
    if heavy:
        df = df.drop(columns=heavy)
    _log(report, "Drop null-heavy optional columns", len(df), len(df),
         f"Dropped (>{int(null_heavy_threshold * 100)}% unknown): {', '.join(heavy) if heavy else 'none'}.")

    # 6. diagnosis: readable label + broad clinical category (CCSR codes are alphanumeric)
    if "dx_code" in df:
        code = _clean_text(df["dx_code"], unknown="")
        df["dx_code"] = code
        if "dx_desc" in df:
            desc = df["dx_desc"].astype("string")
            df["dx_label"] = (code + " - " + desc).where(code != "", desc).astype("string")
        else:
            df["dx_label"] = code
        df["dx_category"] = _dx_category(code)
        if "mdc" in df:  # fall back to MDC where the code has no body-system group
            other = df["dx_category"] == "Other / unclassified"
            df["dx_category"] = df["dx_category"].mask(other, df["mdc"].astype("string"))
    else:
        df["dx_label"] = df["dx_desc"].astype("string")
        df["dx_category"] = df["mdc"] if "mdc" in df else "Other / unclassified"
    _log(report, "Map diagnosis codes to readable labels", len(df), len(df),
         "Built 'dx_label' (code - description) and 'dx_category' (broad clinical group).")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 2: feature engineering
# --------------------------------------------------------------------------- #
def _tukey_far_out(x: pd.Series, k: float = 3.0) -> pd.Series:
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def engineer_features(df: pd.DataFrame, report: list | None = None) -> pd.DataFrame:
    """Add outlier flags and derived analytical columns."""
    report = report if report is not None else []
    df = df.copy()

    # Outliers on the log scale (charges & stays are strongly right-skewed).
    # "Far out" Tukey fences (3 x IQR) keep legitimate long/expensive stays.
    df["charge_outlier"] = _tukey_far_out(np.log10(df["charges"]))
    df["los_outlier"] = _tukey_far_out(np.log1p(df["los"]))
    df["is_outlier"] = df["charge_outlier"] | df["los_outlier"]
    _log(report, "Flag statistical outliers", len(df), len(df),
         f"{int(df['is_outlier'].sum()):,} records flagged (3xIQR on log scale); "
         "kept in the file, excluded by default in the dashboard (toggle in sidebar).")

    df["charge_per_day"] = df["charges"] / df["los"]
    if "costs" in df:
        df["cost_per_day"] = df["costs"] / df["los"]
        df["cost_to_charge"] = df["costs"] / df["charges"]
    df["los_band"] = pd.cut(df["los"], bins=[0, 2, 5, 10, 20, np.inf], labels=LOS_BAND_LABELS)
    df["long_stay"] = df["los"] > LONG_STAY_DAYS
    _log(report, "Engineer new features", len(df), len(df),
         "charge_per_day, cost_per_day, cost_to_charge, los_band, long_stay (>7 days).")
    return restore_dtypes(df)


def restore_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast text fields to (ordered) categoricals - small in memory, fast to group."""
    df = df.copy()
    for c in CATEGORICAL_COLUMNS:
        if c not in df:
            continue
        values = df[c].astype(str)
        if c in ("age_group", "severity"):
            order = AGE_ORDER if c == "age_group" else SEVERITY_ORDER
            seen = set(values)
            cats = [a for a in order if a in seen] + sorted(seen - set(order))
            df[c] = pd.Categorical(values, categories=cats, ordered=True)
        elif c == "los_band":
            df[c] = pd.Categorical(values, categories=LOS_BAND_LABELS, ordered=True)
        else:
            df[c] = values.astype("category")
    for c in ["charge_outlier", "los_outlier", "is_outlier", "long_stay"]:
        if c in df:
            df[c] = df[c].astype(bool)
    return df


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #
def clean_data(raw: pd.DataFrame, overrides: dict[str, str] | None = None):
    """Run the complete pipeline on a raw DataFrame -> (clean_df, report)."""
    mapping, missing = resolve_columns(raw.columns, overrides)
    if missing:
        raise MissingColumnsError(missing, list(raw.columns))
    report: list[dict] = []
    df = basic_clean(raw, mapping, report)
    df = engineer_features(df, report)
    return df, report


def read_clean(path) -> pd.DataFrame:
    """Load a file written by prepare_data.py."""
    return restore_dtypes(pd.read_csv(path, low_memory=False))


# --------------------------------------------------------------------------- #
# Building the clean dataset from the raw file (used by the app AND prepare_data.py)
# --------------------------------------------------------------------------- #
def build_clean_from_raw(path, max_rows: int | None = None, chunksize: int = 200_000, seed: int = 42,
                         progress: Callable[[float, str], None] | None = None):
    """Clean a (possibly huge) raw file chunk by chunk -> (clean_df, report).

    max_rows: keep a uniform random sample of about this many valid rows (None = every row).
    progress: optional callback(fraction 0-1, message).
    """
    say = progress or (lambda f, m: None)
    sep = detect_sep(path)
    header = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig", encoding_errors="replace")
    mapping, missing = resolve_columns(header.columns)
    if missing:
        raise MissingColumnsError(missing, list(header.columns))

    say(0.01, "Counting rows ...")
    total = max(count_rows(path), 1)
    frac = 1.0 if not max_rows else min(1.0, max_rows * 1.10 / total)

    parts, reports, n_raw, n_valid = [], [], 0, 0
    for i, chunk in enumerate(read_raw(path, chunksize=chunksize, sep=sep), 1):
        n_raw += len(chunk)
        rep: list = []
        clean = basic_clean(chunk, mapping, rep)
        n_valid += len(clean)
        if frac < 1:
            clean = clean.sample(frac=frac, random_state=seed + i)
        parts.append(clean.astype({c: "object" for c in clean.select_dtypes("string").columns}))
        reports.append(rep)
        say(min(0.9, 0.05 + 0.85 * n_raw / total), f"Cleaned {n_raw:,} of ~{total:,} rows ...")

    df = pd.concat(parts, ignore_index=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)

    say(0.93, "Engineering features ...")
    report = merge_reports(reports)
    if len(df) < n_valid:
        report.append({"step": "Random sample (size limit)", "rows_before": n_valid, "rows_after": len(df),
                       "rows_removed": n_valid - len(df),
                       "note": f"Uniform random sample (max_rows={max_rows:,}, seed={seed}) to keep the app fast "
                               "and memory-safe; group averages are statistically unchanged."})
    df = engineer_features(df, report)
    report.insert(0, {"step": "Raw rows read", "rows_before": n_raw, "rows_after": n_raw, "rows_removed": 0,
                      "note": os.path.basename(str(path))})
    report.append({"step": "Final dataset", "rows_before": len(df), "rows_after": len(df), "rows_removed": 0,
                   "note": f"{len(df):,} rows x {df.shape[1]} columns"})
    say(1.0, "Done")
    return df, report


def save_clean(df: pd.DataFrame, report: list, data_dir: str, raw_path=None, max_rows=None) -> bool:
    """Write the clean file + report + meta to data_dir. Returns False if the folder is read-only."""
    try:
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(os.path.join(data_dir, CLEAN_NAME), index=False, compression="gzip")
        with open(os.path.join(data_dir, REPORT_NAME), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        meta = {"raw_name": os.path.basename(str(raw_path)) if raw_path else None,
                "raw_size": os.path.getsize(raw_path) if raw_path else None,
                "max_rows": max_rows, "rows": len(df)}
        with open(os.path.join(data_dir, META_NAME), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        return True
    except OSError:
        return False


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def ensure_clean(raw_path, data_dir: str, max_rows: int | None = None, force: bool = False,
                 progress: Callable[[float, str], None] | None = None):
    """Return (clean_df, report, status) using the cached clean file whenever it is still valid.

    status: 'cache' (loaded from data/inpatient_clean.csv.gz) or 'built' (cleaned from the raw file now).
    The cache is reused when it was made from a raw file of the same size with the same row cap, or when the
    raw file is unavailable (e.g. a Git-LFS stub on a server).
    """
    clean_p = os.path.join(data_dir, CLEAN_NAME)
    meta = _read_json(os.path.join(data_dir, META_NAME))
    raw_ok = bool(raw_path) and os.path.exists(raw_path) and not is_lfs_pointer(raw_path)

    if os.path.exists(clean_p) and not force:
        fresh = (not raw_ok) or (meta and meta.get("raw_size") == os.path.getsize(raw_path)
                                 and meta.get("max_rows") == max_rows)
        if fresh:
            return read_clean(clean_p), _read_json(os.path.join(data_dir, REPORT_NAME)), "cache"
    if not raw_ok:
        raise FileNotFoundError("No readable raw data file (is it a Git-LFS pointer?) and no cleaned cache found.")
    df, report = build_clean_from_raw(raw_path, max_rows=max_rows, progress=progress)
    save_clean(df, report, data_dir, raw_path, max_rows)
    return df, report, "built"
