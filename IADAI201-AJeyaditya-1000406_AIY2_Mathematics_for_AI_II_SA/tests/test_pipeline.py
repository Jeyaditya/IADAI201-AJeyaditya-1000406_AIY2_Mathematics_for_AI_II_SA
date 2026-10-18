"""Run with:  python -m pytest -q     (or)     python tests/test_pipeline.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analysis as A  # noqa: E402
from src import processing as P  # noqa: E402
from src.demo_data import make_raw  # noqa: E402


def _clean(n=20_000):
    raw = make_raw(n)
    return raw, *P.clean_data(raw)


def test_cleaning_removes_invalid_and_converts_types():
    raw, df, report = _clean()
    assert len(df) < len(raw)                          # something was removed
    assert (df["los"] >= 1).all()                      # no negative / zero stays
    assert df["los"].max() >= 120                      # '120 +' parsed as 120
    assert df["charges"].dtype == "float64" and (df["charges"] > 0).all()
    assert "birth_weight" not in df.columns            # irrelevant column dropped
    assert not df["severity"].isna().any()
    assert set(df["gender"].astype(str)) <= {"Male", "Female", "Unknown"}
    assert list(df["age_group"].cat.categories) == P.AGE_ORDER
    assert list(df["severity"].cat.categories)[:4] == P.SEVERITY_ORDER
    assert any(r["step"] == "Remove invalid length of stay" and r["rows_removed"] > 0 for r in report)


def test_whitespace_and_blank_markers_standardised():
    _, df, _ = _clean()
    assert not any(str(c) != str(c).strip() for c in df["county"].cat.categories)
    assert "N/A" not in set(df["severity"].astype(str))


def test_engineered_features():
    _, df, _ = _clean()
    for col in ["charge_per_day", "cost_to_charge", "los_band", "long_stay", "is_outlier",
                "dx_label", "dx_category"]:
        assert col in df.columns
    assert np.allclose(df["charge_per_day"], df["charges"] / df["los"])
    assert df["cost_to_charge"].dropna().between(0, 1.5).all()
    assert (df["long_stay"] == (df["los"] > P.LONG_STAY_DAYS)).all()
    assert 0 < df["is_outlier"].mean() < 0.05           # outliers are rare, not everything


def test_dx_category_mapping():
    code = pd.Series([2, 108, 122, 218, 657, 999])
    cat = P._ccs_category(code).tolist()
    assert cat[0] == "Infectious diseases"
    assert cat[1].startswith("Circulatory")
    assert cat[2] == "Respiratory"
    assert cat[3] == "Perinatal conditions"
    assert cat[4].startswith("Mental")
    assert cat[5] == "Other / unclassified"


def test_ccsr_alphanumeric_codes_and_categories():
    code = pd.Series(["CIR019", "rsp002", "MBD001", "PNL012", "108", "", "ZZZ9"], dtype="string")
    cat = P._dx_category(code).tolist()
    assert cat[0].startswith("Circulatory") and cat[1] == "Respiratory"
    assert cat[2].startswith("Mental") and cat[3] == "Perinatal conditions"
    assert cat[4].startswith("Circulatory")          # legacy numeric CCS code still works
    assert cat[5] == "Other / unclassified" and cat[6] == "Other / unclassified"
    _, df, _ = _clean(3000)
    assert df["dx_label"].astype(str).str.match(r"^[A-Z]{3}\d{3} - ").all()
    assert (df["dx_category"].astype(str) != "Other / unclassified").all()


def test_real_header_is_fully_recognised():
    mapping, missing = P.resolve_columns(make_raw(20).columns)
    assert missing == []
    assert mapping["dx_code"] == "CCSR Diagnosis Code"
    assert mapping["dx_desc"] == "CCSR Diagnosis Description"
    assert mapping["severity"] == "APR Severity of Illness Description"
    assert mapping["payment"] == "Payment Typology 1"        # not 2 or 3
    assert mapping["birth_weight"] == "Birth Weight"
    assert not P.wanted_column("Birth Weight") and not P.wanted_column("Payment Typology 2")
    assert not P.wanted_column("Permanent Facility Id") and not P.wanted_column("APR DRG Code")


def test_raw_file_formats_and_ensure_clean(tmp_path):
    raw = make_raw(6000, seed=11)
    for name, sep in [("comma.csv", ","), ("tab.csv", "\t"), ("pipe.csv", "|")]:
        f = tmp_path / name
        raw.to_csv(f, index=False, sep=sep)
        assert P.detect_sep(f) == sep
        df, rep = P.build_clean_from_raw(f, chunksize=2500)
        assert len(df) > 5000 and rep[0]["step"] == "Raw rows read" and rep[0]["rows_before"] == 6000
    # gzip + row cap + caching
    gz = tmp_path / "raw_sparcs.csv.gz"
    raw.to_csv(gz, index=False, compression="gzip")
    data_dir = tmp_path / "data"
    df, rep, status = P.ensure_clean(str(gz), str(data_dir), max_rows=2000)
    assert status == "built" and 1800 <= len(df) <= 2000
    assert any(r["step"].startswith("Random sample") for r in rep)
    df2, _, status2 = P.ensure_clean(str(gz), str(data_dir), max_rows=2000)
    assert status2 == "cache" and len(df2) == len(df)
    _, _, status3 = P.ensure_clean(str(gz), str(data_dir), max_rows=1500)   # cap changed -> rebuild
    assert status3 == "built"


def test_git_lfs_pointer_falls_back_to_cache(tmp_path):
    raw = make_raw(3000, seed=5)
    f = tmp_path / "raw.csv"
    raw.to_csv(f, index=False)
    data_dir = tmp_path / "data"
    P.ensure_clean(str(f), str(data_dir), max_rows=None)
    f.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123456789\n")
    assert P.is_lfs_pointer(f)
    df, _, status = P.ensure_clean(str(f), str(data_dir), max_rows=None)
    assert status == "cache" and len(df) > 2500
    import shutil
    shutil.rmtree(data_dir)
    try:
        P.ensure_clean(str(f), str(data_dir))
        assert False, "should raise"
    except FileNotFoundError:
        pass


def test_missing_required_columns_raise_clear_error(tmp_path):
    f = tmp_path / "bad.csv"
    pd.DataFrame({"foo": ["1"], "bar": ["2"]}).to_csv(f, index=False)
    try:
        P.build_clean_from_raw(f)
        assert False
    except P.MissingColumnsError as e:
        assert "los" in e.missing and "foo" in e.available


def test_column_resolution_and_missing_detection():
    mapping, missing = P.resolve_columns(make_raw(50).columns)
    assert missing == []
    assert mapping["charges"] == "Total Charges"
    _, missing = P.resolve_columns(["foo", "bar"])
    assert "los" in missing


def test_analysis_functions_run_and_make_sense():
    _, df, _ = _clean()
    k = A.kpis(df)
    assert k["discharges"] == len(df) and k["avg_los"] > 0
    dx = A.dx_summary(df, min_cases=50)
    assert dx["avg_los"].is_monotonic_decreasing
    sev = A.severity_summary(df)
    sev = sev[sev["severity"].astype(str) != "Unknown"]
    assert sev["avg_charges"].is_monotonic_increasing   # synthetic data is built this way
    assert not A.age_gender_charges(df).empty
    assert not A.county_summary(df, 50).empty
    pay = A.payment_summary(df)
    assert abs(pay["share_pct"].sum() - 100) < 1e-6
    assert len(A.facility_utilization(df, 5)) == 5
    hm = A.heatmap_matrix(df, "facility", "county")
    assert hm.shape[0] > 0 and hm.shape[1] > 0
    assert not A.region_payment_pivot(df).empty
    assert not A.severity_age_pivot(df).empty
    assert -1 <= A.spearman_los_charges(df) <= 1
    assert 10 < A.charge_concentration(df) <= 100
    ans = A.answer_questions(df, 50)
    assert set(ans) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}
    assert all("Not enough" not in v and "No records" not in v for v in ans.values()), ans


def test_empty_filter_does_not_crash():
    _, df, _ = _clean(3000)
    empty = df.iloc[0:0]
    assert A.kpis(empty) == {}
    assert all("No records" in v for v in A.answer_questions(empty).values())


def test_roundtrip_through_csv(tmp_path):
    _, df, _ = _clean(5000)
    f = tmp_path / "clean.csv.gz"
    df.to_csv(f, index=False)
    back = P.read_clean(f)
    assert len(back) == len(df)
    assert list(back["age_group"].cat.categories) == list(df["age_group"].cat.categories)
    assert back["long_stay"].dtype == bool


if __name__ == "__main__":
    import tempfile
    import pathlib
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            fn(pathlib.Path(tempfile.mkdtemp()))
        else:
            fn()
        print("PASS", fn.__name__)
