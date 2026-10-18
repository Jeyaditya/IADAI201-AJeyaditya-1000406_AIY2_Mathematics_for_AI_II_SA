"""
Synthetic data in the *raw* SPARCS "Hospital Inpatient Discharges" layout.

Used for (a) automated tests and (b) a clearly-labelled demo fallback inside the
dashboard when no real data file is present. It intentionally contains the same
kinds of mess as the real file (currency strings, '120 +' stays, negative stays,
blank severities, an almost-empty Birth Weight column ...) so the cleaning
pipeline is exercised end to end. It is NOT real patient data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_REGIONS = {
    "New York City": ["Bronx", "Kings", "New York", "Queens"],
    "Long Island": ["Nassau", "Suffolk"],
    "Hudson Valley": ["Westchester", "Orange", "Dutchess"],
    "Capital/Adirondack": ["Albany", "Saratoga", "Warren"],
    "Western NY": ["Erie", "Niagara", "Monroe"],
    "Central NY": ["Onondaga", "Oneida"],
}
_DX = {  # CCSR code: (description, LOS multiplier, charge multiplier)  - illustrative codes
    "INF002": ("Septicemia", 2.4, 2.2),
    "CIR011": ("Coronary atherosclerosis and other heart disease", 1.0, 1.8),
    "CIR015": ("Cardiac dysrhythmias", 0.9, 1.1),
    "CIR019": ("Heart failure", 1.6, 1.3),
    "RSP002": ("Pneumonia (except that caused by tuberculosis)", 1.4, 1.1),
    "RSP008": ("Chronic obstructive pulmonary disease and bronchiectasis", 1.3, 0.9),
    "GEN004": ("Urinary tract infections", 1.0, 0.7),
    "MUS010": ("Osteoarthritis", 0.8, 2.0),
    "PNL012": ("Liveborn", 0.6, 0.35),
    "INJ034": ("Complication of device, implant or graft", 1.5, 1.6),
    "MBD002": ("Depressive disorders", 1.7, 0.6),
    "MBD001": ("Schizophrenia spectrum and other psychotic disorders", 2.6, 0.7),
    "DIG004": ("Esophageal disorders", 0.7, 0.8),
    "SKN002": ("Skin and subcutaneous tissue infections", 1.2, 0.8),
}
_PAYERS = ["Medicare", "Medicaid", "Private Health Insurance", "Blue Cross/Blue Shield",
           "Self-Pay", "Managed Care, Unspecified", "Federal/State/Local/VA",
           "Miscellaneous/Other", "Department of Corrections"]
_PAY_P = [0.38, 0.22, 0.14, 0.12, 0.04, 0.04, 0.03, 0.02, 0.01]
_PAY_CHARGE = [1.05, 0.85, 1.15, 1.1, 0.9, 1.0, 1.05, 0.95, 0.9]
_AGES = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"]
_AGE_P = [0.17, 0.10, 0.17, 0.27, 0.29]


def make_raw(n: int = 60_000, seed: int = 7) -> pd.DataFrame:
    """Return a raw-format DataFrame with n rows (all columns are strings)."""
    rng = np.random.default_rng(seed)

    # facilities live in exactly one county
    fac_rows = []
    for region, counties in _REGIONS.items():
        for county in counties:
            for k in range(rng.integers(2, 5)):
                fac_rows.append((region, county, f"{county} {['General','Memorial','Medical Center','Community'][k % 4]} Hospital",
                                 rng.uniform(0.85, 1.25)))
    fac = pd.DataFrame(fac_rows, columns=["region", "county", "facility", "fac_mult"])
    pick = rng.choice(len(fac), size=n, p=rng.dirichlet(np.ones(len(fac)) * 2))
    f = fac.iloc[pick].reset_index(drop=True)

    codes = rng.choice(list(_DX), size=n)  # CCSR-style alphanumeric codes
    los_mult = np.array([_DX[c][1] for c in codes])
    chg_mult = np.array([_DX[c][2] for c in codes])
    sev_idx = np.clip(rng.choice(4, size=n, p=[0.42, 0.34, 0.18, 0.06]) + (los_mult > 2).astype(int) * rng.integers(0, 2, n), 0, 3)
    sev_name = np.array(["Minor", "Moderate", "Major", "Extreme"])[sev_idx]
    age_idx = rng.choice(5, size=n, p=_AGE_P)
    pay_idx = rng.choice(len(_PAYERS), size=n, p=_PAY_P)

    base_los = rng.lognormal(mean=0.55, sigma=0.65, size=n)
    los = np.maximum(1, np.round(base_los * los_mult * (1 + 0.55 * sev_idx) * (1 + 0.08 * age_idx) * f["fac_mult"].to_numpy() ** 0.6))
    charges = (rng.lognormal(9.1, 0.55, n) * chg_mult * (1 + 0.28 * los) ** 0.9
               * (1 + 0.55 * sev_idx) * np.array(_PAY_CHARGE)[pay_idx] * f["fac_mult"].to_numpy())
    costs = charges * rng.uniform(0.22, 0.45, n)

    sev_code = sev_idx + 1
    df = pd.DataFrame({
        "Hospital Service Area": f["region"],
        "Hospital County": f["county"],
        "Operating Certificate Number": rng.integers(1000000, 7999999, n).astype(str),
        "Permanent Facility Id": rng.integers(100, 9999, n).astype(str),
        "Facility Name": f["facility"],
        "Age Group": np.array(_AGES)[age_idx],
        "Zip Code - 3 digits": rng.choice(["100", "104", "112", "OOS", "142"], size=n),
        "Gender": rng.choice(["F", "M", "U"], size=n, p=[0.55, 0.449, 0.001]),
        "Race": rng.choice(["White", "Black/African American", "Other Race", "Multi-racial"], size=n, p=[0.55, 0.2, 0.22, 0.03]),
        "Ethnicity": rng.choice(["Not Span/Hispanic", "Spanish/Hispanic", "Unknown"], size=n, p=[0.75, 0.2, 0.05]),
        "Length of Stay": los.astype(int).astype(str),
        "Type of Admission": rng.choice(["Emergency", "Elective", "Urgent", "Newborn", "Trauma"], size=n, p=[0.55, 0.2, 0.15, 0.08, 0.02]),
        "Patient Disposition": rng.choice(["Home or Self Care", "Skilled Nursing Home", "Home w/ Home Health Services", "Expired"], size=n, p=[0.62, 0.15, 0.2, 0.03]),
        "Discharge Year": rng.choice(["2019", "2020", "2021"], size=n),
        "CCSR Diagnosis Code": codes,
        "CCSR Diagnosis Description": [_DX[c][0] for c in codes],
        "CCSR Procedure Code": "",
        "CCSR Procedure Description": "",
        "APR DRG Code": rng.integers(1, 960, n).astype(str),
        "APR DRG Description": "Illustrative DRG",
        "APR MDC Code": rng.integers(1, 25, n).astype(str),
        "APR MDC Description": rng.choice(["Diseases and Disorders of the Circulatory System", "Respiratory System", "Newborns"], size=n),
        "APR Severity of Illness Code": sev_code.astype(str),
        "APR Severity of Illness Description": sev_name,
        "APR Risk of Mortality": rng.choice(["Minor", "Moderate", "Major", "Extreme"], size=n, p=[0.6, 0.25, 0.1, 0.05]),
        "APR Medical Surgical Description": rng.choice(["Medical", "Surgical"], size=n, p=[0.75, 0.25]),
        "Payment Typology 1": np.array(_PAYERS)[pay_idx],
        "Payment Typology 2": rng.choice(["", "Medicaid", "Self-Pay"], size=n),
        "Payment Typology 3": "",
        "Birth Weight": np.where(rng.random(n) < 0.93, "0", "3200"),
        "Emergency Department Indicator": rng.choice(["Y", "N"], size=n),
        "Total Charges": [f"${v:,.2f}" for v in charges],
        "Total Costs": [f"${v:,.2f}" for v in costs],
    })

    # ---- inject realistic mess -------------------------------------------------
    bad = rng.random(n)
    df.loc[bad < 0.002, "Length of Stay"] = "-3"           # impossible stay
    df.loc[(bad >= 0.002) & (bad < 0.004), "Length of Stay"] = ""
    df.loc[rng.random(n) < 0.003, "Length of Stay"] = "120 +"  # SPARCS top-code
    df.loc[rng.random(n) < 0.002, "Total Charges"] = "$0.00"
    df.loc[rng.random(n) < 0.002, "Total Charges"] = ""
    df.loc[rng.random(n) < 0.03, "APR Severity of Illness Description"] = "N/A"
    df.loc[rng.random(n) < 0.01, "Gender"] = " m "
    df.loc[rng.random(n) < 0.01, "Hospital County"] = "  " + df["Hospital County"] + "  "
    return df
