"""
validate_input_data.py
======================
Validates all InputData files for DH_Cascade_Framework_v2.

Checks performed:
  1. All required Excel files exist
  2. Time-series files have shape (8760, 15)
  3. Scalar files have 15 county values
  4. Transmission matrix files are (15, 15)
  5. County names are consistent across all files
  6. No missing values (NaN)
  7. No negative values in demand / capacity / potential files
  8. Temperature profiles are physically plausible (> 200 K, < 420 K)
  9. Solar CF in [0, 1]
  10. Eligibility matrix is binary (0/1) and symmetric

Usage
-----
    python validate_input_data.py           # print report to stdout
    python validate_input_data.py --strict  # exit with code 1 if any check fails

    from validate_input_data import validate_all
    results = validate_all()                # returns dict of check results
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

_BASE = Path(__file__).parent / "InputData"
_READ = dict(index_col=0, engine="openpyxl", sheet_name="data")

COUNTIES = [
    "Harju", "Hiiu", "Ida-Viru", "Jogeva", "Jarva",
    "Laane", "Laane-Viru", "Polva", "Parnu", "Rapla",
    "Saare", "Tartu", "Valga", "Viljandi", "Voru",
]
N_HOURS    = 8760
N_COUNTIES = 15

# ── Required files ────────────────────────────────────────────────────────────
REQUIRED_FILES: dict[str, str] = {
    # Time series (8760 × 15)
    "HeatDemand/heatDemand_HTDHN_GWh.xlsx":              "timeseries",
    "HeatDemand/heatDemand_LTDHN_GWh.xlsx":              "timeseries",
    "HeatDemand/heatDemand_VLTDHN_GWh.xlsx":             "timeseries",
    "Temperature/ERA5_ambientTemperature_K.xlsx":         "timeseries",
    "Temperature/supplyTemperature_HTDHN_K.xlsx":         "timeseries",
    "Temperature/returnTemperature_HTDHN_K.xlsx":         "timeseries",
    "Temperature/supplyTemperature_LTDHN_K.xlsx":         "timeseries",
    "Temperature/returnTemperature_LTDHN_K.xlsx":         "timeseries",
    "Temperature/supplyTemperature_VLTDHN_K.xlsx":        "timeseries",
    "Temperature/returnTemperature_VLTDHN_K.xlsx":        "timeseries",
    "HeatSources/industrialWasteHeatOperationRate.xlsx":  "timeseries",
    "HeatSources/solarThermalOperationRate.xlsx":         "timeseries",
    # Scalar (15 × 1)
    "HeatSources/industrialWasteHeatPotential_GWh.xlsx":  "scalar",
    "HeatSources/geothermalPotential_GWh.xlsx":           "scalar",
    "HeatSources/biomassPotential_GWh.xlsx":              "scalar",
    # Matrix (15 × 15)
    "DHNetwork/DHPipeEligibility.xlsx":                   "matrix",
    "DHNetwork/DHPipeDistance_km.xlsx":                   "matrix",
    "DHNetwork/DHPipeLosses.xlsx":                        "matrix",
    # Cost tables (variable shape, checked separately)
    "CostAssumptions/heatTechnologyCostAssumptions.xlsx":       "table",
    "CostAssumptions/thermalStorageCostAssumptions.xlsx":       "table",
    "CostAssumptions/districtHeatingPipeCostAssumptions.xlsx":  "table",
}


def _load(rel_path: str) -> pd.DataFrame | None:
    p = _BASE / rel_path
    if not p.exists():
        return None
    return pd.read_excel(p, **_READ)


# ── Individual check functions ────────────────────────────────────────────────

class CheckResult:
    def __init__(self, name: str):
        self.name    = name
        self.passed  = True
        self.issues  = []

    def fail(self, msg: str):
        self.passed = False
        self.issues.append(msg)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        out = f"  [{status}] {self.name}"
        for iss in self.issues:
            out += f"\n         ⚠  {iss}"
        return out


def check_files_exist() -> CheckResult:
    r = CheckResult("All required files exist")
    for rel_path in REQUIRED_FILES:
        if not (_BASE / rel_path).exists():
            r.fail(f"Missing: InputData/{rel_path}")
    return r


def check_shapes() -> CheckResult:
    r = CheckResult("Data shapes correct")
    for rel_path, ftype in REQUIRED_FILES.items():
        df = _load(rel_path)
        if df is None:
            continue  # already caught by check_files_exist
        if ftype == "timeseries":
            if df.shape != (N_HOURS, N_COUNTIES):
                r.fail(f"{rel_path}: expected ({N_HOURS},{N_COUNTIES}), got {df.shape}")
        elif ftype == "scalar":
            if df.shape[0] != N_COUNTIES:
                r.fail(f"{rel_path}: expected {N_COUNTIES} rows, got {df.shape[0]}")
        elif ftype == "matrix":
            if df.shape != (N_COUNTIES, N_COUNTIES):
                r.fail(f"{rel_path}: expected ({N_COUNTIES},{N_COUNTIES}), got {df.shape}")
    return r


def check_county_names() -> CheckResult:
    r = CheckResult("County names consistent across all files")
    expected = set(COUNTIES)
    for rel_path, ftype in REQUIRED_FILES.items():
        df = _load(rel_path)
        if df is None:
            continue
        if ftype in ("timeseries",):
            actual = set(df.columns.tolist())
            extra   = actual - expected
            missing = expected - actual
            if extra:
                r.fail(f"{rel_path}: unexpected columns {extra}")
            if missing:
                r.fail(f"{rel_path}: missing columns {missing}")
        elif ftype in ("scalar",):
            actual = set(df.index.tolist())
            missing = expected - actual
            if missing:
                r.fail(f"{rel_path}: missing row indices {missing}")
        elif ftype == "matrix":
            row_miss = expected - set(df.index.tolist())
            col_miss = expected - set(df.columns.tolist())
            if row_miss:
                r.fail(f"{rel_path}: missing row indices {row_miss}")
            if col_miss:
                r.fail(f"{rel_path}: missing column indices {col_miss}")
    return r


def check_no_nan() -> CheckResult:
    r = CheckResult("No missing values (NaN)")
    for rel_path in REQUIRED_FILES:
        df = _load(rel_path)
        if df is None:
            continue
        n_nan = df.isnull().sum().sum()
        if n_nan > 0:
            r.fail(f"{rel_path}: {n_nan} NaN values")
    return r


def check_no_negative_demand() -> CheckResult:
    r = CheckResult("Demand and potentials are non-negative")
    non_neg_files = [
        "HeatDemand/heatDemand_HTDHN_GWh.xlsx",
        "HeatDemand/heatDemand_LTDHN_GWh.xlsx",
        "HeatDemand/heatDemand_VLTDHN_GWh.xlsx",
        "HeatSources/industrialWasteHeatPotential_GWh.xlsx",
        "HeatSources/geothermalPotential_GWh.xlsx",
        "HeatSources/biomassPotential_GWh.xlsx",
        "HeatSources/industrialWasteHeatOperationRate.xlsx",
        "HeatSources/solarThermalOperationRate.xlsx",
        "DHNetwork/DHPipeDistance_km.xlsx",
        "DHNetwork/DHPipeLosses.xlsx",
    ]
    for rel_path in non_neg_files:
        df = _load(rel_path)
        if df is None:
            continue
        n_neg = (df.select_dtypes("number") < -1e-9).sum().sum()
        if n_neg > 0:
            r.fail(f"{rel_path}: {n_neg} negative values")
    return r


def check_temperatures() -> CheckResult:
    r = CheckResult("Temperature values physically plausible (200–420 K)")
    temp_files = [
        "Temperature/ERA5_ambientTemperature_K.xlsx",
        "Temperature/supplyTemperature_HTDHN_K.xlsx",
        "Temperature/returnTemperature_HTDHN_K.xlsx",
        "Temperature/supplyTemperature_LTDHN_K.xlsx",
        "Temperature/returnTemperature_LTDHN_K.xlsx",
        "Temperature/supplyTemperature_VLTDHN_K.xlsx",
        "Temperature/returnTemperature_VLTDHN_K.xlsx",
    ]
    for rel_path in temp_files:
        df = _load(rel_path)
        if df is None:
            continue
        nums = df.select_dtypes("number")
        low  = (nums < 200).sum().sum()
        high = (nums > 420).sum().sum()
        if low:
            r.fail(f"{rel_path}: {low} values below 200 K")
        if high:
            r.fail(f"{rel_path}: {high} values above 420 K")
    # Supply must exceed return at every timestep
    for level in ("HTDHN", "LTDHN", "VLTDHN"):
        sup = _load(f"Temperature/supplyTemperature_{level}_K.xlsx")
        ret = _load(f"Temperature/returnTemperature_{level}_K.xlsx")
        if sup is None or ret is None:
            continue
        violations = (sup.select_dtypes("number").values <= ret.select_dtypes("number").values).sum()
        if violations:
            r.fail(f"{level}: supply <= return in {violations} timesteps")
    return r


def check_solar_cf() -> CheckResult:
    r = CheckResult("Solar thermal CF in [0, 1]")
    df = _load("HeatSources/solarThermalOperationRate.xlsx")
    if df is None:
        return r
    nums = df.select_dtypes("number")
    low  = (nums < -1e-9).sum().sum()
    high = (nums > 1.0 + 1e-9).sum().sum()
    if low:
        r.fail(f"solarThermalOperationRate: {low} values < 0")
    if high:
        r.fail(f"solarThermalOperationRate: {high} values > 1")
    return r


def check_eligibility_matrix() -> CheckResult:
    r = CheckResult("DH pipe eligibility is binary (0/1) and symmetric")
    df = _load("DHNetwork/DHPipeEligibility.xlsx")
    if df is None:
        return r
    nums = df.select_dtypes("number").values
    # Binary check
    non_binary = ~np.isin(nums, [0, 1])
    if non_binary.any():
        r.fail(f"DHPipeEligibility: {non_binary.sum()} non-binary values")
    # Symmetry check
    if not np.allclose(nums, nums.T):
        r.fail("DHPipeEligibility: matrix is not symmetric")
    # Self-connection should be zero
    diag = np.diag(nums)
    if diag.any():
        r.fail("DHPipeEligibility: diagonal is non-zero (self-connections)")
    return r


def check_demand_magnitudes() -> CheckResult:
    """Sanity check: total annual demand is in the right ballpark."""
    r = CheckResult("Heat demand annual totals physically reasonable")
    targets = {
        "HeatDemand/heatDemand_HTDHN_GWh.xlsx":  (1000, 8000),   # GWh/yr
        "HeatDemand/heatDemand_LTDHN_GWh.xlsx":  (200,  4000),
        "HeatDemand/heatDemand_VLTDHN_GWh.xlsx": (50,   2000),
    }
    for rel_path, (lo, hi) in targets.items():
        df = _load(rel_path)
        if df is None:
            continue
        total = df.select_dtypes("number").values.sum()
        if not (lo <= total <= hi):
            r.fail(f"{rel_path}: annual total = {total:.0f} GWh, expected {lo}–{hi} GWh")
    return r


# ── Main validation runner ────────────────────────────────────────────────────

def validate_all(verbose: bool = True) -> dict:
    """
    Run all checks and return a dict of {check_name: CheckResult}.

    Parameters
    ----------
    verbose : bool
        If True, print results to stdout.
    """
    checks = [
        check_files_exist(),
        check_shapes(),
        check_county_names(),
        check_no_nan(),
        check_no_negative_demand(),
        check_temperatures(),
        check_solar_cf(),
        check_eligibility_matrix(),
        check_demand_magnitudes(),
    ]

    n_pass = sum(1 for c in checks if c.passed)
    n_fail = len(checks) - n_pass

    if verbose:
        print("\n" + "═" * 65)
        print("  DH_Cascade_Framework_v2 — Input Data Validation Report")
        print("═" * 65)
        for c in checks:
            print(c)
        print("─" * 65)
        print(f"  Result: {n_pass}/{len(checks)} checks passed", end="")
        if n_fail == 0:
            print("  ✓ ALL PASS")
        else:
            print(f"  ✗ {n_fail} FAILED")
        print("═" * 65 + "\n")

    return {c.name: c for c in checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate DH InputData files")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 1 if any check fails")
    args = parser.parse_args()

    results = validate_all(verbose=True)
    if args.strict and any(not r.passed for r in results.values()):
        sys.exit(1)
