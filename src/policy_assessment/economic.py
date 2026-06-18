"""
economic.py — C1-C4 Economic Indicators
=========================================
Indicators:
  C1  Levelised Cost of Heat (LCOH) — €/MWh_heat
  C2  Net present value (NPV) — M€
  C3  Payback period — years
  C4  Employment impact — FTE-years per MW installed

Theory
------
  LCOH = (Σ_t (CAPEX_t·CRF + OPEX_t)) / Σ_t gen_t

  CRF  = r(1+r)^n / ((1+r)^n - 1)   capital recovery factor
  r    = discount rate (default 5 %)
  n    = technology lifetime [years]
  NPV  = Σ_t ( (Revenue_t - OPEX_t) / (1+r)^t ) - CAPEX
  PBP  = CAPEX / (Annual_savings)

References:
  IRENA (2020) Renewable Power Generation Costs.
  Connolly et al. (2014) Heat Roadmap Europe. Energy Policy 65, 475-489.
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))
from paths import FIGURES_DIR, TABLES_DIR, ensure_dirs

# ── Techno-economic parameters (€2025, IEA/IRENA baselines) ──────────────────
TECH_PARAMS = pd.DataFrame({
    "technology":    ["Heat pump","Biomass boiler","Gas boiler","Solar thermal","Geothermal","TTES"],
    "CAPEX_kEur_MW": [900,         350,             200,          800,            1500,         80],
    "OPEX_EUR_MWh":  [5.0,         8.0,             3.0,          2.0,            4.0,          1.0],
    "fuel_EUR_MWh":  [40.0,        25.0,            55.0,         0.0,            5.0,          0.0],
    "lifetime_yr":   [20,          25,              20,           25,             30,           20],
    "COP_or_eff":    [3.5,         0.88,            0.90,         0.42,           5.0,          0.95],
    "FTE_per_MW":    [0.05,        0.12,            0.03,         0.04,           0.08,         0.02],
}).set_index("technology")

DISCOUNT_RATE = 0.05    # 5%
HEAT_PRICE    = 80.0    # €/MWh_heat — market reference


def _crf(r: float, n: int) -> float:
    """Capital recovery factor."""
    return r * (1 + r)**n / ((1 + r)**n - 1)


def compute(generation_GWh: dict | None = None,
            capacity_MW: dict | None = None) -> pd.DataFrame:
    """
    Compute C1-C4 economic indicators.

    Parameters
    ----------
    generation_GWh : {technology: annual GWh delivered}
    capacity_MW    : {technology: installed MW}  (estimated if None)

    Returns pd.DataFrame (technology × C1..C4)
    """
    if generation_GWh is None:
        generation_GWh = {t: np.random.uniform(50, 500) for t in TECH_PARAMS.index}
    if capacity_MW is None:
        capacity_MW = {t: g * 1e3 / 8760 * 2.5 for t, g in generation_GWh.items()}

    rows = []
    for tech in TECH_PARAMS.index:
        gen  = generation_GWh.get(tech, 0.0)   # GWh/yr
        cap  = capacity_MW.get(tech, 0.0)       # MW
        if gen <= 0 or cap <= 0:
            rows.append({"technology": tech, "C1_LCOH_EUR_MWh":0,"C2_NPV_MEur":0,
                          "C3_PBP_yr":0,"C4_FTE_yr":0,"CAPEX_MEur":0,"OPEX_EUR_MWh":0})
            continue

        p   = TECH_PARAMS.loc[tech]
        crf = _crf(DISCOUNT_RATE, int(p["lifetime_yr"]))

        CAPEX_MEur   = p["CAPEX_kEur_MW"] * cap / 1e3
        CAPEX_annual = CAPEX_MEur * 1e6 * crf          # €/yr
        OPEX_annual  = p["OPEX_EUR_MWh"] * gen * 1e3   # €/yr
        fuel_annual  = p["fuel_EUR_MWh"] / p["COP_or_eff"] * gen * 1e3  # €/yr

        total_cost = CAPEX_annual + OPEX_annual + fuel_annual

        # C1 — LCOH
        c1 = total_cost / (gen * 1e3) if gen > 0 else 0.0  # €/MWh

        # C2 — NPV (over lifetime)
        n = int(p["lifetime_yr"])
        revenue_annual = HEAT_PRICE * gen * 1e3  # €/yr
        annual_profit  = revenue_annual - OPEX_annual - fuel_annual
        annuity_factor = (1 - (1+DISCOUNT_RATE)**(-n)) / DISCOUNT_RATE
        c2 = (annual_profit * annuity_factor - CAPEX_MEur * 1e6) / 1e6  # M€

        # C3 — Simple payback period
        c3 = CAPEX_MEur * 1e6 / max(annual_profit, 1.0)  # years

        # C4 — Employment
        c4 = p["FTE_per_MW"] * cap  # FTE-yr

        rows.append({
            "technology":      tech,
            "C1_LCOH_EUR_MWh": round(c1, 2),
            "C2_NPV_MEur":     round(c2, 2),
            "C3_PBP_yr":       round(c3, 1),
            "C4_FTE_yr":       round(c4, 1),
            "CAPEX_MEur":      round(CAPEX_MEur, 2),
            "OPEX_EUR_MWh":    round(p["OPEX_EUR_MWh"], 2),
        })

    return pd.DataFrame(rows).set_index("technology")


def export_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_dirs()
    out = path or TABLES_DIR / "sustainability_C_economic.csv"
    df.to_csv(out)
    return out


def plot(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    ensure_dirs()
    out_dir = out_dir or FIGURES_DIR
    techs  = df.index.tolist()
    colors = ["#2980b9","#27ae60","#c0392b","#f39c12","#8e44ad","#1abc9c"][:len(techs)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Economic Analysis — C1-C4 Indicators\nDH_Cascade_Framework_v4", fontsize=12, fontweight="bold")

    def _bar(ax, col, ylabel, title):
        vals = df[col].values
        bars = ax.bar(techs, vals, color=colors, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v + (abs(v)*0.02 + 0.3)*(1 if v>=0 else -1),
                    f"{v:.1f}", ha="center", fontsize=7.5)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8)

    _bar(axes[0,0], "C1_LCOH_EUR_MWh", "LCOH (€/MWh)", "C1 — Levelised Cost of Heat")
    _bar(axes[0,1], "C2_NPV_MEur",      "NPV (M€)",     "C2 — Net Present Value")
    _bar(axes[1,0], "C3_PBP_yr",        "Years",        "C3 — Simple Payback Period")
    _bar(axes[1,1], "C4_FTE_yr",        "FTE-yr",       "C4 — Employment Impact")

    fig.tight_layout()
    out_path = out_dir / "Figure_C_Economic.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def test() -> None:
    gen = {"Heat pump":400,"Biomass boiler":300,"Gas boiler":200,"Solar thermal":50,"Geothermal":80,"TTES":30}
    df  = compute(gen)
    assert "C1_LCOH_EUR_MWh" in df.columns
    assert (df["C1_LCOH_EUR_MWh"] > 0).all()
    assert (df["C3_PBP_yr"] >= 0).all()
    gas_lcoh = df.loc["Gas boiler","C1_LCOH_EUR_MWh"]
    hp_lcoh  = df.loc["Heat pump", "C1_LCOH_EUR_MWh"]
    assert hp_lcoh < gas_lcoh * 1.5, "HP LCOH should be competitive with gas"
    print("  ✓ economic: all unit tests passed")


if __name__ == "__main__":
    ensure_dirs()
    gen = {"Heat pump":400,"Biomass boiler":300,"Gas boiler":200,"Solar thermal":50,"Geothermal":80,"TTES":30}
    df  = compute(gen)
    print(df.to_string())
    print("CSV →", export_csv(df))
    print("PNG →", plot(df))
    test()
