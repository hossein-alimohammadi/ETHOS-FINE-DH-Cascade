"""
exergy.py — T1-T5 Exergy Analysis
===================================
Indicators:
  T1  Exergy efficiency (η_ex) per technology
  T2  Exergy destruction ratio (y_D) per technology
  T3  Fuel-product exergy diagram (Sankey inputs)
  T4  Thermo-economic cost (cF, cP) in €/GJ_ex
  T5  Exergy-based CO2 allocation

Theory
------
  Ex_fuel   = Q_fuel × (1 − T0/T_H)          [karnot factor, supply temperature T_H]
  Ex_product = Q_heat × (1 − T0/T_supply)     [useful exergy delivered]
  η_ex      = Ex_product / Ex_fuel             [exergy efficiency, 0-1]
  y_D       = Ex_dest / Ex_fuel_total          [exergy destruction ratio, 0-1]
  cP        = cF / η_ex                        [thermo-economic cost of product]

References:
  Bejan, A. et al. (1996) Thermal Design and Optimization. Wiley.
  Lazzaretto & Tsatsaronis (2006) SPECO method. Energy 31(8), 1257-1289.
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
sys.path.insert(0, str(_here.parent / "01_Core_Model"))
from paths import FIGURES_DIR, TABLES_DIR, ensure_dirs

# ── Physical constants ────────────────────────────────────────────────────────
T0 = 273.15 + 5.0       # dead-state temperature [K], annual mean Estonia
T_SUPPLY = {             # supply temperature [K] per technology
    "Heat pump":     273.15 + 55.0,   # LTDHN supply temp
    "Biomass boiler": 273.15 + 90.0,  # HTDHN supply temp
    "Gas boiler":    273.15 + 90.0,
    "Solar thermal": 273.15 + 50.0,
    "Geothermal":    273.15 + 60.0,
    "TTES":          273.15 + 70.0,   # thermal energy storage
}
COP = {                  # coefficient of performance (HP) or efficiency (combustion)
    "Heat pump":      3.5,
    "Biomass boiler": 0.88,
    "Gas boiler":     0.90,
    "Solar thermal":  0.42,
    "Geothermal":     5.0,
    "TTES":           0.95,
}
CO2_INTENSITY = {        # kgCO2/MWh_fuel (primary fuel)
    "Heat pump":      50.0,   # Estonian grid mix
    "Biomass boiler": 15.0,   # biogenic carbon (sustainable forestry)
    "Gas boiler":    202.0,   # natural gas
    "Solar thermal":   0.0,
    "Geothermal":      0.0,
    "TTES":            0.0,
}
FUEL_COST = {            # €/MWh_fuel
    "Heat pump":      40.0,   # electricity
    "Biomass boiler": 25.0,
    "Gas boiler":     55.0,
    "Solar thermal":   0.0,
    "Geothermal":      5.0,
    "TTES":            0.0,
}

TECHNOLOGIES = list(T_SUPPLY.keys())


# =============================================================================
def compute(generation_GWh: dict | None = None) -> pd.DataFrame:
    """
    Compute T1-T5 exergy indicators.

    Parameters
    ----------
    generation_GWh : dict  {technology: annual GWh delivered}
                    If None, uses a default illustrative dataset.

    Returns
    -------
    pd.DataFrame with columns:
        T1_eta_ex, T2_yD, T3_exfuel_GWh, T3_exprod_GWh, T3_exdest_GWh,
        T4_cF_euro_per_GJex, T4_cP_euro_per_GJex, T5_CO2_kt
    """
    if generation_GWh is None:
        generation_GWh = {t: np.random.uniform(50, 500) for t in TECHNOLOGIES}

    rows = []
    total_ex_dest = sum(
        gen * (1 - (1 - T0/T_SUPPLY[t]) * COP[t]) / COP[t] * 1e3   # MWh_ex
        for t, gen in generation_GWh.items() if t in COP
    )

    for tech in TECHNOLOGIES:
        gen = generation_GWh.get(tech, 0.0)   # GWh_heat delivered
        if gen <= 0:
            rows.append({**{k: 0 for k in [
                "T1_eta_ex","T2_yD","T3_exfuel_GWh","T3_exprod_GWh",
                "T3_exdest_GWh","T4_cF_euro_per_GJex","T4_cP_euro_per_GJex","T5_CO2_kt"]},
                "technology": tech})
            continue

        cop  = COP[tech]
        T_H  = T_SUPPLY[tech]
        kf   = 1 - T0 / T_H                  # Carnot factor of supply

        # Exergy of delivered heat (product)
        ex_prod  = gen * kf                   # GWh_ex
        # Exergy of input fuel
        ex_fuel  = gen / cop                  # GWh_ex (electricity or primary fuel)
        # Exergy destruction
        ex_dest  = ex_fuel - ex_prod          # GWh_ex

        # T1 — exergy efficiency
        eta_ex = ex_prod / ex_fuel if ex_fuel > 0 else 0.0

        # T2 — exergy destruction ratio (relative to system total)
        y_D = ex_dest / max(total_ex_dest / 1e3, 1e-9)   # −

        # T4 — thermo-economic cost
        cF_eur_MWh = FUEL_COST[tech]            # €/MWh_fuel
        cF = cF_eur_MWh / 3.6                   # €/GJ_fuel → ×1000/3600 = /3.6
        cP = cF / eta_ex if eta_ex > 0 else 0.0

        # T5 — CO2 allocation via exergy
        co2_kt = gen / cop * CO2_INTENSITY[tech] / 1e3  # kt_CO2/yr

        rows.append({
            "technology":            tech,
            "T1_eta_ex":             round(eta_ex, 4),
            "T2_yD":                 round(y_D, 4),
            "T3_exfuel_GWh":         round(ex_fuel, 2),
            "T3_exprod_GWh":         round(ex_prod, 2),
            "T3_exdest_GWh":         round(ex_dest, 2),
            "T4_cF_euro_per_GJex":   round(cF, 4),
            "T4_cP_euro_per_GJex":   round(cP, 4),
            "T5_CO2_kt":             round(co2_kt, 3),
        })

    return pd.DataFrame(rows).set_index("technology")


def export_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save exergy results to CSV."""
    ensure_dirs()
    out = path or TABLES_DIR / "sustainability_T_exergy.csv"
    df.to_csv(out)
    return out


def plot(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    """Generate exergy indicator figure (4 panels)."""
    ensure_dirs()
    out_dir = out_dir or FIGURES_DIR
    techs = df.index.tolist()
    colors = ["#2980b9","#27ae60","#c0392b","#f39c12","#8e44ad","#1abc9c"][:len(techs)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Exergy Analysis — T1-T5 Indicators\nDH_Cascade_Framework_v4", fontsize=12, fontweight="bold")

    # T1 — exergy efficiency
    ax = axes[0, 0]
    vals = df["T1_eta_ex"].values * 100
    bars = ax.bar(techs, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("Exergy efficiency η_ex (%)"); ax.set_title("T1 — Exergy Efficiency")
    ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 110)

    # T2 — exergy destruction ratio
    ax = axes[0, 1]
    vals = df["T2_yD"].values
    bars = ax.bar(techs, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.002, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Exergy destruction ratio y_D (−)"); ax.set_title("T2 — Exergy Destruction Ratio")
    ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8)

    # T3 — fuel-product-destruction waterfall
    ax = axes[1, 0]
    x = np.arange(len(techs)); w = 0.28
    ax.bar(x - w,   df["T3_exfuel_GWh"],  w, label="Ex fuel",        color="#3498db", alpha=0.85)
    ax.bar(x,       df["T3_exprod_GWh"],   w, label="Ex product",     color="#2ecc71", alpha=0.85)
    ax.bar(x + w,   df["T3_exdest_GWh"],   w, label="Ex destruction", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Exergy (GWh_ex/yr)"); ax.set_title("T3 — Fuel-Product-Destruction")
    ax.legend(fontsize=8)

    # T4/T5 — thermo-economic cost & CO2 (dual axis)
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.bar(x - 0.18, df["T4_cP_euro_per_GJex"], 0.35, color="#9b59b6", alpha=0.85, label="cP (€/GJ_ex)")
    ax2.plot(x, df["T5_CO2_kt"], "o--", color="#e74c3c", lw=1.5, label="CO₂ (kt/yr)", ms=7)
    ax.set_xticks(x); ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Thermo-econ. cost cP (€/GJ_ex)"); ax2.set_ylabel("CO₂ allocation (kt/yr)", color="#e74c3c")
    ax.set_title("T4 Thermo-economic cost  |  T5 CO₂ allocation")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "Figure_T_Exergy.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def test() -> None:
    """Unit tests for exergy module."""
    df = compute()
    assert "T1_eta_ex" in df.columns, "Missing T1_eta_ex column"
    assert (df["T1_eta_ex"] >= 0).all() and (df["T1_eta_ex"] <= 1).all(), "η_ex outside [0,1]"
    assert (df["T2_yD"] >= 0).all(), "y_D must be non-negative"
    assert (df["T3_exdest_GWh"] >= -1e-6).all(), "Ex destruction cannot be negative"
    hp = df.loc["Heat pump"]
    assert hp["T1_eta_ex"] > df.loc["Gas boiler"]["T1_eta_ex"], \
        "HP should have higher exergy efficiency than gas boiler"
    print("  ✓ exergy: all unit tests passed")


if __name__ == "__main__":
    ensure_dirs()
    gen = {"Heat pump": 400, "Biomass boiler": 300, "Gas boiler": 200,
           "Solar thermal": 50, "Geothermal": 80, "TTES": 30}
    df = compute(gen)
    print(df.to_string())
    csv_path = export_csv(df)
    fig_path = plot(df)
    print(f"CSV → {csv_path}")
    print(f"PNG → {fig_path}")
    test()
