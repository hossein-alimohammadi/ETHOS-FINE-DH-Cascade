"""
lifecycle.py — E1-E4 Life-Cycle Assessment (LCA) Indicators
=============================================================
Indicators:
  E1  Global warming potential (GWP100) — kg CO2-eq / MWh_heat
  E2  Cumulative energy demand (CED) — MJ_primary / MWh_heat
  E3  Acidification potential (AP) — g SO2-eq / MWh_heat
  E4  Land use (LU) — m² × yr / MWh_heat

Theory
------
  GWP = Σ_tech (gen_tech / COP_tech × EF_CO2_tech) / gen_total
  CED = Σ_tech (gen_tech / COP_tech × EF_CED_tech) / gen_total
  Characterisation factors follow ecoinvent 3.9 / SimaPro 9.5 defaults.

References:
  ISO 14040/14044 (2006). Environmental management — LCA.
  Frischknecht et al. (2007) ecoinvent data v2.0. Swiss Centre for LCI, Dübendorf.
  Pehnt (2006) Dynamic life cycle assessment of renewable energy technologies.
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

# ── Characterisation factors (ecoinvent 3.9, Nordic context) ─────────────────
# Per MWh of primary fuel consumed (not heat delivered)
LCA_FACTORS = pd.DataFrame({
    "technology":    ["Heat pump","Biomass boiler","Gas boiler","Solar thermal","Geothermal","TTES"],
    "COP_or_eff":   [3.5,         0.88,            0.90,         0.42,           5.0,          0.95],
    # E1 kgCO2-eq/MWh_fuel
    "E1_GWP100":    [50.0,        28.5,            202.0,        0.0,            5.0,          0.0],
    # E2 MJ/MWh_fuel (primary energy)
    "E2_CED":       [9.0,         6.5,             11.2,         1.5,            3.2,          0.5],
    # E3 g SO2-eq/MWh_fuel
    "E3_AP":        [0.35,        0.80,            0.52,         0.04,           0.06,         0.0],
    # E4 m²·yr/MWh_fuel (land transformation)
    "E4_LU":        [0.05,        4.50,            0.02,         3.80,           0.10,         0.0],
}).set_index("technology")


def compute(generation_GWh: dict | None = None) -> pd.DataFrame:
    """
    Compute E1-E4 LCA indicators per technology and system average.

    Returns pd.DataFrame (technology × indicator), last row = "System average".
    """
    if generation_GWh is None:
        generation_GWh = {t: np.random.uniform(50, 500) for t in LCA_FACTORS.index}

    rows = []
    gen_total = sum(generation_GWh.values())

    for tech in LCA_FACTORS.index:
        gen  = generation_GWh.get(tech, 0.0)   # GWh_heat
        cop  = LCA_FACTORS.loc[tech, "COP_or_eff"]
        fuel = gen / cop                          # GWh primary fuel

        e1 = LCA_FACTORS.loc[tech, "E1_GWP100"] * fuel / gen if gen > 0 else 0.0
        e2 = LCA_FACTORS.loc[tech, "E2_CED"]   * fuel / gen * 3600 if gen > 0 else 0.0  # →MJ/MWh
        e3 = LCA_FACTORS.loc[tech, "E3_AP"]    * fuel / gen if gen > 0 else 0.0
        e4 = LCA_FACTORS.loc[tech, "E4_LU"]    * fuel / gen if gen > 0 else 0.0

        rows.append({
            "technology":        tech,
            "gen_GWh":           round(gen, 1),
            "fuel_GWh":          round(fuel, 2),
            "E1_kgCO2eq_MWh":    round(e1, 3),
            "E2_MJ_MWh":         round(e2, 1),
            "E3_gSO2eq_MWh":     round(e3, 4),
            "E4_m2yr_MWh":       round(e4, 4),
        })

    df = pd.DataFrame(rows).set_index("technology")

    # System-weighted average
    w = df["gen_GWh"] / gen_total if gen_total > 0 else 1.0 / len(df)
    sys_avg = {
        "gen_GWh":        gen_total,
        "fuel_GWh":       df["fuel_GWh"].sum(),
        "E1_kgCO2eq_MWh": (df["E1_kgCO2eq_MWh"] * w).sum(),
        "E2_MJ_MWh":      (df["E2_MJ_MWh"]      * w).sum(),
        "E3_gSO2eq_MWh":  (df["E3_gSO2eq_MWh"]  * w).sum(),
        "E4_m2yr_MWh":    (df["E4_m2yr_MWh"]    * w).sum(),
    }
    df.loc["System average"] = sys_avg
    return df


def export_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_dirs()
    out = path or TABLES_DIR / "sustainability_E_lifecycle.csv"
    df.to_csv(out)
    return out


def plot(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    ensure_dirs()
    out_dir = out_dir or FIGURES_DIR
    techs  = [t for t in df.index if t != "System average"]
    colors = ["#2980b9","#27ae60","#c0392b","#f39c12","#8e44ad","#1abc9c"][:len(techs)]
    ind    = ["E1_kgCO2eq_MWh","E2_MJ_MWh","E3_gSO2eq_MWh","E4_m2yr_MWh"]
    labels = ["E1 GWP100\n(kgCO₂-eq/MWh)","E2 CED\n(MJ/MWh)","E3 AP\n(g SO₂-eq/MWh)","E4 LU\n(m²·yr/MWh)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Life-Cycle Assessment — E1-E4 Indicators\nDH_Cascade_Framework_v4", fontsize=12, fontweight="bold")

    for ax, col, label in zip(axes.flat, ind, labels):
        vals  = df.loc[techs, col].values
        bars  = ax.bar(techs, vals, color=colors, edgecolor="white")
        avg_v = df.loc["System average", col]
        ax.axhline(avg_v, color="black", lw=1.5, ls="--", label=f"Sys. avg: {avg_v:.2f}")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v*1.02, f"{v:.2f}",
                    ha="center", fontsize=7.5)
        ax.set_ylabel(label); ax.set_title(label.split("\n")[0])
        ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "Figure_E_Lifecycle.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def test() -> None:
    df = compute()
    assert "E1_kgCO2eq_MWh" in df.columns
    assert "System average" in df.index
    hp = df.loc["Heat pump", "E1_kgCO2eq_MWh"]
    gas = df.loc["Gas boiler", "E1_kgCO2eq_MWh"]
    assert hp < gas, "HP GWP must be lower than gas boiler"
    print("  ✓ lifecycle: all unit tests passed")


if __name__ == "__main__":
    ensure_dirs()
    gen = {"Heat pump":400,"Biomass boiler":300,"Gas boiler":200,"Solar thermal":50,"Geothermal":80,"TTES":30}
    df  = compute(gen)
    print(df.to_string())
    print("CSV →", export_csv(df))
    print("PNG →", plot(df))
    test()
