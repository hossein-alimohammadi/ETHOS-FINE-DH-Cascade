"""
flexibility.py — F1-F3 Flexibility Indicators
===============================================
Indicators:
  F1  Demand flexibility potential (DFP) — GWh shiftable per day
  F2  Ramp rate capability — MW/min average across fleet
  F3  Storage utilisation index (SUI) — ratio of used storage capacity

Theory
------
  DFP  = Σ_tech (P_flex_tech × Δt_shift_h)    [GWh]
         where P_flex_tech is the power that can be shifted up/down ± ΔT
  RR   = Σ_tech (P_installed_tech × ramp_pct_per_min)  [MW/min]
  SUI  = E_cycled / E_capacity_total               [−]

References:
  Lund et al. (2015) Review of energy system flexibility measures and enabling technologies.
  Bloess et al. (2018) Power-to-heat for flexible energy systems. Applied Energy 212.
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

# ── Technology flexibility parameters ─────────────────────────────────────────
FLEX_PARAMS = pd.DataFrame({
    "technology":       ["Heat pump","Biomass boiler","Gas boiler","Solar thermal","Geothermal","TTES"],
    "flex_fraction":    [0.40,        0.10,            0.35,         0.05,           0.05,         0.90],
    "shift_hours":      [4.0,         2.0,             3.0,          1.0,            1.0,         12.0],
    "ramp_pct_per_min": [0.50,        0.15,            0.40,         0.01,           0.05,         1.00],
    "stor_fraction":    [0.10,        0.00,            0.00,         0.00,           0.00,         1.00],
}).set_index("technology")


def compute(generation_GWh: dict | None = None,
            capacity_MW: dict | None = None,
            tes_capacity_GWh: float = 2.0,
            tes_cycled_GWh: float = 1.2) -> pd.DataFrame:
    """
    Compute F1-F3 flexibility indicators.

    Returns pd.DataFrame (technology × F1..F3)
    """
    rng = np.random.default_rng(0)
    if generation_GWh is None:
        generation_GWh = {t: rng.uniform(50,500) for t in FLEX_PARAMS.index}
    if capacity_MW is None:
        capacity_MW = {t: g*1e3/8760*2.5 for t, g in generation_GWh.items()}

    rows = []
    for tech in FLEX_PARAMS.index:
        gen = generation_GWh.get(tech, 0.0)
        cap = capacity_MW.get(tech, 0.0)
        fp  = FLEX_PARAMS.loc[tech]

        # F1 — demand flexibility potential [GWh/day]
        f1 = cap * fp["flex_fraction"] * fp["shift_hours"] / 1e3  # GWh/day

        # F2 — ramp rate [MW/min]
        f2 = cap * fp["ramp_pct_per_min"] / 100.0

        # F3 — storage utilisation
        f3 = tes_cycled_GWh / max(tes_capacity_GWh, 0.001) if fp["stor_fraction"] > 0 else 0.0

        rows.append({
            "technology":    tech,
            "gen_GWh":       round(gen, 1),
            "cap_MW":        round(cap, 1),
            "F1_DFP_GWh_d":  round(f1, 3),
            "F2_ramp_MW_min":round(f2, 3),
            "F3_SUI":        round(f3, 4),
        })

    df = pd.DataFrame(rows).set_index("technology")
    df.loc["TOTAL", "F1_DFP_GWh_d"]  = round(df["F1_DFP_GWh_d"].sum(), 3)
    df.loc["TOTAL", "F2_ramp_MW_min"] = round(df["F2_ramp_MW_min"].sum(), 3)
    df.loc["TOTAL", "F3_SUI"]         = round(tes_cycled_GWh / max(tes_capacity_GWh, 0.001), 4)
    return df


def export_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_dirs()
    out = path or TABLES_DIR / "sustainability_F_flexibility.csv"
    df.to_csv(out)
    return out


def plot(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    ensure_dirs()
    out_dir = out_dir or FIGURES_DIR
    tech_rows = df.drop("TOTAL", errors="ignore")
    techs  = tech_rows.index.tolist()
    colors = ["#2980b9","#27ae60","#c0392b","#f39c12","#8e44ad","#1abc9c"][:len(techs)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Flexibility Analysis — F1-F3 Indicators\nDH_Cascade_Framework_v4", fontsize=12, fontweight="bold")

    for ax, col, ylabel, title in [
        (axes[0], "F1_DFP_GWh_d",   "GWh/day",  "F1 — Demand Flexibility Potential"),
        (axes[1], "F2_ramp_MW_min",  "MW/min",   "F2 — Ramp Rate Capability"),
        (axes[2], "F3_SUI",          "SUI (−)",  "F3 — Storage Utilisation Index"),
    ]:
        vals = tech_rows[col].values
        bars = ax.bar(techs, vals, color=colors, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, v*1.03, f"{v:.3f}", ha="center", fontsize=7.5)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8)

    axes[2].axhline(0.8, color="black", ls="--", lw=1.2, label="High util. target (0.8)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "Figure_F_Flexibility.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def test() -> None:
    df = compute()
    assert "F1_DFP_GWh_d" in df.columns
    assert "TOTAL" in df.index
    assert (df.loc[df.index!="TOTAL","F1_DFP_GWh_d"] >= 0).all()
    assert (df.loc["TOTAL","F1_DFP_GWh_d"] > 0)
    print("  ✓ flexibility: all unit tests passed")


if __name__ == "__main__":
    ensure_dirs()
    gen = {"Heat pump":400,"Biomass boiler":300,"Gas boiler":200,"Solar thermal":50,"Geothermal":80,"TTES":30}
    df  = compute(gen)
    print(df.to_string())
    print("CSV →", export_csv(df))
    print("PNG →", plot(df))
    test()
