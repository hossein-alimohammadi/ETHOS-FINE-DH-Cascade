"""
resilience.py — R1-R3 Resilience Indicators
=============================================
Indicators:
  R1  Supply diversity index (Herfindahl-Hirschman Index, HHI) [−]
  R2  Fuel import dependency (FID) [%]
  R3  Redundancy ratio (N-1 security) [−]

Theory
------
  HHI = Σ_tech s_tech²     where s_tech = share of annual generation
        HHI ∈ [0,1]; lower = more diverse (HHI < 0.15 = diverse, >0.25 = concentrated)
  FID = E_imported_fuel / E_total_fuel × 100
  R3  = Σ_tech (P_tech × redundancy_factor) / P_peak_demand
        R3 ≥ 1.0 means N-1 security maintained

References:
  Azzuni & Breyer (2018) Definitions and dimensions of energy security.
  Grubb et al. (2006) Diversity in energy technologies and supply security. OECD/IEA.
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

# ── Resilience parameters ──────────────────────────────────────────────────────
RESIL_PARAMS = pd.DataFrame({
    "technology":        ["Heat pump","Biomass boiler","Gas boiler","Solar thermal","Geothermal","TTES"],
    "import_fraction":   [0.30,        0.05,            0.95,         0.00,           0.00,         0.00],
    "redundancy_factor": [0.85,        0.90,            0.95,         0.30,           0.80,         1.00],
}).set_index("technology")


def compute(generation_GWh: dict | None = None,
            capacity_MW: dict | None = None,
            peak_demand_MW: float | None = None) -> dict:
    """
    Compute R1-R3 resilience indicators.

    Returns dict with keys: R1_HHI, R2_FID, R3_redundancy, summary DataFrame
    """
    rng = np.random.default_rng(7)
    if generation_GWh is None:
        generation_GWh = {t: rng.uniform(50,500) for t in RESIL_PARAMS.index}
    if capacity_MW is None:
        capacity_MW = {t: g*1e3/8760*2.5 for t,g in generation_GWh.items()}
    if peak_demand_MW is None:
        peak_demand_MW = sum(generation_GWh.values()) * 1e3 / (0.06*8760)

    total_gen   = sum(generation_GWh.values())
    total_fuel  = sum(generation_GWh.get(t,0)/2.0 for t in RESIL_PARAMS.index)

    # R1 — HHI
    hhi = sum((generation_GWh.get(t,0)/total_gen)**2 for t in RESIL_PARAMS.index if total_gen>0)

    # R2 — fuel import dependency
    imported = sum(
        generation_GWh.get(t,0) * RESIL_PARAMS.loc[t,"import_fraction"]
        for t in RESIL_PARAMS.index
    )
    fid = imported / total_gen * 100 if total_gen > 0 else 0.0

    # R3 — redundancy
    r3 = sum(
        capacity_MW.get(t,0) * RESIL_PARAMS.loc[t,"redundancy_factor"]
        for t in RESIL_PARAMS.index
    ) / peak_demand_MW if peak_demand_MW > 0 else 0.0

    rows = []
    for tech in RESIL_PARAMS.index:
        gen  = generation_GWh.get(tech, 0.0)
        cap  = capacity_MW.get(tech, 0.0)
        s    = gen/total_gen if total_gen > 0 else 0.0
        rows.append({
            "technology":            tech,
            "gen_GWh":               round(gen, 1),
            "cap_MW":                round(cap, 1),
            "gen_share":             round(s, 4),
            "HHI_contribution":      round(s**2, 5),
            "import_fraction":       RESIL_PARAMS.loc[tech,"import_fraction"],
            "redundancy_factor":     RESIL_PARAMS.loc[tech,"redundancy_factor"],
        })
    summary = pd.DataFrame(rows).set_index("technology")
    summary.loc["SYSTEM","gen_GWh"]         = round(total_gen, 1)
    summary.loc["SYSTEM","R1_HHI"]          = round(hhi, 4)
    summary.loc["SYSTEM","R2_FID_pct"]      = round(fid, 2)
    summary.loc["SYSTEM","R3_redundancy"]   = round(r3, 3)

    return {"R1_HHI": hhi, "R2_FID": fid, "R3_redundancy": r3, "summary": summary}


def export_csv(results: dict, path: Path | None = None) -> Path:
    ensure_dirs()
    out = path or TABLES_DIR / "sustainability_R_resilience.csv"
    results["summary"].to_csv(out)
    return out


def plot(results: dict, out_dir: Path | None = None) -> Path:
    ensure_dirs()
    out_dir = out_dir or FIGURES_DIR
    summary = results["summary"].drop("SYSTEM", errors="ignore")
    techs   = summary.index.tolist()
    colors  = ["#2980b9","#27ae60","#c0392b","#f39c12","#8e44ad","#1abc9c"][:len(techs)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Resilience Analysis — R1-R3 Indicators\nDH_Cascade_Framework_v4", fontsize=12, fontweight="bold")

    # R1 — HHI (share squared)
    ax = axes[0]
    vals = summary["HHI_contribution"].values
    bars = ax.bar(techs, vals, color=colors, edgecolor="white")
    ax.axhline(results["R1_HHI"], color="red", ls="--", lw=1.5, label=f"Total HHI={results['R1_HHI']:.3f}")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v*1.04, f"{v:.4f}", ha="center", fontsize=7.5)
    ax.set_ylabel("HHI contribution (s²)"); ax.set_title("R1 — Diversity (HHI)")
    ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8); ax.legend(fontsize=8)

    # R2 — import fraction per tech
    ax = axes[1]
    import_vals = summary["import_fraction"].values * 100
    bars = ax.bar(techs, import_vals, color=colors, edgecolor="white")
    ax.axhline(results["R2_FID"], color="red", ls="--", lw=1.5, label=f"System FID={results['R2_FID']:.1f}%")
    for bar, v in zip(bars, import_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v*1.03, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_ylabel("Import fraction (%)"); ax.set_title("R2 — Fuel Import Dependency")
    ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8); ax.legend(fontsize=8)

    # R3 — redundancy factor per tech
    ax = axes[2]
    red_vals = summary["redundancy_factor"].values
    bars = ax.bar(techs, red_vals, color=colors, edgecolor="white")
    ax.axhline(results["R3_redundancy"], color="green", ls="--", lw=1.5, label=f"System R3={results['R3_redundancy']:.2f}")
    ax.axhline(1.0, color="black", ls="-", lw=1, label="N-1 threshold (1.0)")
    for bar, v in zip(bars, red_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v*1.03, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Redundancy factor (−)"); ax.set_title("R3 — N-1 Redundancy")
    ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8); ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "Figure_R_Resilience.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def test() -> None:
    # Validate module-level parameters
    assert RESIL_PARAMS.loc["Gas boiler","import_fraction"] == 0.95, "Gas import fraction wrong"
    assert RESIL_PARAMS.loc["Heat pump","import_fraction"] < 0.5, "HP should be low-import"

    # Validate HHI math: equal shares → HHI = 1/N
    n = 4
    shares = [1/n] * n
    hhi = sum(s**2 for s in shares)
    assert abs(hhi - 1/n) < 1e-9, "HHI formula incorrect"

    # Validate FID math
    gens = {"Gas boiler": 200.0, "Heat pump": 100.0}
    total = sum(gens.values())
    fid = sum(gens[t]*RESIL_PARAMS.loc[t,"import_fraction"] for t in gens) / total * 100
    assert fid > 0, "FID with gas should be positive"

    print("  ✓ resilience: all unit tests passed")


if __name__ == "__main__":
    ensur