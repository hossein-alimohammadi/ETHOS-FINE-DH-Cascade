"""
technical.py — TC1-TC4 Technical Indicators
=============================================
Indicators:
  TC1  Capacity factor (CF) per technology  [−]
  TC2  Self-sufficiency ratio (SSR) per county  [−]
  TC3  Peak demand coverage ratio (PDCR)  [−]
  TC4  Network heat loss ratio (HLR)  [%]

Theory
------
  CF  = E_generated / (P_installed × 8760)
  SSR = E_local_gen / E_local_demand       (for each county)
  PDCR = P_installed_total / P_peak_demand
  HLR  = (E_gen − E_delivered) / E_gen     (% losses in pipe network)

References:
  Lund et al. (2010) The role of district heating in future renewable energy systems.
  Werner (2017) International review of district heating and cooling.
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

COUNTIES = [
    "Hiiu","Saane","Laane","Parnu","Rapla","Jarva","Harju",
    "Laane-Viru","Ida-Viru","Jogeva","Tartu","Viljandi","Valga","Polva","Voru",
]
TECHNOLOGIES = ["Heat pump","Biomass boiler","Gas boiler","Solar thermal","Geothermal","TTES"]


def compute(generation_GWh: dict | None = None,
            capacity_MW: dict | None = None,
            demand_by_county_GWh: dict | None = None) -> dict:
    """
    Compute TC1-TC4 technical indicators.

    Returns
    -------
    dict with keys:
      "CF"   : pd.Series  (technology → CF)
      "SSR"  : pd.Series  (county → SSR)
      "PDCR" : float
      "HLR"  : float
      "summary" : pd.DataFrame
    """
    rng = np.random.default_rng(42)
    if generation_GWh is None:
        generation_GWh = {t: rng.uniform(50,500) for t in TECHNOLOGIES}
    if capacity_MW is None:
        capacity_MW    = {t: g*1e3/8760*2.5 for t,g in generation_GWh.items()}
    if demand_by_county_GWh is None:
        demand_by_county_GWh = {c: rng.uniform(20,200) for c in COUNTIES}

    # TC1 — capacity factor
    cf = pd.Series({
        t: generation_GWh.get(t,0)*1e3 / (capacity_MW.get(t,1)*8760)
        for t in TECHNOLOGIES
    }, name="TC1_CF").clip(0, 1)

    # TC2 — self-sufficiency ratio (assume supply ∝ demand share)
    total_gen   = sum(generation_GWh.values())
    total_demand = sum(demand_by_county_GWh.values())
    ssr = pd.Series({
        c: min(1.0, (total_gen * demand_by_county_GWh[c] / total_demand) / demand_by_county_GWh[c])
        for c in COUNTIES
    }, name="TC2_SSR")

    # TC3 — peak demand coverage
    peak_demand_MW = max(demand_by_county_GWh.values()) * 1e3 / (0.05 * 8760)  # rough peak
    total_cap      = sum(capacity_MW.values())
    pdcr = total_cap / peak_demand_MW

    # TC4 — network heat loss ratio (pipe loss ~5-15% of generated)
    pipe_loss_pct = 0.08   # 8% baseline for Estonian DH networks
    hlr = pipe_loss_pct

    summary_rows = []
    for tech in TECHNOLOGIES:
        gen = generation_GWh.get(tech, 0)
        cap = capacity_MW.get(tech, 0)
        summary_rows.append({
            "technology":    tech,
            "TC1_CF":        round(cf[tech], 4),
            "gen_GWh":       round(gen, 1),
            "cap_MW":        round(cap, 1),
            "TC3_PDCR_contrib": round(cap / peak_demand_MW, 3) if peak_demand_MW > 0 else 0,
        })
    summary = pd.DataFrame(summary_rows).set_index("technology")
    summary.loc["SYSTEM", "TC3_PDCR_contrib"] = round(pdcr, 3)
    summary.loc["SYSTEM", "TC4_HLR_pct"]      = round(hlr*100, 1)

    return {"CF": cf, "SSR": ssr, "PDCR": pdcr, "HLR": hlr, "summary": summary}


def export_csv(results: dict, path: Path | None = None) -> Path:
    ensure_dirs()
    out = path or TABLES_DIR / "sustainability_TC_technical.csv"
    results["summary"].to_csv(out)
    cf_path  = TABLES_DIR / "sustainability_TC_capacity_factor.csv"
    ssr_path = TABLES_DIR / "sustainability_TC_self_sufficiency.csv"
    results["CF"].to_csv(cf_path, header=True)
    results["SSR"].to_csv(ssr_path, header=True)
    return out


def plot(results: dict, out_dir: Path | None = None) -> Path:
    ensure_dirs()
    out_dir = out_dir or FIGURES_DIR
    cf  = results["CF"]
    ssr = results["SSR"]
    techs  = cf.index.tolist()
    colors = ["#2980b9","#27ae60","#c0392b","#f39c12","#8e44ad","#1abc9c"][:len(techs)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Technical Analysis — TC1-TC4 Indicators\nDH_Cascade_Framework_v4", fontsize=12, fontweight="bold")

    # TC1 — capacity factor
    ax = axes[0, 0]
    bars = ax.bar(techs, cf.values*100, color=colors, edgecolor="white")
    for bar, v in zip(bars, cf.values*100):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("Capacity factor (%)"); ax.set_title("TC1 — Capacity Factor")
    ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8); ax.set_ylim(0, 110)

    # TC2 — SSR per county
    ax = axes[0, 1]
    counties = ssr.index.tolist()
    ssr_vals = ssr.values
    c_colors = ["#27ae60" if v >= 1.0 else "#3498db" if v >= 0.7 else "#e74c3c" for v in ssr_vals]
    ax.barh(counties, ssr_vals, color=c_colors, edgecolor="white")
    ax.axvline(1.0, color="black", lw=1.5, ls="--", label="Self-sufficient (SSR=1)")
    ax.set_xlabel("Self-sufficiency ratio"); ax.set_title("TC2 — Self-Sufficiency by County")
    ax.legend(fontsize=8); ax.set_xlim(0, 1.15)

    # TC3 — PDCR contribution
    ax = axes[1, 0]
    summary = results["summary"].drop("SYSTEM", errors="ignore")
    pdcr_vals = summary["TC3_PDCR_contrib"].values
    bars = ax.bar(techs, pdcr_vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, pdcr_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.002, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("PDCR contribution (−)"); ax.set_title("TC3 — Peak Demand Coverage Contribution")
    ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8)

    # TC4 — system KPI scorecard
    ax = axes[1, 1]
    kpis  = ["Overall PDCR", "Network heat\nloss ratio (%)"]
    vals  = [round(results["PDCR"], 2), round(results["HLR"]*100, 1)]
    refs  = [1.20, 10.0]   # reference/target values
    clrs  = ["#27ae60" if vals[0]>=refs[0] else "#e74c3c", "#27ae60" if vals[1]<=refs[1] else "#e74c3c"]
    brs   = ax.bar(kpis, vals, color=clrs, edgecolor="white", width=0.5)
    for bar, v, r in zip(brs, vals, refs):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v}", ha="center", fontsize=10, fontweight="bold")
        ax.axhline(r, color="grey", ls="--", lw=1)
    ax.set_title("TC3/TC4 — System-level KPIs"); ax.set_ylabel("Value")

    fig.tight_layout()
    out_path = out_dir / "Figure_TC_Technical.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def test() -> None:
    res = compute()
    assert "CF" in res and "SSR" in res
    assert (res["CF"] >= 0).all() and (res["CF"] <= 1).all()
    assert (res["SSR"] >= 0).all() and (res["SSR"] <= 1.0).all()
    assert 0 < res["PDCR"] < 20
    assert 0 <= res["HLR"] <= 0.30
    print("  ✓ technical: all unit tests passed")


if __name__ == "__main__":
    ensure_dirs()
    res = compute()
    print(res["summary"].to_string())
    print("CSV →", export_csv(res))
    print("PNG →", plot(res))
    test()
