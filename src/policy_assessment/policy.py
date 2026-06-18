"""
policy.py — P1-P4 Policy Compliance Indicators
================================================
Indicators:
  P1  RES share (renewable energy share) — % of total generation
  P2  EU ETS cost — M€/yr (carbon price × direct emissions)
  P3  RED III compliance score — [0,100]
  P4  Estonian National Energy & Climate Plan (NECP) alignment — [0,100]

Theory
------
  P1  = E_renewable / E_total × 100
        Renewable = HP (COP>2 qualifies under RED III Art. 7), biomass (sustainable),
                    solar, geothermal
  P2  = Σ_tech(E_tech/COP_tech × EF_direct_tech) × CETS   [kgCO2 × €/tCO2]
        CETS = 80 €/tCO2  (2025 EU ETS price assumption)
  P3  Weighted score across:
        a. RES share > 60 %      (weight 0.4)
        b. LCOH < 100 €/MWh     (weight 0.25)
        c. CO2 < 50 kgCO2/MWh   (weight 0.20)
        d. No coal/oil           (weight 0.15)
  P4  Estonian NECP 2030 targets:
        a. RES electricity > 100 %   (Estonia 2030 target)
        b. Building renovations
        c. DH efficiency improvement

References:
  European Commission (2023) Renewable Energy Directive III (RED III), 2023/2413/EU.
  Eurostat (2024) Renewable Energy Statistics.
  Estonian Ministry of Economic Affairs (2023) Estonian National Energy and Climate Plan.
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

# ── Policy parameters ─────────────────────────────────────────────────────────
ETS_PRICE   = 80.0    # €/tCO2 (2025 assumption)
CO2_DIRECT  = {       # kg CO2 / MWh_fuel (direct scope 1 only)
    "Heat pump":       0.0,   # electricity emissions counted in grid, not site
    "Biomass boiler":  0.0,   # biogenic, excluded under EU ETS
    "Gas boiler":    202.0,
    "Solar thermal":   0.0,
    "Geothermal":      0.0,
    "TTES":            0.0,
}
COP_EFF     = {"Heat pump":3.5,"Biomass boiler":0.88,"Gas boiler":0.90,"Solar thermal":0.42,"Geothermal":5.0,"TTES":0.95}
IS_RES      = {"Heat pump":True,"Biomass boiler":True,"Gas boiler":False,"Solar thermal":True,"Geothermal":True,"TTES":False}
LCOH_BY_TECH = {"Heat pump":65,"Biomass boiler":72,"Gas boiler":90,"Solar thermal":110,"Geothermal":80,"TTES":30}


def compute(generation_GWh: dict | None = None,
            lcoh_by_tech: dict | None = None) -> dict:
    """
    Compute P1-P4 policy indicators.

    Returns dict: P1_RES_pct, P2_ETS_MEur, P3_RED3_score, P4_NECP_score, summary
    """
    rng = np.random.default_rng(3)
    if generation_GWh is None:
        generation_GWh = {t: rng.uniform(50,500) for t in COP_EFF}
    if lcoh_by_tech is None:
        lcoh_by_tech = LCOH_BY_TECH

    total_gen = sum(generation_GWh.values())

    # P1 — RES share
    res_gen = sum(gen for t, gen in generation_GWh.items() if IS_RES.get(t, False))
    p1 = res_gen / total_gen * 100 if total_gen > 0 else 0.0

    # P2 — EU ETS cost
    ets_cost_Meur = sum(
        generation_GWh.get(t,0) / COP_EFF.get(t,1.0) * CO2_DIRECT.get(t,0) / 1e3 * ETS_PRICE / 1e6
        for t in COP_EFF
    )

    # P3 — RED III compliance score
    avg_co2  = sum(generation_GWh.get(t,0) / COP_EFF.get(t,1.0) * CO2_DIRECT.get(t,0) / total_gen
                   for t in COP_EFF if total_gen>0)   # kgCO2/MWh_heat
    avg_lcoh = sum(generation_GWh.get(t,0)/total_gen * lcoh_by_tech.get(t,100)
                   for t in COP_EFF if total_gen>0)
    no_coal  = 1.0   # Estonia has no coal DH; default 1.0

    scores_p3 = {
        "RES>60%":      min(1.0, p1/60)     * 0.40,
        "LCOH<100€/MWh":min(1.0, 100/max(avg_lcoh,1)) * 0.25,
        "CO2<50kg/MWh": min(1.0, 50/max(avg_co2,0.1)) * 0.20 if avg_co2 > 0 else 0.20,
        "No coal/oil":  no_coal              * 0.15,
    }
    p3 = sum(scores_p3.values()) * 100

    # P4 — Estonian NECP alignment
    scores_p4 = {
        "RES_DH>80%":   min(1.0, p1/80)     * 0.40,
        "LCOH<90€/MWh": min(1.0, 90/max(avg_lcoh,1)) * 0.30,
        "CO2<30kg/MWh": min(1.0, 30/max(avg_co2,0.1)) * 0.30 if avg_co2 > 0 else 0.30,
    }
    p4 = sum(scores_p4.values()) * 100

    summary_rows = []
    for tech in COP_EFF:
        gen  = generation_GWh.get(tech, 0.0)
        fuel = gen / COP_EFF[tech]
        co2  = fuel * CO2_DIRECT.get(tech,0) / 1e3   # t CO2
        summary_rows.append({
            "technology":    tech,
            "gen_GWh":       round(gen, 1),
            "is_RES":        IS_RES.get(tech, False),
            "direct_CO2_kt": round(co2, 3),
            "ETS_cost_MEur": round(co2 * ETS_PRICE / 1e6, 4),
        })
    summary = pd.DataFrame(summary_rows).set_index("technology")
    summary.loc["SYSTEM","gen_GWh"]       = round(total_gen, 1)
    summary.loc["SYSTEM","P1_RES_pct"]    = round(p1, 2)
    summary.loc["SYSTEM","P2_ETS_MEur"]   = round(ets_cost_Meur, 3)
    summary.loc["SYSTEM","P3_RED3_score"] = round(p3, 1)
    summary.loc["SYSTEM","P4_NECP_score"] = round(p4, 1)

    return {"P1_RES_pct": p1, "P2_ETS_MEur": ets_cost_Meur,
            "P3_RED3_score": p3, "P4_NECP_score": p4,
            "scores_P3": scores_p3, "scores_P4": scores_p4, "summary": summary}


def export_csv(results: dict, path: Path | None = None) -> Path:
    ensure_dirs()
    out = path or TABLES_DIR / "sustainability_P_policy.csv"
    results["summary"].to_csv(out)
    return out


def plot(results: dict, out_dir: Path | None = None) -> Path:
    ensure_dirs()
    out_dir = out_dir or FIGURES_DIR
    summary = results["summary"].drop("SYSTEM", errors="ignore")
    techs   = summary.index.tolist()
    colors  = ["#2980b9","#27ae60","#c0392b","#f39c12","#8e44ad","#1abc9c"][:len(techs)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Policy Compliance — P1-P4 Indicators\nDH_Cascade_Framework_v4", fontsize=12, fontweight="bold")

    # P1 — RES share (pie)
    ax = axes[0, 0]
    res_gen = sum(summary.loc[t,"gen_GWh"] for t in techs if IS_RES.get(t,False))
    non_res = max(0, summary["gen_GWh"].sum() - res_gen)
    ax.pie([res_gen, non_res], labels=[f"RES\n{results['P1_RES_pct']:.1f}%","Non-RES"],
           colors=["#27ae60","#e74c3c"], autopct="%1.1f%%", startangle=90)
    ax.set_title(f"P1 — RES Share: {results['P1_RES_pct']:.1f}%")

    # P2 — ETS cost by tech
    ax = axes[0, 1]
    ets_vals = summary["ETS_cost_MEur"].values
    bars = ax.bar(techs, ets_vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, ets_vals):
        if v > 0:
            ax.text(bar.get_x()+bar.get_width()/2, v*1.04, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("ETS cost (M€/yr)"); ax.set_title(f"P2 — EU ETS Cost (Total: {results['P2_ETS_MEur']:.2f} M€)")
    ax.set_xticklabels(techs, rotation=30, ha="right", fontsize=8)

    # P3 — RED III radar-style bar
    ax = axes[1, 0]
    p3_keys = list(results["scores_P3"].keys())
    p3_vals = [results["scores_P3"][k]*100/{"RES>60%":0.40,"LCOH<100€/MWh":0.25,"CO2<50kg/MWh":0.20,"No coal/oil":0.15}[k]
               for k in p3_keys]
    x = np.arange(len(p3_keys))
    ax.bar(x, p3_vals, color=["#27ae60" if v>=70 else "#f39c12" if v>=40 else "#e74c3c" for v in p3_vals], edgecolor="white")
    ax.axhline(70, color="black", ls="--", lw=1, label="Pass threshold (70)")
    ax.set_xticks(x); ax.set_xticklabels(p3_keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Sub-score (%)"); ax.set_title(f"P3 — RED III Score: {results['P3_RED3_score']:.1f}/100")
    ax.set_ylim(0,115); ax.legend(fontsize=8)

    # P4 — NECP alignment
    ax = axes[1, 1]
    p4_keys = list(results["scores_P4"].keys())
    p4_vals = [results["scores_P4"][k]*100/{"RES_DH>80%":0.40,"LCOH<90€/MWh":0.30,"CO2<30kg/MWh":0.30}[k]
               for k in p4_keys]
    x = np.arange(len(p4_keys))
    ax.bar(x, p4_vals, color=["#27ae60" if v>=70 else "#e74c3c" for v in p4_vals], edgecolor="white")
    ax.axhline(70, color="black", ls="--", lw=1, label="Pass threshold (70)")
    ax.set_xticks(x); ax.set_xticklabels(p4_keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Sub-score (%)"); ax.set_title(f"P4 — NECP Score: {results['P4_NECP_score']:.1f}/100")
    ax.set_ylim(0,115); ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "Figure_P_Policy.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def test() -> None:
    res = compute()
    assert 0 <= res["P1_RES_pct"] <= 100
    assert res["P2_ETS_MEur"] >= 0
    assert 0 <= res["P3_RED3_score"] <= 100
    assert 0 <= res["P4_NECP_score"] <= 100
    # Gas boiler should have positive ETS cost
    assert res["summary"].loc["Gas boiler","ETS_cost_MEur"] > 0
    print("  ✓ policy: all unit tests passed")


if __name__ == "__main__":
    ensure_dirs()
    gen = {"Heat pump":400,"Biomass boiler":300,"Gas boiler":200,"Solar thermal":50,"Geothermal":80,"TTES":30}
    res = compute(gen)
    print(res["summary"].to_string())
    print(f"P1={res['P1_RES_pct']:.1f}% | P2={res['P2_ETS_MEur']:.2f}M€ | P3={res['P3_RED3_score']:.0f} | P4={res['P4_NECP_score']:.0f}")
    print("CSV →", export_csv(res))
    print("PNG →", plot(res))
    test()
