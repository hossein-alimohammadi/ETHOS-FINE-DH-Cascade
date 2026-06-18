"""
mtoi.py — MTOI (Multi-Technology Optimisation Interface) Platform Adapter
==========================================================================
The MTOI module acts as a platform adapter that:
  1. Aggregates all sustainability indicator modules
  2. Computes a composite Sustainability Performance Index (SPI)
  3. Generates a sustainability dashboard figure
  4. Writes a combined CSV export

Composite SPI formula:
  SPI = Σ_dim w_dim × Score_dim(0−100)
  Dimensions:
    T  Exergy efficiency (η_ex average)            w=0.15
    E  LCA GWP100 (inverse — lower is better)      w=0.20
    C  LCOH (inverse)                              w=0.20
    TC Capacity factor average                     w=0.10
    F  Demand flexibility potential                w=0.10
    R  Supply diversity (1−HHI, rescaled)          w=0.10
    P  Policy compliance (RED III score)           w=0.15

References:
  Nilsson & Griggs (2016) Map the interactions between Sustainable Development Goals.
  Lynskey et al. (2021) Multi-dimensional energy sustainability assessment.
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))
from paths import FIGURES_DIR, TABLES_DIR, REPORTS_DIR, ensure_dirs

# ── Import all sustainability modules ─────────────────────────────────────────
import importlib, sys as _sys
def _load(name):
    spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

exergy     = _load("exergy")
lifecycle  = _load("lifecycle")
economic   = _load("economic")
technical  = _load("technical")
flexibility= _load("flexibility")
resilience = _load("resilience")
policy     = _load("policy")
mcda_mod   = _load("mcda")

# ── SPI weights ───────────────────────────────────────────────────────────────
SPI_WEIGHTS = {"T":0.15, "E":0.20, "C":0.20, "TC":0.10, "F":0.10, "R":0.10, "P":0.15}
SPI_LABELS  = {
    "T":  "Exergy",
    "E":  "LCA",
    "C":  "Economic",
    "TC": "Technical",
    "F":  "Flexibility",
    "R":  "Resilience",
    "P":  "Policy",
}


def compute(generation_GWh: dict | None = None,
            capacity_MW: dict | None = None) -> dict:
    """
    Aggregate all sustainability indicators and compute SPI.

    Returns dict with all module results + SPI breakdown.
    """
    rng = np.random.default_rng(99)
    if generation_GWh is None:
        generation_GWh = {
            "Heat pump":400,"Biomass boiler":300,"Gas boiler":200,
            "Solar thermal":50,"Geothermal":80,"TTES":30
        }
    if capacity_MW is None:
        capacity_MW = {t: g*1e3/8760*2.5 for t,g in generation_GWh.items()}

    print("  Running exergy module...")
    res_T  = exergy.compute(generation_GWh)
    print("  Running lifecycle module...")
    res_E  = lifecycle.compute(generation_GWh)
    print("  Running economic module...")
    res_C  = economic.compute(generation_GWh, capacity_MW)
    print("  Running technical module...")
    res_TC = technical.compute(generation_GWh, capacity_MW)
    print("  Running flexibility module...")
    res_F  = flexibility.compute(generation_GWh, capacity_MW)
    print("  Running resilience module...")
    res_R  = resilience.compute(generation_GWh, capacity_MW)
    print("  Running policy module...")
    res_P  = policy.compute(generation_GWh)
    print("  Running MCDA module...")
    res_MCDA = mcda_mod.compute()

    # ── Score each dimension on [0,100] ──────────────────────────────────────
    # T — average exergy efficiency (%)
    valid_T = res_T["T1_eta_ex"].replace(0, np.nan).dropna()
    score_T = float(valid_T.mean()) * 100 if len(valid_T) > 0 else 50

    # E — GWP100: lower is better; scale against 200 kg benchmark → 100 = 0 kg
    sys_co2 = float(res_E.loc["System average","E1_kgCO2eq_MWh"])
    score_E = max(0, min(100, (1 - sys_co2/200) * 100))

    # C — LCOH: lower is better; scale against 120 €/MWh benchmark
    gen_total = sum(generation_GWh.values())
    lcoh_sys  = sum(res_C.loc[t,"C1_LCOH_EUR_MWh"]*generation_GWh.get(t,0)
                    for t in res_C.index) / max(gen_total, 1)
    score_C = max(0, min(100, (1 - lcoh_sys/120) * 100))

    # TC — average capacity factor (%)
    valid_CF = res_TC["CF"].replace(0, np.nan).dropna()
    score_TC = float(valid_CF.mean()) * 100 if len(valid_CF) > 0 else 30

    # F — flexibility score (DFP vs demand)
    dfp_total = float(res_F.loc["TOTAL","F1_DFP_GWh_d"])
    score_F   = min(100, dfp_total / max(gen_total/365, 1) * 100)

    # R — resilience: 1-HHI rescaled; HHI < 0.15 = 100, HHI > 0.4 = 0
    hhi = float(res_R["R1_HHI"])
    score_R = max(0, min(100, (1 - hhi/0.4) * 100))

    # P — RED III policy score (already 0-100)
    score_P = float(res_P["P3_RED3_score"])

    dim_scores = {
        "T":  round(score_T,  1),
        "E":  round(score_E,  1),
        "C":  round(score_C,  1),
        "TC": round(score_TC, 1),
        "F":  round(score_F,  1),
        "R":  round(score_R,  1),
        "P":  round(score_P,  1),
    }

    SPI = sum(SPI_WEIGHTS[k] * v for k, v in dim_scores.items())

    return {
        "generation_GWh": generation_GWh,
        "capacity_MW":    capacity_MW,
        "res_T":  res_T,  "res_E":  res_E,  "res_C":  res_C,
        "res_TC": res_TC, "res_F":  res_F,  "res_R":  res_R,
        "res_P":  res_P,  "res_MCDA": res_MCDA,
        "dim_scores": dim_scores,
        "SPI": round(SPI, 2),
    }


def export_csv(results: dict, path: Path | None = None) -> Path:
    ensure_dirs()
    out = path or TABLES_DIR / "sustainability_SPI_summary.csv"
    rows = []
    for dim, score in results["dim_scores"].items():
        rows.append({
            "dimension":     dim,
            "label":         SPI_LABELS[dim],
            "weight":        SPI_WEIGHTS[dim],
            "score_0_100":   score,
            "weighted":      round(SPI_WEIGHTS[dim]*score, 2),
        })
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"dimension":"SPI","label":"Composite SPI","weight":1.0,
                        "score_0_100":results["SPI"],"weighted":results["SPI"]}
    df.to_csv(out, index=False)
    return out


def plot(results: dict, out_dir: Path | None = None) -> Path:
    ensure_dirs()
    out_dir = out_dir or FIGURES_DIR
    dims    = list(SPI_WEIGHTS.keys())
    scores  = [results["dim_scores"][d] for d in dims]
    labels  = [SPI_LABELS[d] for d in dims]
    N       = len(dims)
    angles  = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    # Close the radar
    scores_r  = scores  + [scores[0]]
    angles_r  = angles  + [angles[0]]
    labels_r  = labels  + [labels[0]]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Sustainability Performance Index (SPI) Dashboard\n"
        f"DH_Cascade_Framework_v4  |  SPI = {results['SPI']:.1f}/100",
        fontsize=12, fontweight="bold"
    )
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Radar chart
    ax = fig.add_subplot(gs[0, 0], polar=True)
    ax.plot(angles_r, scores_r, "o-", color="#2980b9", lw=2)
    ax.fill(angles_r, scores_r, alpha=0.25, color="#2980b9")
    ax.plot(angles_r, [70]*len(angles_r), "--", color="#e74c3c", lw=1, alpha=0.6, label="Target (70)")
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=9)
    ax.set_ylim(0, 100); ax.set_yticks([20,40,60,80,100])
    ax.set_title(f"SPI Radar\n(SPI={results['SPI']:.1f})", fontsize=10, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    # Bar chart — weighted contribution
    ax2 = fig.add_subplot(gs[0, 1])
    contrib = [SPI_WEIGHTS[d]*results["dim_scores"][d] for d in dims]
    colors  = ["#27ae60" if v >= 7 else "#f39c12" if v >= 4 else "#e74c3c" for v in contrib]
    bars    = ax2.bar(labels, contrib, color=colors, edgecolor="white")
    ax2.axhline(results["SPI"]/N, color="black", ls="--", lw=1, label=f"Mean contrib. ({results['SPI']/N:.1f})")
    for bar, v in zip(bars, contrib):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.2, f"{v:.1f}", ha="center", fontsize=8)
    ax2.set_ylabel("Weighted score"); ax2.set_title("SPI Dimension Contributions")
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8); ax2.legend(fontsize=8)

    # Score table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    table_data = [[SPI_LABELS[d], f"{SPI_WEIGHTS[d]*100:.0f}%",
                   f"{results['dim_scores'][d]:.1f}/100",
                   f"{SPI_WEIGHTS[d]*results['dim_scores'][d]:.2f}"] for d in dims]
    table_data.append(["COMPOSITE SPI", "100%", "", f"{results['SPI']:.2f}/100"])
    col_labels = ["Dimension","Weight","Score","Weighted"]
    tbl = ax3.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif r == len(dims)+1:
            cell.set_facecolor("#2980b9"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax3.set_title("SPI Component Table", fontsize=10, fontweight="bold", pad=5)

    out_path = out_dir / "Figure_MTOI_SPI_Dashboard.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def test() -> None:
    print("  Running MTOI aggregate test...")
    res = compute()
    assert 0 <= res["SPI"] <= 100, f"SPI={res['SPI']} outside [0,100]"
    assert len(res["dim_scores"]) == 7
    assert all(0 <= v <= 100 for v in res["dim_scores"].values())
    print(f"  ✓ mtoi: SPI={res['SPI']:.1f} | all tests passed")


if __name__ == "__main__":
    ensure_dirs()
    results = compute()
    print(f"\n{'='*40}")
    print(f"  Sustainability Performance Index (SPI): {results['SPI']:.2f}/100")
    print(f"{'='*40}")
    for d, s in results["dim_scores"].items():
        print(f"  {SPI_LABELS[d]:<15}: {s:.1f}/100")
    print()
    print("CSV →", export_csv(results))
    print("PNG →", plot(results))
    test()
