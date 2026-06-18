"""
mcda.py — MCDA (TOPSIS + Pareto) Multi-Criteria Decision Analysis
==================================================================
Methods:
  TOPSIS  Technique for Order of Preference by Similarity to Ideal Solution
  Pareto  Pareto-front identification across cost vs CO2

Theory
------
TOPSIS (Hwang & Yoon 1981):
  1. Normalise decision matrix:  r_ij = x_ij / sqrt(Σ x_ij²)
  2. Weight:                     v_ij = w_j × r_ij
  3. Ideal best A+:              max benefit, min cost attributes
  4. Ideal worst A−:             opposite
  5. Separation:                 S+ = sqrt(Σ(v_ij − v+_j)²), S− similarly
  6. Closeness:                  C_i = S−_i / (S+_i + S−_i)  ∈ [0,1]

Pareto dominance:
  Option A dominates B if A is at least as good on all criteria and strictly
  better on at least one.

References:
  Hwang C.L. & Yoon K. (1981) Multiple attribute decision making. Springer.
  Miettinen K. (1999) Nonlinear multi-objective optimization. Kluwer.
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

# ── Default criteria and weights ───────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "LCOH_EUR_MWh":      0.25,   # cost      (minimise)
    "CO2_kgCO2_MWh":     0.20,   # env       (minimise)
    "RES_share_pct":     0.20,   # RES       (maximise)
    "exergy_eff_pct":    0.15,   # technical (maximise)
    "redundancy":        0.10,   # resilience(maximise)
    "employment_FTE_MW": 0.10,   # social    (maximise)
}
# Direction: +1 = benefit (maximise), -1 = cost (minimise)
CRITERIA_DIR = {
    "LCOH_EUR_MWh":     -1,
    "CO2_kgCO2_MWh":   -1,
    "RES_share_pct":   +1,
    "exergy_eff_pct":  +1,
    "redundancy":      +1,
    "employment_FTE_MW":+1,
}

# Default scenario matrix (rows = alternatives, cols = criteria)
DEFAULT_SCENARIOS = pd.DataFrame({
    "scenario":         ["Baseline", "High RES", "Low Cost", "Low CO2", "Balanced"],
    "LCOH_EUR_MWh":    [75, 90, 60, 95, 80],
    "CO2_kgCO2_MWh":  [45, 20, 60, 10, 35],
    "RES_share_pct":   [55, 85, 40, 90, 70],
    "exergy_eff_pct":  [40, 55, 30, 60, 48],
    "redundancy":      [1.1,1.3, 0.9, 1.4, 1.2],
    "employment_FTE_MW":[0.07,0.09,0.04,0.08,0.07],
}).set_index("scenario")


def topsis(matrix: pd.DataFrame,
           weights: dict | None = None,
           criteria_dir: dict | None = None) -> pd.Series:
    """
    Run TOPSIS on decision matrix.

    Parameters
    ----------
    matrix      : pd.DataFrame  (alternatives × criteria), numeric
    weights     : dict {criterion: weight}  (must sum ≈ 1)
    criteria_dir: dict {criterion: +1 or -1}

    Returns
    -------
    pd.Series of TOPSIS closeness scores C_i ∈ [0,1] indexed by scenario name
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if criteria_dir is None:
        criteria_dir = CRITERIA_DIR

    criteria = list(weights.keys())
    X   = matrix[criteria].values.astype(float)
    W   = np.array([weights[c] for c in criteria])
    DIR = np.array([criteria_dir[c] for c in criteria])

    # Step 1: normalise
    norms = np.sqrt((X**2).sum(axis=0))
    norms = np.where(norms == 0, 1, norms)
    R = X / norms

    # Step 2: weighted
    V = R * W

    # Step 3 & 4: ideal best/worst
    A_plus  = np.where(DIR > 0, V.max(axis=0), V.min(axis=0))
    A_minus = np.where(DIR > 0, V.min(axis=0), V.max(axis=0))

    # Step 5: separation
    S_plus  = np.sqrt(((V - A_plus)**2).sum(axis=1))
    S_minus = np.sqrt(((V - A_minus)**2).sum(axis=1))

    # Step 6: closeness
    denom = S_plus + S_minus
    C = np.where(denom == 0, 0.0, S_minus / denom)

    return pd.Series(C, index=matrix.index, name="TOPSIS_closeness")


def pareto_front(matrix: pd.DataFrame,
                 obj1: str, obj2: str,
                 dir1: int = -1, dir2: int = -1) -> pd.Series:
    """
    Identify Pareto-optimal alternatives on two objectives.

    Returns pd.Series[bool] — True if Pareto-optimal.
    """
    n = len(matrix)
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            # Does j dominate i?
            better_or_eq = 0
            strictly_better = 0
            for col, d in [(obj1, dir1), (obj2, dir2)]:
                vi, vj = matrix[col].iloc[i], matrix[col].iloc[j]
                if d * vj >= d * vi:
                    better_or_eq += 1
                if d * vj >  d * vi:
                    strictly_better += 1
            if better_or_eq == 2 and strictly_better >= 1:
                on_front[i] = False
                break
    return pd.Series(on_front, index=matrix.index, name="pareto_optimal")


def compute(scenarios: pd.DataFrame | None = None,
            weights: dict | None = None) -> dict:
    """
    Run full MCDA: TOPSIS + Pareto front.

    Returns dict: topsis_scores, pareto_mask, ranking, scenarios
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS.copy()
    if weights is None:
        weights = DEFAULT_WEIGHTS

    scores  = topsis(scenarios, weights)
    pareto  = pareto_front(scenarios, "LCOH_EUR_MWh", "CO2_kgCO2_MWh")
    ranking = scores.sort_values(ascending=False)

    return {
        "topsis_scores": scores,
        "pareto_mask":   pareto,
        "ranking":       ranking,
        "scenarios":     scenarios,
        "weights":       weights,
    }


def export_csv(results: dict, path: Path | None = None) -> Path:
    ensure_dirs()
    out = path or TABLES_DIR / "sustainability_MCDA.csv"
    df  = results["scenarios"].copy()
    df["TOPSIS_closeness"] = results["topsis_scores"]
    df["pareto_optimal"]   = results["pareto_mask"]
    df["TOPSIS_rank"]      = results["topsis_scores"].rank(ascending=False).astype(int)
    df.to_csv(out)
    return out


def plot(results: dict, out_dir: Path | None = None) -> Path:
    ensure_dirs()
    out_dir  = out_dir or FIGURES_DIR
    scores   = results["topsis_scores"]
    pareto   = results["pareto_mask"]
    scenarios = results["scenarios"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MCDA — TOPSIS + Pareto Analysis\nDH_Cascade_Framework_v4", fontsize=12, fontweight="bold")

    # Panel 1 — TOPSIS bar chart
    ax = axes[0]
    ranked = scores.sort_values(ascending=False)
    colors = ["#2ecc71" if pareto.loc[s] else "#3498db" for s in ranked.index]
    bars   = ax.bar(ranked.index, ranked.values, color=colors, edgecolor="white")
    for bar, v in zip(bars, ranked.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("TOPSIS closeness C_i ∈ [0,1]"); ax.set_title("TOPSIS Ranking")
    ax.set_xticklabels(ranked.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#2ecc71", label="Pareto-optimal"),
        mpatches.Patch(color="#3498db", label="Non-Pareto"),
    ], fontsize=8)

    # Panel 2 — Pareto scatter (cost vs CO2)
    ax = axes[1]
    sc_x = scenarios["LCOH_EUR_MWh"].values
    sc_y = scenarios["CO2_kgCO2_MWh"].values
    pf   = pareto.values
    ax.scatter(sc_x[~pf], sc_y[~pf], c="#3498db", s=90, zorder=5, label="Dominated")
    ax.scatter(sc_x[ pf], sc_y[ pf], c="#e74c3c", s=120, marker="*", zorder=6, label="Pareto-front")
    for name, xi, yi in zip(scenarios.index, sc_x, sc_y):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(6,4), fontsize=8)
    pf_pts = scenarios[pareto].sort_values("LCOH_EUR_MWh")
    ax.plot(pf_pts["LCOH_EUR_MWh"], pf_pts["CO2_kgCO2_MWh"], "r--", lw=1.2, alpha=0.6)
    ax.set_xlabel("LCOH (€/MWh)"); ax.set_ylabel("CO₂ intensity (kg/MWh)")
    ax.set_title("Pareto Front — Cost vs CO₂")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "Figure_MCDA.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def test() -> None:
    res = compute()
    scores = res["topsis_scores"]
    assert (scores >= 0).all() and (scores <= 1).all(), "TOPSIS scores outside [0,1]"
    assert res["pareto_mask"].any(), "At least one Pareto-optimal alternative expected"
    best = res["ranking"].index[0]
    assert scores[best] == scores.max(), "Top-ranked has max score"
    print("  ✓ mcda: all unit tests passed")


if __name__ == "__main__":
    ensure_dirs()
    res = compute()
    print("TOPSIS ranking:")
    print(res["ranking"].to_string())
    print("\nPareto-optimal:", res["pareto_mask"][res["pareto_mask"]].index.tolist())
    print("CSV →", export_csv(res))
    print("PNG →", plot(res))
    test()
