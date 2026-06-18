"""
DH_Cascade_Framework_v4 — Sustainability Layer
===============================================
Nine indicator modules, each providing compute() / export_csv() / plot() / test().

Dimensions
----------
  T1-T5   exergy      Exergy efficiency, destruction, thermo-economic cost, CO2
  E1-E4   lifecycle   GWP100, CED, acidification, land use
  C1-C4   economic    LCOH, NPV, payback, employment
  TC1-TC4 technical   Capacity factor, self-sufficiency, peak coverage, heat loss
  F1-F3   flexibility DFP, ramp rate, storage utilisation
  R1-R3   resilience  HHI diversity, fuel import dependency, N-1 redundancy
  P1-P4   policy      RES share, EU ETS cost, RED III score, NECP alignment
  MCDA    mcda        TOPSIS ranking + Pareto front (cost vs CO2)
  SPI     mtoi        Composite Sustainability Performance Index (aggregator)

Usage
-----
  from 02_Sustainability_Layer import exergy, lifecycle, mcda, mtoi
  df  = exergy.compute(generation_GWh)
  png = exergy.plot(df)

Or run the full SPI pipeline:
  from 02_Sustainability_Layer.mtoi import compute, plot, export_csv
  results = compute(generation_GWh, capacity_MW)
  export_csv(results)
  plot(results)
"""

__all__ = [
    "exergy", "lifecycle", "economic", "technical",
    "flexibility", "resilience", "policy", "mcda", "mtoi",
]
