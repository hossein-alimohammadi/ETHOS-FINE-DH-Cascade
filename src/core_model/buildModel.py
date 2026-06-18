"""
buildModel.py
=============
Builds the ETHOS.FINE EnergySystemModel for the Estonia district heating
optimisation — DH_Cascade_Framework_v3 (single heat commodity, 15 counties).

Model scope (v3)
----------------
  Commodity  : heat  [GW_th]
  Locations  : 15 Estonian counties
  Components :
    - Heat demand       — fn.Sink  (operationRateFix, aggregated HTDHN+LTDHN+VLTDHN)
    - Biomass boiler    — fn.Source (capacityMax from biomass potential data)
    - Gas boiler        — fn.Source (unconstrained capacity)
    - Large heat pump   — fn.Source (unconstrained capacity)
    - Thermal storage   — fn.Storage (short-term pressurised tank)
    - DH pipes          — fn.Transmission (eligibility + distance matrix)

NOT included in v3 (planned for v4+)
--------------------------------------
  - HTDHN / LTDHN / VLTDHN cascade split
  - CHP, geothermal, solar thermal, industrial waste heat
  - Exergy / LCA / resilience / flexibility indicators
  - MCDA / Pareto analysis

Cost conventions
----------------
  costUnit = "1e9 Euro"  (Bn€)

  Technology costs come from InputData/CostAssumptions/ tables, which store
  values in Bn€ and can be replaced with real data without changing this file.

  Fuel costs (commodity costs) are hardcoded as Estonian market proxies
  (SYNTHETIC) and should be updated when real price data is available.
"""

import fine as fn
import pandas as pd
import numpy as np
from pathlib import Path

from getDHData import getDHData, COUNTIES

# ── Fuel / carbon costs (SYNTHETIC — replace with real market prices) ─────────
BIOMASS_FUEL_EURO_PER_MWH   = 35.0    # EUR/MWh_fuel  — wood chips, Estonia
GAS_FUEL_EURO_PER_MWH       = 80.0    # EUR/MWh_fuel  — natural gas
ELECTRICITY_EURO_PER_MWH    = 70.0    # EUR/MWh_el    — grid (day-ahead average)
CARBON_PRICE_EURO_PER_T     = 65.0    # EUR/tCO2      — EU ETS (SYNTHETIC)

# Combustion emission factors (tCO2/MWh_fuel)
BIOMASS_EF   = 0.000   # biogenic — zero under EU ETS
GAS_EF       = 0.202   # IPCC Tier 1 natural gas

_BN  = 1.0e9     # 1 Bn€
_MWH_PER_GWH = 1000.0


def _commodity_cost_bn_per_gwh(fuel_euro_mwh: float, efficiency: float,
                                emission_factor: float = 0.0) -> float:
    """
    Convert fuel cost and carbon price to Bn€/GWh of heat output.

    Parameters
    ----------
    fuel_euro_mwh   : fuel cost in EUR/MWh_fuel
    efficiency      : boiler / COP efficiency (heat out / fuel in)
    emission_factor : tCO2/MWh_fuel (0 for CO2-neutral fuels)
    """
    carbon_euro_mwh_fuel  = emission_factor * CARBON_PRICE_EURO_PER_T
    total_euro_mwh_fuel   = fuel_euro_mwh + carbon_euro_mwh_fuel
    total_euro_mwh_heat   = total_euro_mwh_fuel / efficiency
    return total_euro_mwh_heat * _MWH_PER_GWH / _BN


def buildModel(data: dict,
               interestRate: float = 0.08,
               cop: "float | pd.Series" = 3.50,
               biomass_capacity_min: float = 0.0,
               biomass_op_rate_min: float = 0.0,
               gas_capacity_min: float = 0.0,
               hp_capacity_max=None) -> fn.EnergySystemModel:
    """
    Construct the ETHOS.FINE EnergySystemModel for Estonia DH v3.

    Parameters
    ----------
    data : dict
        Output of getDHData().  All 24 keys must be present.
    interestRate : float
        WACC applied uniformly to all investment components (default 8 %).
    cop : float or pd.Series
        Heat pump COP.  If float, a single value is applied to all counties.
        If pd.Series (index = county names), each county gets its own COP.
        Default: 3.50 (constant COP, v4_2 baseline).
        For temperature-dependent COP, pass output of
        heat_pump_cop.demand_weighted_cop().

    Returns
    -------
    fn.EnergySystemModel
        Unoptimised model.  Call esM.optimize() or esM.cluster() next.

    Notes
    -----
    The demand Sink receives the sum of HTDHN + LTDHN + VLTDHN demand profiles
    so that the single-commodity model must supply all three heat levels.
    Cascade temperature constraints are not enforced in v3.
    """

    # ── Cost table look-ups ──────────────────────────────────────────────────
    tech = data["Heat technologies, costAssumptions"]   # (12 × 6) DataFrame
    stor = data["Thermal storage, costAssumptions"]     # (3 × 9) DataFrame
    pipe = data["DH pipes, costAssumptions"]            # (6 × 3) DataFrame

    # Row keys — must match index values in heatTechnologyCostAssumptions.xlsx
    _BIOMASS = "Biomass boiler"
    _GAS     = "Gas boiler (natural gas)"
    _HP      = "Large-scale heat pump (HTDHN)"
    _TES     = "HTDHN TES (short-term, pressurised tank)"

    def _t(row, col):
        """Scalar lookup from tech table."""
        return float(tech.loc[row, col])

    def _s(row, col):
        """Scalar lookup from storage table."""
        return float(stor.loc[row, col])

    def _p(row):
        """Scalar lookup from pipe table (single 'Value' column)."""
        return float(pipe.loc[row, "Value"])

    # ── Aggregate total heat demand ───────────────────────────────────────────
    # HTDHN (65 %) + LTDHN (25 %) + VLTDHN (10 %) → single "heat" commodity
    # Units: GWh/h = GW  (hoursPerTimeStep=1)
    demand_total: pd.DataFrame = (
        data["Heat demand HTDHN, operationRateFix"]
        + data["Heat demand LTDHN, operationRateFix"]
        + data["Heat demand VLTDHN, operationRateFix"]
    )
    # Shape: (8760, 15) — columns are county names, index 0..8759

    # ── Create EnergySystemModel ──────────────────────────────────────────────
    esM = fn.EnergySystemModel(
        locations=set(COUNTIES),
        commodities={"heat"},
        numberOfTimeSteps=8760,
        commodityUnitsDict={"heat": r"GW$_{th}$"},
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",      # Bn€  — matches CostAssumptions tables
        lengthUnit="km",
        verboseLogLevel=0,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Heat demand sink
    # ═══════════════════════════════════════════════════════════════════════════
    esM.add(fn.Sink(
        esM,
        name="Heat demand",
        commodity="heat",
        hasCapacityVariable=False,
        operationRateFix=demand_total,   # GW per hour per county
    ))

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Biomass boiler
    #    Capacity upper bound from county-level biomass resource potential.
    # ═══════════════════════════════════════════════════════════════════════════
    biomass_invest = _t(_BIOMASS, "investPerCapacity_BnEuro_per_GW")
    biomass_opex   = _t(_BIOMASS, "opexPerCapacity_BnEuro_per_GW_yr")
    biomass_life   = int(_t(_BIOMASS, "lifetime_yr"))
    biomass_eta    = _t(_BIOMASS, "eta_or_COP_ref")
    biomass_fuel   = _commodity_cost_bn_per_gwh(
        BIOMASS_FUEL_EURO_PER_MWH, biomass_eta, BIOMASS_EF)

    # biomass_capacity_min: scalar applied uniformly, clipped to county potential
    _bio_cap_min = None
    if biomass_capacity_min > 0:
        bio_max_series = data["Biomass boiler, capacityMax"]
        _bio_cap_min = bio_max_series.clip(upper=biomass_capacity_min).clip(lower=0)
    _bio_op_min = biomass_op_rate_min if biomass_op_rate_min > 0 else None

    esM.add(fn.Source(
        esM,
        name="Biomass boiler",
        commodity="heat",
        hasCapacityVariable=True,
        capacityMax=data["Biomass boiler, capacityMax"],  # pd.Series (15,) GW
        **({} if _bio_cap_min is None else {"capacityMin": _bio_cap_min}),
        **({} if _bio_op_min is None else {"operationRateMin": _bio_op_min}),
        investPerCapacity=biomass_invest,
        opexPerCapacity=biomass_opex,
        commodityCost=biomass_fuel,
        interestRate=interestRate,
        economicLifetime=biomass_life,
    ))

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Natural gas boiler
    #    No binding capacity constraint — model selects optimal capacity.
    # ═══════════════════════════════════════════════════════════════════════════
    gas_invest = _t(_GAS, "investPerCapacity_BnEuro_per_GW")
    gas_opex   = _t(_GAS, "opexPerCapacity_BnEuro_per_GW_yr")
    gas_life   = int(_t(_GAS, "lifetime_yr"))
    gas_eta    = _t(_GAS, "eta_or_COP_ref")
    gas_fuel   = _commodity_cost_bn_per_gwh(
        GAS_FUEL_EURO_PER_MWH, gas_eta, GAS_EF)

    _gas_cap_min = gas_capacity_min if gas_capacity_min > 0 else None

    esM.add(fn.Source(
        esM,
        name="Gas boiler",
        commodity="heat",
        hasCapacityVariable=True,
        **({} if _gas_cap_min is None else {"capacityMin": _gas_cap_min}),
        investPerCapacity=gas_invest,
        opexPerCapacity=gas_opex,
        commodityCost=gas_fuel,
        interestRate=interestRate,
        economicLifetime=gas_life,
    ))

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Large-scale heat pump
    #    Electricity-driven; effective fuel cost = electricity_price / COP.
    #    No binding capacity constraint.
    # ═══════════════════════════════════════════════════════════════════════════
    hp_invest = _t(_HP, "investPerCapacity_BnEuro_per_GW")
    hp_opex   = _t(_HP, "opexPerCapacity_BnEuro_per_GW_yr")
    hp_life   = int(_t(_HP, "lifetime_yr"))
    hp_cop    = _t(_HP, "eta_or_COP_ref")   # reference COP (used if cop arg == 3.50)
    # COP override: use caller-supplied COP (constant float or county Series)
    # This allows temperature-dependent COP to be injected without changing FINE API.
    # If cop is a Series, commodityCost becomes a county-varying Series (Bn€/GWh).
    if isinstance(cop, pd.Series):
        # County-specific fuel cost Series [Bn€/GWh]
        hp_fuel = (ELECTRICITY_EURO_PER_MWH / cop) * _MWH_PER_GWH / _BN
        # Ensure county order matches FINE's expected index
        hp_fuel = hp_fuel.reindex(COUNTIES)
    else:
        effective_cop = float(cop)
        hp_fuel = (ELECTRICITY_EURO_PER_MWH / effective_cop) * _MWH_PER_GWH / _BN

    # hp_capacity_max: pd.Series (county-level cap) or None (unconstrained)
    _hp_cap_max_kw = hp_capacity_max  # may be None or pd.Series

    esM.add(fn.Source(
        esM,
        name="Heat pump",
        commodity="heat",
        hasCapacityVariable=True,
        **({} if _hp_cap_max_kw is None else {"capacityMax": _hp_cap_max_kw}),
        investPerCapacity=hp_invest,
        opexPerCapacity=hp_opex,
        commodityCost=hp_fuel,
        interestRate=interestRate,
        economicLifetime=hp_life,
    ))

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Thermal energy storage — HTDHN pressurised hot-water tank
    #    investPerCapacity in Bn€/GWh (energy capacity).
    # ═══════════════════════════════════════════════════════════════════════════
    tes_invest  = _s(_TES, "investPerCapacity_BnEuro_per_GWh")
    tes_life    = int(_s(_TES, "lifetime_yr"))
    tes_opex_fr = _s(_TES, "opexFraction_of_CAPEX_yr")
    tes_opex    = tes_invest * tes_opex_fr          # Bn€/GWh/yr (fixed O&M)
    charge_eff  = _s(_TES, "chargeEff")
    dis_eff     = _s(_TES, "dischargeEff")
    self_dis    = _s(_TES, "selfDischarge_per_hour")
    soc_min     = _s(_TES, "minSOC_fraction")
    soc_max     = _s(_TES, "maxSOC_fraction")

    esM.add(fn.Storage(
        esM,
        name="Thermal storage",
        commodity="heat",
        hasCapacityVariable=True,
        chargeEfficiency=charge_eff,
        dischargeEfficiency=dis_eff,
        selfDischarge=self_dis,
        # chargeRate / dischargeRate: max charge power per unit of storage energy (1/h)
        # 0.125 = 8-hour fill/empty time, appropriate for large hot-water tanks
        chargeRate=0.125,
        dischargeRate=0.125,
        stateOfChargeMin=soc_min,
        stateOfChargeMax=soc_max,
        investPerCapacity=tes_invest,
        opexPerCapacity=tes_opex,
        interestRate=interestRate,
        economicLifetime=tes_life,
    ))

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Inter-county DH transmission (long-distance insulated pipes)
    #
    # Bug fix (v3 review): FINE does NOT multiply investPerCapacity by distances.
    # investPerCapacity must be a per-connection cost matrix (Bn€/GW), computed
    # here as pipe_invest_per_km × distance[i,j].  Non-eligible connections have
    # distance=0, so they naturally get zero investment cost.
    #
    # losses: FINE accepts a scalar (per-km loss rate) which it multiplies by
    # distances internally, or a pre-computed matrix.  We pass a matrix so that
    # the heat-loss fraction for each connection is explicit.
    #
    # locationalEligibility (not "eligibility") is the correct FINE parameter.
    # ═══════════════════════════════════════════════════════════════════════════
    pipe_invest_per_km = _p("investPerCapacityDistance_BnEuro_per_GW_km")
    pipe_life          = int(_p("lifetime_yr"))
    pipe_max_cap_gw    = _p("maxCapacity_GW_per_pipe")
    pipe_loss_per_km   = _p("heatLoss_fraction_per_km")
    dist_km            = data["DH pipes, distances"]           # 15×15 km matrix

    # Per-connection investment cost (Bn€/GW); zero for ineligible pairs
    pipe_invest_matrix = dist_km * pipe_invest_per_km
    # Per-connection max capacity (GW); zero for ineligible pairs
    pipe_cap_max       = data["DH pipes, eligibility"] * pipe_max_cap_gw

    esM.add(fn.Transmission(
        esM,
        name="DH pipes",
        commodity="heat",
        hasCapacityVariable=True,
        locationalEligibility=data["DH pipes, eligibility"],
        distances=dist_km,
        losses=pipe_loss_per_km,    # scalar [fraction/km]; FINE multiplies internally
        investPerCapacity=pipe_invest_matrix,
        economicLifetime=pipe_life,
        capacityMax=pipe_cap_max,
    ))

    return esM
