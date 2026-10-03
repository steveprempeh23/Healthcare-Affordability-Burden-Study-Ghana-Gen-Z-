"""
Stylized Monte Carlo simulation: projected household out-of-pocket (OOP)
healthcare burden for Ghanaian Gen Z cohort, 2026-2051 (25-year horizon).

Calibration sources (see paper References, Chicago style):
- Minimum wage series 2015-2026 (GHS 7.00 -> GHS 21.77): Finex Insights 2025;
  Playroll 2026.
- Long-run CPI inflation average and volatility: Trading Economics / Ghana
  Statistical Service CPI series (1998-2026 average 17.16%, historical range
  0.4%-63.1%).
- Recent inflation regime shifts (2022 shock ~37-54%; 2023 ~23-40%; 2025-26
  disinflation to 3-4%): Trading Economics news series; Bloomberg (2022);
  MyJoyOnline (2022).
- Out-of-pocket share of current health expenditure (~26%), government health
  spending (~4.2% of GDP, below Abuja 15%-of-budget target): World Bank/WHO
  Global Health Expenditure Database; World Bank (2023) Public Health
  Expenditure for Universal Health Coverage in Ghana.
- Catastrophic health expenditure (CHE) incidence: Sataru, Twumasi-Ankrah, and
  Seddoh (2022), Frontiers in Health Services (GLSS7); Wang, Ataguba, and
  colleagues, cross-SSA CHE incidence range 12.9%-20.4%.

This is a stylized, author-calibrated stress-test simulation in the tradition
of WHO/World Bank UHC financial-protection modeling. It is NOT a structural
econometric forecast; parameters are order-of-magnitude estimates drawn from
the cited historical series, documented for transparency and replication.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(20260814)

N_HOUSEHOLDS = 10_000
N_YEARS = 26  # 2026 -> 2051 inclusive (26 annual observations spanning a
              # 25-year horizon; cohort aged 25 -> 50). NOTE: an earlier
              # version of this code used N_YEARS=25, which produced only
              # 25 data points (2026-2050) -- one year short of the stated
              # 2051 endpoint. Fixed for Table 1/Figure 1/Table 4 accuracy.

# ---- Core calibrated parameters -------------------------------------------
NOMINAL_WAGE_CAGR = 0.109       # 11-year minimum wage CAGR, 2015-2026
BASE_OOP_FRACTION = 0.06        # typical non-crisis annual OOP share of income
OOP_SHARE_OF_HEALTH_SPEND = 0.26  # national OOP share of current health exp.
LIQUIDITY_BUFFER = 1 / 12       # ~1 month of income held as liquid savings

SCENARIOS = {
    "status_quo": dict(
        p_shock_year=0.12,          # prob. of entering a high-inflation year
        shock_inflation_mean=0.30, shock_inflation_sd=0.10,
        stable_inflation_mean=0.10, stable_inflation_sd=0.03,
        import_premium_shock=0.06,  # extra health-cost inflation, shock yrs
        import_premium_stable=0.02,
        p_health_crisis=0.08,       # annual probability of a catastrophic
                                     # health event per household
        oop_share_drift=0.0,        # annual change in OOP share of spending
    ),
    "reform": dict(
        p_shock_year=0.12,
        shock_inflation_mean=0.30, shock_inflation_sd=0.10,
        stable_inflation_mean=0.10, stable_inflation_sd=0.03,
        import_premium_shock=0.03,
        import_premium_stable=0.01,
        p_health_crisis=0.064,      # -20% relative, better primary care
        oop_share_drift=-0.006,     # OOP share of spend falls as gov't
                                     # financing rises toward Abuja target
    ),
    "shock_heavy": dict(
        p_shock_year=0.25,          # currency/inflation shocks twice as
                                     # frequent
        shock_inflation_mean=0.35, shock_inflation_sd=0.12,
        stable_inflation_mean=0.10, stable_inflation_sd=0.03,
        import_premium_shock=0.09,
        import_premium_stable=0.02,
        p_health_crisis=0.08,
        oop_share_drift=0.004,      # NHIS financing erodes further
    ),
}


def run_scenario(params, n=N_HOUSEHOLDS, years=N_YEARS, seed=20260814):
    """Runs one scenario with its OWN freshly-seeded random number generator,
    rather than drawing from a shared global stream. This guarantees that
    calling run_scenario with the same (params, n, years, seed) ALWAYS
    produces bit-identical results, regardless of what other simulations
    were run earlier in the same process -- required for reproducibility
    (a prior version of this code shared one mutable global RNG object
    across calls, which meant results silently depended on call order)."""
    local_rng = np.random.default_rng(seed)
    income_idx = np.ones(n)
    health_cost_idx = np.ones(n)
    oop_share = np.full(n, OOP_SHARE_OF_HEALTH_SPEND)

    annual_oop_pct = np.zeros((years, n))
    crisis_flags = np.zeros((years, n), dtype=bool)
    cumulative_crises = np.zeros(n, dtype=int)
    cumulative_debt_events = np.zeros(n, dtype=int)

    for t in range(years):
        shock = local_rng.random(n) < params["p_shock_year"]
        inflation = np.where(
            shock,
            local_rng.normal(params["shock_inflation_mean"], params["shock_inflation_sd"], n),
            local_rng.normal(params["stable_inflation_mean"], params["stable_inflation_sd"], n),
        )
        inflation = np.clip(inflation, -0.02, 0.65)

        import_premium = np.where(shock, params["import_premium_shock"], params["import_premium_stable"])
        health_cost_growth = inflation + import_premium
        wage_growth = NOMINAL_WAGE_CAGR - 0.0  # nominal wage path fixed at historical CAGR

        income_idx *= (1 + wage_growth)
        health_cost_idx *= (1 + health_cost_growth)
        oop_share = np.clip(oop_share + params["oop_share_drift"], 0.10, 0.55)

        # cost/income gap ratio: how much faster health costs have grown
        # than income since year 0 (captures affordability erosion)
        gap_ratio = health_cost_idx / income_idx

        crisis = local_rng.random(n) < params["p_health_crisis"]
        base_burden = BASE_OOP_FRACTION * oop_share / OOP_SHARE_OF_HEALTH_SPEND * gap_ratio
        crisis_multiplier = local_rng.uniform(3.0, 7.0, n)  # catastrophic events cost
                                                       # 3-7x a normal year, in
                                                       # line with CHE severity
                                                       # literature
        burden = np.where(crisis, base_burden * crisis_multiplier, base_burden)
        burden = np.clip(burden, 0.01, 1.2)

        annual_oop_pct[t] = burden
        crisis_flags[t] = crisis
        cumulative_crises += crisis.astype(int)
        cumulative_debt_events += ((burden - LIQUIDITY_BUFFER) > 0.15).astype(int) & crisis

    return {
        "annual_oop_pct": annual_oop_pct,
        "crisis_flags": crisis_flags,
        "cumulative_crises": cumulative_crises,
        "cumulative_debt_events": cumulative_debt_events,
    }


import hashlib

results = {}
for name, params in SCENARIOS.items():
    # Python's built-in hash() is randomized per-process for strings (security
    # feature), so it CANNOT be used here -- it would silently reintroduce
    # non-reproducibility. hashlib.md5 is deterministic across runs/processes.
    scenario_seed = 20260814 + int(hashlib.md5(name.encode()).hexdigest(), 16) % 100000
    results[name] = run_scenario(params, seed=scenario_seed)

# ---- Summary tables ---------------------------------------------------------
summary_rows = []
for name, res in results.items():
    final_year_burden = res["annual_oop_pct"][-1]
    mean_burden_path = res["annual_oop_pct"].mean(axis=1)  # length N_YEARS
    p_at_least_one_crisis_25y = (res["cumulative_crises"] >= 1).mean()
    mean_debt_events_25y = res["cumulative_debt_events"].mean()
    summary_rows.append({
        "scenario": name,
        "median_oop_pct_income_2051": np.median(final_year_burden) * 100,
        "p25_2051": np.percentile(final_year_burden, 25) * 100,
        "p75_2051": np.percentile(final_year_burden, 75) * 100,
        "p90_2051": np.percentile(final_year_burden, 90) * 100,
        "mean_oop_pct_income_2026": mean_burden_path[0] * 100,
        "mean_oop_pct_income_2051": mean_burden_path[-1] * 100,
        "prob_at_least_1_catastrophic_event_by_50": p_at_least_one_crisis_25y * 100,
        "mean_debt_triggering_events_by_50": mean_debt_events_25y,
    })

summary = pd.DataFrame(summary_rows).set_index("scenario")
summary.to_csv("/home/claude/arf_paper/summary_table.csv")
print(summary.round(2).to_string())

# Year-by-year mean path for each scenario, for the figure and appendix table
paths = pd.DataFrame({
    name: results[name]["annual_oop_pct"].mean(axis=1) * 100
    for name in SCENARIOS
})
paths.index = range(2026, 2026 + N_YEARS)
paths.index.name = "year"
paths.to_csv("/home/claude/arf_paper/annual_paths.csv")
print("\nYear-by-year mean OOP burden (% of income):")
print(paths.round(1).iloc[[0, 4, 9, 14, 19, 24]].to_string())
