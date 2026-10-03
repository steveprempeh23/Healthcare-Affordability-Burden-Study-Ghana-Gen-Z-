"""
Requirement 5: Debt-trigger sensitivity check.
Requirement 6: Probabilistic (global) sensitivity analysis, complementing
the existing one-at-a-time (OAT) analysis.

Both re-use the exact primary-simulation mechanics from simulation.py.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/claude/arf_paper")
from simulation import SCENARIOS, N_HOUSEHOLDS, N_YEARS, BASE_OOP_FRACTION, OOP_SHARE_OF_HEALTH_SPEND, NOMINAL_WAGE_CAGR

STATUS_QUO = SCENARIOS["status_quo"]

# =============================================================================
# PART A (Requirement 5): Debt-trigger sensitivity
# =============================================================================
def run_with_debt_params(params, liquidity_buffer_months, threshold_pp, n=N_HOUSEHOLDS, years=N_YEARS, seed=1):
    """Identical mechanics to simulation.run_scenario, but with the
    liquidity buffer (in months of income) and debt-trigger threshold
    (percentage points above the buffer) as explicit, variable inputs."""
    local_rng = np.random.default_rng(seed)
    income_idx = np.ones(n)
    health_cost_idx = np.ones(n)
    oop_share = np.full(n, OOP_SHARE_OF_HEALTH_SPEND)
    liquidity_buffer = liquidity_buffer_months / 12.0
    threshold = threshold_pp / 100.0

    cumulative_debt_events = np.zeros(n, dtype=int)
    cumulative_crises = np.zeros(n, dtype=int)

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
        income_idx *= (1 + NOMINAL_WAGE_CAGR)
        health_cost_idx *= (1 + health_cost_growth)
        oop_share = np.clip(oop_share + params["oop_share_drift"], 0.10, 0.55)
        gap_ratio = health_cost_idx / income_idx

        crisis = local_rng.random(n) < params["p_health_crisis"]
        base_burden = BASE_OOP_FRACTION * oop_share / OOP_SHARE_OF_HEALTH_SPEND * gap_ratio
        mult = local_rng.uniform(3.0, 7.0, n)
        burden = np.where(crisis, base_burden * mult, base_burden)
        burden = np.clip(burden, 0.01, 1.20)

        debt_event = ((burden - liquidity_buffer) > threshold) & crisis
        cumulative_debt_events += debt_event.astype(int)
        cumulative_crises += crisis.astype(int)

    return {
        "mean_debt_events": cumulative_debt_events.mean(),
        "p_at_least_1_debt_event": (cumulative_debt_events >= 1).mean() * 100,
        "mean_crises": cumulative_crises.mean(),
    }


print("=" * 78)
print("PART A: Debt-trigger sensitivity (Requirement 5)")
print("=" * 78)
print("Base case: 1-month liquidity buffer, 15pp threshold\n")

buffer_grid = [0.5, 1.0, 2.0, 3.0]
threshold_grid = [10, 15, 20, 25]

debt_sensitivity_rows = []
for buf in buffer_grid:
    for thr in threshold_grid:
        res = run_with_debt_params(STATUS_QUO, buf, thr, seed=42)
        debt_sensitivity_rows.append({
            "liquidity_buffer_months": buf, "threshold_pp": thr,
            "mean_debt_events": res["mean_debt_events"],
            "p_at_least_1_debt_event": res["p_at_least_1_debt_event"],
        })
debt_df = pd.DataFrame(debt_sensitivity_rows)
debt_df.to_csv("/home/claude/arf_paper/debt_trigger_sensitivity.csv", index=False)
print(debt_df.round(2).to_string(index=False))

base_case = debt_df[(debt_df.liquidity_buffer_months == 1.0) & (debt_df.threshold_pp == 15)]
print(f"\nBase case (1 month, 15pp): mean debt events = {base_case['mean_debt_events'].values[0]:.2f}, "
      f"P(>=1 event) = {base_case['p_at_least_1_debt_event'].values[0]:.1f}%")
low = debt_df[(debt_df.liquidity_buffer_months == 3.0) & (debt_df.threshold_pp == 25)]
high = debt_df[(debt_df.liquidity_buffer_months == 0.5) & (debt_df.threshold_pp == 10)]
print(f"Most conservative (3mo, 25pp): mean debt events = {low['mean_debt_events'].values[0]:.2f}, "
      f"P(>=1 event) = {low['p_at_least_1_debt_event'].values[0]:.1f}%")
print(f"Most permissive (0.5mo, 10pp): mean debt events = {high['mean_debt_events'].values[0]:.2f}, "
      f"P(>=1 event) = {high['p_at_least_1_debt_event'].values[0]:.1f}%")

# =============================================================================
# PART B (Requirement 6): Probabilistic / global sensitivity analysis
# =============================================================================
print("\n" + "=" * 78)
print("PART B: Probabilistic (global) sensitivity analysis (Requirement 6)")
print("=" * 78)

PARAM_RANGES = {
    "p_health_crisis": (0.04, 0.15),
    "severity_high": (4.0, 10.0),
    "p_shock_year": (0.06, 0.25),
    "import_premium_shock": (0.02, 0.12),
    "base_oop_fraction": (0.03, 0.10),
    "nominal_wage_cagr": (0.08, 0.13),
    "oop_share_of_health": (0.20, 0.32),
}

N_DRAWS = 300
N_HOUSEHOLDS_PSA = 2000
draw_rng = np.random.default_rng(2026)

def run_scenario_variable_base(base_oop_fraction, oop_share_of_health, severity_high,
                                 p_shock_year, import_premium_shock, p_health_crisis,
                                 nominal_wage_cagr, n=N_HOUSEHOLDS_PSA, years=N_YEARS, seed=None):
    local_rng = np.random.default_rng(seed)
    income_idx = np.ones(n)
    health_cost_idx = np.ones(n)
    oop_share = np.full(n, oop_share_of_health)
    for t in range(years):
        shock = local_rng.random(n) < p_shock_year
        inflation = np.where(
            shock,
            local_rng.normal(STATUS_QUO["shock_inflation_mean"], STATUS_QUO["shock_inflation_sd"], n),
            local_rng.normal(STATUS_QUO["stable_inflation_mean"], STATUS_QUO["stable_inflation_sd"], n),
        )
        inflation = np.clip(inflation, -0.02, 0.65)
        import_premium = np.where(shock, import_premium_shock, STATUS_QUO["import_premium_stable"])
        income_idx *= (1 + nominal_wage_cagr)
        health_cost_idx *= (1 + inflation + import_premium)
        oop_share = np.clip(oop_share + STATUS_QUO["oop_share_drift"], 0.10, 0.55)
        gap_ratio = health_cost_idx / income_idx
        crisis = local_rng.random(n) < p_health_crisis
        base_burden = base_oop_fraction * oop_share / oop_share_of_health * gap_ratio
        mult = local_rng.uniform(3.0, severity_high, n)
        burden = np.where(crisis, base_burden * mult, base_burden)
        burden = np.clip(burden, 0.01, 1.20)
    return burden.mean() * 100

psa_results = []
psa_params = []
for draw in range(N_DRAWS):
    sampled = {k: draw_rng.uniform(lo, hi) for k, (lo, hi) in PARAM_RANGES.items()}
    outcome = run_scenario_variable_base(
        base_oop_fraction=sampled["base_oop_fraction"],
        oop_share_of_health=sampled["oop_share_of_health"],
        severity_high=sampled["severity_high"],
        p_shock_year=sampled["p_shock_year"],
        import_premium_shock=sampled["import_premium_shock"],
        p_health_crisis=sampled["p_health_crisis"],
        nominal_wage_cagr=sampled["nominal_wage_cagr"],
        seed=3000 + draw,
    )
    psa_results.append(outcome)
    psa_params.append(sampled)

psa_df = pd.DataFrame(psa_params)
psa_df["outcome_2051_pct"] = psa_results
psa_df.to_csv("/home/claude/arf_paper/probabilistic_sensitivity.csv", index=False)

print(f"\n{N_DRAWS} joint parameter draws, primary simulation (2051), status-quo structure:")
print(f"  Mean outcome: {psa_df.outcome_2051_pct.mean():.2f}%")
print(f"  Median outcome: {psa_df.outcome_2051_pct.median():.2f}%")
print(f"  P10-P90 range: {psa_df.outcome_2051_pct.quantile(0.10):.2f}% - {psa_df.outcome_2051_pct.quantile(0.90):.2f}%")
print(f"  Min-Max range: {psa_df.outcome_2051_pct.min():.2f}% - {psa_df.outcome_2051_pct.max():.2f}%")
print(f"  % of draws exceeding 20% (catastrophic threshold): {(psa_df.outcome_2051_pct > 20).mean()*100:.1f}%")

print("\nSpearman correlation of each parameter with the 2051 outcome (joint draws):")
corrs = psa_df.corr(method="spearman")["outcome_2051_pct"].drop("outcome_2051_pct").sort_values(ascending=False)
print(corrs.round(3).to_string())
corrs.to_csv("/home/claude/arf_paper/psa_correlations.csv")
