"""
Reconciliation analysis for Manuscript Revision Requirement #1:
"Resolve the GLSS7 1.08% versus Simulation 8% Issue"

This script:
1. Precisely quantifies the mismatch between the original 8% (SSA-literature-
   blended) calibration and the author's own GLSS7 (2016/17) empirical
   finding (1.08% CHE incidence at the 10% threshold).
2. Builds an alternative "Ghana-empirical-calibrated" parameterization,
   recalibrating the annual catastrophic-event probability so the model's
   OWN year-1 implied CHE(10%) incidence matches the real empirical value.
3. Re-runs all three scenarios (status quo, reform, shock-heavy) under both
   calibrations, through both horizons (2051 and 2061).
4. Reports whether the paper's main conclusions (scenario ordering, direction
   of policy effect) hold under the alternative, much lower calibration.

Definitional note (required by the revision instructions): the simulation's
"catastrophic health event" is a stochastic ANNUAL HAZARD -- an input
probability that a household experiences a health shock large enough to
multiply that year's OOP burden by a random severity factor (3-7x baseline).
This is NOT the same quantity as GLSS7's CHE INCIDENCE, which is a REALIZED,
OBSERVED outcome (the share of households whose OOP burden, however it arose,
crossed a stated threshold in one specific survey year). The two are related
but not interchangeable: a household can experience the simulation's "hazard"
without crossing 10% of income (if severity happens to be low and the
cost/income gap ratio is still small), and conversely the empirical incidence
reflects the full population's realized experience, not an underlying latent
hazard rate. This script treats the empirical 1.08% as the best available
real-world anchor for what fraction of households cross the 10% threshold in
a GIVEN YEAR under 2016/17 conditions, and back-solves the hazard parameter
so the model reproduces that anchor in its own initial year.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/claude/arf_paper")
from simulation import SCENARIOS as ORIGINAL_SCENARIOS, run_scenario, N_HOUSEHOLDS, BASE_OOP_FRACTION

rng_check = np.random.default_rng(1)

# ---------------------------------------------------------------------------
# Step 1: back-solve the hazard probability that reproduces the empirical
# 10%-threshold CHE incidence (1.08%) in the model's OWN year-1 mechanics.
# ---------------------------------------------------------------------------
threshold = 0.10
severity_low, severity_high = 3.0, 7.0
crisis_burden_floor = BASE_OOP_FRACTION * severity_low
crisis_burden_ceiling = BASE_OOP_FRACTION * severity_high
if threshold <= crisis_burden_floor:
    p_cross_given_crisis = 1.0
else:
    p_cross_given_crisis = max(0.0, (crisis_burden_ceiling - threshold) / (crisis_burden_ceiling - crisis_burden_floor))

EMPIRICAL_CHE_10PCT = 0.0108
GHANA_CALIBRATED_P_HEALTH_CRISIS = EMPIRICAL_CHE_10PCT / p_cross_given_crisis

print("=" * 78)
print("STEP 1: Back-solving the Ghana-empirical-calibrated hazard parameter")
print("=" * 78)
print(f"Crisis-year burden range under current severity multiplier (3x-7x):")
print(f"  {crisis_burden_floor*100:.1f}% to {crisis_burden_ceiling*100:.1f}% of income")
print(f"P(burden > 10% | crisis event occurs) = {p_cross_given_crisis*100:.1f}%")
print(f"Target: empirical CHE(10%) = {EMPIRICAL_CHE_10PCT*100:.2f}%")
print(f"=> Ghana-calibrated p_health_crisis = {GHANA_CALIBRATED_P_HEALTH_CRISIS*100:.3f}%")
print(f"   (versus original SSA-literature-blended value: 8.000%)")
print()

# ---------------------------------------------------------------------------
# Step 2: build the alternative scenario set, preserving the SAME relative
# structure the original scenarios used, re-anchored to the Ghana value.
# ---------------------------------------------------------------------------
GHANA_CALIBRATED_SCENARIOS = {
    "status_quo": dict(ORIGINAL_SCENARIOS["status_quo"]),
    "reform": dict(ORIGINAL_SCENARIOS["reform"]),
    "shock_heavy": dict(ORIGINAL_SCENARIOS["shock_heavy"]),
}
relative_reform = ORIGINAL_SCENARIOS["reform"]["p_health_crisis"] / ORIGINAL_SCENARIOS["status_quo"]["p_health_crisis"]
relative_shock = ORIGINAL_SCENARIOS["shock_heavy"]["p_health_crisis"] / ORIGINAL_SCENARIOS["status_quo"]["p_health_crisis"]

GHANA_CALIBRATED_SCENARIOS["status_quo"]["p_health_crisis"] = GHANA_CALIBRATED_P_HEALTH_CRISIS
GHANA_CALIBRATED_SCENARIOS["reform"]["p_health_crisis"] = GHANA_CALIBRATED_P_HEALTH_CRISIS * relative_reform
GHANA_CALIBRATED_SCENARIOS["shock_heavy"]["p_health_crisis"] = GHANA_CALIBRATED_P_HEALTH_CRISIS * relative_shock

# ---------------------------------------------------------------------------
# Step 3: re-run both calibrations, both horizons (2051 / 25yr, 2061 / 35yr)
# ---------------------------------------------------------------------------
def run_all(scenario_set, years):
    out = {}
    for name, params in scenario_set.items():
        scenario_seed = 20260814 + int(__import__("hashlib").md5(name.encode()).hexdigest(), 16) % 100000
        res = run_scenario(params, n=N_HOUSEHOLDS, years=years, seed=scenario_seed)
        out[name] = res["annual_oop_pct"].mean(axis=1) * 100
    return out

print("=" * 78)
print("STEP 2/3: Re-running all scenarios under both calibrations")
print("=" * 78)

orig_26 = run_all(ORIGINAL_SCENARIOS, 26)   # 2026-2051 inclusive
ghana_26 = run_all(GHANA_CALIBRATED_SCENARIOS, 26)
orig_36 = run_all(ORIGINAL_SCENARIOS, 36)   # 2026-2061 inclusive
ghana_36 = run_all(GHANA_CALIBRATED_SCENARIOS, 36)

comparison = pd.DataFrame({
    "scenario": ["status_quo", "reform", "shock_heavy"] * 2,
    "calibration": ["Original (8% SSA-blended)"] * 3 + ["Ghana-empirical (GLSS7-anchored)"] * 3,
    "mean_2026": [orig_26[s][0] for s in ["status_quo", "reform", "shock_heavy"]] +
                 [ghana_26[s][0] for s in ["status_quo", "reform", "shock_heavy"]],
    "mean_2051": [orig_26[s][25] for s in ["status_quo", "reform", "shock_heavy"]] +
                 [ghana_26[s][25] for s in ["status_quo", "reform", "shock_heavy"]],
    "mean_2061": [orig_36[s][35] for s in ["status_quo", "reform", "shock_heavy"]] +
                 [ghana_36[s][35] for s in ["status_quo", "reform", "shock_heavy"]],
})
comparison.to_csv("/home/claude/arf_paper/calibration_comparison.csv", index=False)
print(comparison.round(2).to_string(index=False))
print()

print("=" * 78)
print("STEP 4: Do the main conclusions still hold under the lower calibration?")
print("=" * 78)
for horizon, col in [("2051", "mean_2051"), ("2061", "mean_2061")]:
    ghana_rows = comparison[comparison["calibration"].str.contains("Ghana")]
    order_preserved = (
        ghana_rows[ghana_rows.scenario == "reform"][col].values[0]
        < ghana_rows[ghana_rows.scenario == "status_quo"][col].values[0]
        < ghana_rows[ghana_rows.scenario == "shock_heavy"][col].values[0]
    )
    print(f"{horizon}: scenario ordering (reform < status quo < shock-heavy) "
          f"{'PRESERVED' if order_preserved else 'NOT PRESERVED'} under Ghana-empirical calibration")

print()
print("Absolute magnitude comparison, status-quo scenario:")
sq_orig_2051 = comparison.query("scenario=='status_quo' and calibration.str.contains('Original')")["mean_2051"].values[0]
sq_ghana_2051 = comparison.query("scenario=='status_quo' and calibration.str.contains('Ghana')")["mean_2051"].values[0]
print(f"  2051: {sq_orig_2051:.2f}% (original) vs {sq_ghana_2051:.2f}% (Ghana-empirical) "
      f"-- a {sq_orig_2051/sq_ghana_2051:.1f}x difference")
