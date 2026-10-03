"""
Extends the original 25-year simulation (simulation.py) to 35 years so the
Gen Z cohort's trajectory can be observed through age 60 (2061), not just
age 50 (2051). Uses the IDENTICAL model, parameters, and scenario definitions
as simulation.py -- this is a longer run of the same documented model, not a
new set of assumptions.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/claude/arf_paper")
from simulation import SCENARIOS, run_scenario, N_HOUSEHOLDS

N_YEARS_EXT = 36  # 2026 -> 2061 inclusive, cohort aged 25 -> 60

import hashlib

results_ext = {}
for name, params in SCENARIOS.items():
    scenario_seed = 20260814 + int(hashlib.md5(name.encode()).hexdigest(), 16) % 100000
    results_ext[name] = run_scenario(params, n=N_HOUSEHOLDS, years=N_YEARS_EXT, seed=scenario_seed)

paths_ext = pd.DataFrame({
    name: results_ext[name]["annual_oop_pct"].mean(axis=1) * 100
    for name in SCENARIOS
})
paths_ext.index = range(2026, 2026 + N_YEARS_EXT)
paths_ext.index.name = "year"
paths_ext.to_csv("/home/claude/arf_paper/annual_paths_extended.csv")

print("Extended trajectory (status_quo), key years:")
for yr in [2026, 2036, 2051, 2056, 2061]:
    if yr in paths_ext.index:
        row = paths_ext.loc[yr]
        print(f"  {yr} (age {25 + (yr-2026)}): status_quo={row['status_quo']:.2f}%  "
              f"reform={row['reform']:.2f}%  shock_heavy={row['shock_heavy']:.2f}%")

# Also save the households' final-year distributions at age 50 and age 60
# for a distributional comparison
summary_ages = {}
for name in SCENARIOS:
    burden_2051 = results_ext[name]["annual_oop_pct"][2051 - 2026] * 100  # age 50
    burden_2061 = results_ext[name]["annual_oop_pct"][2061 - 2026] * 100  # age 60
    summary_ages[name] = {
        "mean_at_age50": burden_2051.mean(), "median_at_age50": np.median(burden_2051),
        "mean_at_age60": burden_2061.mean(), "median_at_age60": np.median(burden_2061),
    }
summary_ages_df = pd.DataFrame(summary_ages).T
summary_ages_df.to_csv("/home/claude/arf_paper/summary_at_age50_60.csv")
print("\nSummary at age 50 vs age 60:")
print(summary_ages_df.round(2).to_string())
