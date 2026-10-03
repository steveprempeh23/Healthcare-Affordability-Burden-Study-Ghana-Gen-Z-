"""
Sensitivity analysis for the Gen Z Ghana healthcare-affordability simulation.

Approach: one-at-a-time (OAT) tornado analysis. Starting from the status-quo
scenario's calibrated parameters, we vary each uncertain parameter across a
plausible low/high range (holding all others fixed) and re-run the 25-year
simulation, recording the change in mean 2051 OOP burden. This tells you
which parameters the headline conclusion actually depends on -- exactly what
a reviewer will ask for before trusting the point estimates in the paper.

Run: python3 sensitivity_analysis.py
Outputs: sensitivity_results.csv, tornado_chart.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260814)

N_HOUSEHOLDS = 5_000   # smaller n for speed across many runs; still stable
N_YEARS = 26  # 2026-2051 inclusive, matching the corrected simulation.py

BASE_PARAMS = dict(
    nominal_wage_cagr=0.109,
    base_oop_fraction=0.06,
    oop_share_of_health=0.26,
    p_shock_year=0.12,
    shock_inflation_mean=0.30, shock_inflation_sd=0.10,
    stable_inflation_mean=0.10, stable_inflation_sd=0.03,
    import_premium_shock=0.06, import_premium_stable=0.02,
    p_health_crisis=0.08,
    severity_low=3.0, severity_high=7.0,
)

# Plausible low/high bounds for each uncertain parameter, informed by the
# range of values reported across the cited literature (see paper Table 1
# and its source notes for the low/high justification of each).
RANGES = {
    "p_health_crisis": (0.04, 0.15),        # SSA CHE incidence range 12.9-20.4%
                                              # vs. Ghana-specific declining-incidence
                                              # findings (Sataru et al. 2022) at low end
    "severity_high": (4.0, 10.0),            # catastrophic severity multiplier, upper bound
    "p_shock_year": (0.06, 0.25),            # historical shock frequency vs. shock-heavy case
    "import_premium_shock": (0.02, 0.12),    # currency pass-through strength
    "base_oop_fraction": (0.03, 0.10),       # non-crisis baseline OOP share
    "nominal_wage_cagr": (0.08, 0.13),       # around the historical 10.9% point estimate
    "oop_share_of_health": (0.20, 0.32),     # OOP % of current health expenditure
}


def run(params, n=N_HOUSEHOLDS, years=N_YEARS, seed=None):
    local_rng = np.random.default_rng(seed) if seed is not None else rng
    income_idx = np.ones(n)
    health_cost_idx = np.ones(n)
    final_burden = None
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
        income_idx *= (1 + params["nominal_wage_cagr"])
        health_cost_idx *= (1 + health_cost_growth)

        gap_ratio = health_cost_idx / income_idx
        crisis = local_rng.random(n) < params["p_health_crisis"]
        base_burden = params["base_oop_fraction"] * gap_ratio
        mult = local_rng.uniform(params["severity_low"], params["severity_high"], n)
        burden = np.where(crisis, base_burden * mult, base_burden)
        burden = np.clip(burden, 0.01, 1.5)
        final_burden = burden
    return final_burden.mean()


baseline_mean = run(BASE_PARAMS, seed=1)
print(f"Baseline (status-quo) mean 2051 OOP burden: {baseline_mean*100:.1f}%\n")

rows = []
for pname, (lo, hi) in RANGES.items():
    p_lo = dict(BASE_PARAMS); p_lo[pname] = lo
    p_hi = dict(BASE_PARAMS); p_hi[pname] = hi
    mean_lo = run(p_lo, seed=2)
    mean_hi = run(p_hi, seed=3)
    rows.append({
        "parameter": pname,
        "low_value": lo, "high_value": hi,
        "mean_oop_at_low": mean_lo * 100,
        "mean_oop_at_high": mean_hi * 100,
        "swing_pp": abs(mean_hi - mean_lo) * 100,
    })

results = pd.DataFrame(rows).sort_values("swing_pp", ascending=True)
results.to_csv("/home/claude/arf_paper/sensitivity_results.csv", index=False)
print(results.round(2).to_string(index=False))

# ---- Tornado chart ----
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
y = np.arange(len(results))
low = results["mean_oop_at_low"].values
high = results["mean_oop_at_high"].values
left = np.minimum(low, high)
width = np.abs(high - low)

ax.barh(y, width, left=left, color="#B3541E", alpha=0.75, height=0.55)
ax.axvline(baseline_mean * 100, color="#333333", linestyle="--", linewidth=1)
ax.text(baseline_mean * 100 + 0.5, len(results) - 0.3, "baseline", fontsize=8, color="#333333")

labels = {
    "p_health_crisis": "Catastrophic-event probability",
    "severity_high": "Severity multiplier (upper bound)",
    "p_shock_year": "Currency-shock frequency",
    "import_premium_shock": "Import cost pass-through (shock yrs)",
    "base_oop_fraction": "Baseline non-crisis OOP share",
    "nominal_wage_cagr": "Nominal wage growth (CAGR)",
    "oop_share_of_health": "OOP share of health expenditure",
}
ax.set_yticks(y)
ax.set_yticklabels([labels[p] for p in results["parameter"]], fontsize=9.5)
ax.set_xlabel("Mean household OOP burden in 2051 (% of income)")
ax.set_title("Sensitivity of 2051 projection to each uncertain parameter\n(one-at-a-time, status-quo scenario)", fontsize=11.5, loc="left", pad=12)
ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig("/home/claude/arf_paper/tornado_chart.png", facecolor="white")
print("\ntornado_chart.png written")
