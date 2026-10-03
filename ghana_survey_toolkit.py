"""
Ghana Survey Toolkit -- auto-discovery + analysis for GLSS7 / AHIES microdata.

WHY THIS EXISTS:
You have dozens of .dta/.sav files across GLSS7 and AHIES 2022-2024, with
undocumented variable names. Instead of manually opening a codebook for
every file, this scans everything, reads ONLY the metadata (variable names
+ labels -- fast, does not load the actual data rows), and searches for
columns matching what you need: household ID, sample weights, health/OOP
spending, total consumption/income, region.

TWO STEPS:

  Step 1 -- SCAN (find out what you actually have)
      python3 ghana_survey_toolkit.py scan --root "C:\\path\\to\\health reaserch"
    This walks every .dta/.sav file under --root, and writes
    scan_report.csv listing every file with its likely-relevant columns
    flagged by category. Read this file first. It does NOT load full data,
    so it's safe to run even on the huge GLSS7 PartB files.

  Step 2 -- ANALYZE (once you know which files/columns to use)
      python3 ghana_survey_toolkit.py analyze --config config.json
    You fill in config.json (a template is auto-created the first time you
    run `analyze` without one) based on what scan_report.csv told you, then
    this merges the files, applies survey weights properly, and computes:
      - weighted mean/median OOP burden (% of consumption/income)
      - catastrophic health expenditure (CHE) incidence at 10/20/25%
        thresholds
      - breakdown by region and by consumption decile (the regressivity
        check)

Requires: pandas, numpy, pyreadstat  (pip install pyreadstat --break-system-packages)
"""

import argparse
import json
import os
import sys
import re
import numpy as np
import pandas as pd
import pyreadstat

# ---------------------------------------------------------------------------
# Category keywords -- edit/extend these if scan misses something relevant
# to your specific research question.
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "household_id": ["household id", "hhid", "hh id", "case id", "houseid", "hid"],
    "panel_id": ["panel id", "panel hhid", "tracking id", "persistent id"],
    "weight": ["weight", "sample weight", "household weight", "expansion factor", "hhwt"],
    "health_spending": [
        "out-of-pocket", "out of pocket", "health expenditure", "hospital",
        "medical expense", "cost of treatment", "amount paid", "drug",
        "clinic", "nhis", "health insurance", "illness", "sick", "consult",
        "medicine", "treatment cost",
    ],
    "consumption_income": [
        "total expenditure", "consumption", "income", "expenditure per capita",
        "welfare", "poverty line", "annual expenditure", "total income",
    ],
    "region": ["region", "district", "locality", "rural", "urban", "ecozone"],
    "demographics": ["age", "sex", "gender", "education", "occupation"],
}


def scan_file(path):
    """Read only metadata (variable names + labels) from a .dta or .sav file."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".dta":
            _, meta = pyreadstat.read_dta(path, metadataonly=True)
        elif ext == ".sav":
            _, meta = pyreadstat.read_sav(path, metadataonly=True)
        else:
            return None
    except Exception as e:
        return {"error": str(e)}

    col_names = meta.column_names
    col_labels = meta.column_labels or [""] * len(col_names)
    return list(zip(col_names, col_labels))


def match_categories(col_name, col_label):
    text = f"{col_name} {col_label or ''}".lower()
    matches = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            matches.append(category)
    return matches


def cmd_scan(args):
    rows = []
    n_files = 0
    n_errors = 0
    for dirpath, _, filenames in os.walk(args.root):
        for fn in filenames:
            if not fn.lower().endswith((".dta", ".sav")):
                continue
            full_path = os.path.join(dirpath, fn)
            n_files += 1
            print(f"Scanning: {full_path}")
            result = scan_file(full_path)
            if result is None:
                continue
            if isinstance(result, dict) and "error" in result:
                n_errors += 1
                rows.append({"file": full_path, "column": "", "label": "",
                              "categories": "", "error": result["error"]})
                continue
            for col_name, col_label in result:
                cats = match_categories(col_name, col_label)
                if cats:
                    rows.append({
                        "file": full_path, "column": col_name, "label": col_label,
                        "categories": ";".join(cats), "error": "",
                    })

    report = pd.DataFrame(rows)
    out_path = os.path.join(args.root, "scan_report.csv")
    report.to_csv(out_path, index=False)

    print(f"\nScanned {n_files} files ({n_errors} could not be read).")
    print(f"Report written to: {out_path}")
    if not report.empty:
        print("\nFiles with the most category matches (open these first):")
        counts = report[report["error"] == ""].groupby("file")["categories"].count()
        print(counts.sort_values(ascending=False).head(15).to_string())
        print("\nOpen scan_report.csv, filter by 'categories' column, and use it")
        print("to fill in config.json for the analyze step.")
    else:
        print("\nNo matches found. Either the keyword list needs extending for")
        print("this survey's naming conventions, or the files could not be read")
        print("(check the 'error' column if present).")


CONFIG_TEMPLATE = {
    "household_file": "PATH TO FILE WITH household id + weight + region",
    "health_file": "PATH TO FILE WITH health/OOP spending columns",
    "consumption_file": "PATH TO FILE WITH total consumption/income aggregate",
    "columns": {
        "hh_id": "COLUMN NAME for household id (must match across all 3 files)",
        "weight": "COLUMN NAME for sample weight",
        "region": "COLUMN NAME for region",
        "age": "OPTIONAL: COLUMN NAME for age (household head or member) -- omit or leave blank to skip age breakdown",
        "oop_components": ["LIST", "of health spending COLUMN NAMES to sum"],
        "total_consumption": "COLUMN NAME for total household consumption/income",
    },
    "che_thresholds": [0.10, 0.20, 0.25],
}


def load_dta_or_sav(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dta":
        df, _ = pyreadstat.read_dta(path)
    elif ext == ".sav":
        df, _ = pyreadstat.read_sav(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return df


AGE_BINS = [0, 18, 30, 40, 50, 60, 70, 150]
AGE_LABELS = ["<18", "18-29", "30-39", "40-49", "50-59", "60-69", "70+"]


def run_analysis(cfg, out_dir):
    """Shared analysis logic used by both `analyze` (manual config) and
    `auto` (self-discovered config)."""
    hh_id = cfg["columns"]["hh_id"]
    weight_col = cfg["columns"]["weight"]
    region_col = cfg["columns"]["region"]
    oop_cols = cfg["columns"]["oop_components"]
    cons_col = cfg["columns"]["total_consumption"]
    age_col = cfg["columns"].get("age", None)  # optional
    sex_col = cfg["columns"].get("sex", None)  # optional
    literacy_col = cfg["columns"].get("literacy", None)  # optional
    education_col = cfg["columns"].get("education", None)  # optional
    rel_col = cfg["columns"].get("relationship_to_head", None)  # optional
    head_code = cfg["columns"].get("head_code", 1)  # optional, default GLSS convention
    budget_category_columns = cfg["columns"].get("budget_category_columns", {}) or {}  # optional

    print("Loading household file...")
    hh_raw = load_dta_or_sav(cfg["household_file"])
    n_hh_rows = len(hh_raw)
    n_hh_unique = hh_raw[hh_id].nunique()
    if n_hh_rows != n_hh_unique:
        print(f"WARNING: household_file has {n_hh_rows:,} rows but only")
        print(f"  {n_hh_unique:,} unique {hh_id} values -- this file is NOT")
        print(f"  one-row-per-household. Merging as-is will duplicate")
        print(f"  households and bias every weighted statistic below.")
        if rel_col and rel_col in hh_raw.columns:
            n_before = len(hh_raw)
            hh_raw = hh_raw[hh_raw[rel_col] == head_code]
            n_after = len(hh_raw)
            print(f"  Filtered to head-of-household rows only, using")
            print(f"  '{rel_col}' == {head_code}: {n_before:,} -> {n_after:,} rows.")
            missing_heads = n_hh_unique - n_after
            if missing_heads != 0:
                print(f"  NOTE: {n_hh_unique:,} unique households but {n_after:,} head")
                print(f"  rows found after filtering -- {abs(missing_heads):,} household(s)")
                print(f"  {'missing a head row' if missing_heads > 0 else 'have more than one head row'}.")
                print(f"  Falling back to first-row-per-household for any household")
                print(f"  without exactly one matched head row.")
                hh_raw = pd.concat([hh_raw, load_dta_or_sav(cfg['household_file'])]) \
                    .drop_duplicates(subset=[hh_id], keep="first")
            hh_raw = hh_raw.drop_duplicates(subset=[hh_id], keep="first")
        else:
            print(f"  Deduplicating by keeping the first row per household id.")
            print(f"  (If this file is meant to be person-level, pick a different")
            print(f"  file for household_file in the config instead.)")
            hh_raw = hh_raw.drop_duplicates(subset=[hh_id], keep="first")
            if age_col or sex_col:
                print(f"  NOTE: household_file has multiple rows per household (person-")
                print(f"  level roster) and no relationship-to-head column was found.")
                print(f"  'age'/'sex' were taken from the FIRST row per household, which")
                print(f"  is usually but not guaranteed to be the household head.")
    extra_cols = []
    for c in [age_col, sex_col, literacy_col, education_col]:
        if c and c not in extra_cols:
            extra_cols.append(c)
    keep_cols = [hh_id, weight_col, region_col] + extra_cols
    hh = hh_raw[keep_cols]

    print("Loading health file...")
    health = load_dta_or_sav(cfg["health_file"])
    missing = [c for c in oop_cols if c not in health.columns]
    if missing:
        print(f"WARNING: these oop_components are not in the health file: {missing}")
        oop_cols = [c for c in oop_cols if c in health.columns]
    if not oop_cols:
        raise ValueError("No valid oop_components columns found in health file.")
    if health[hh_id].duplicated().any():
        health = health.groupby(hh_id)[oop_cols].sum().reset_index()
    health["total_oop"] = health[oop_cols].sum(axis=1)

    print("Loading consumption file...")
    cons_raw = load_dta_or_sav(cfg["consumption_file"])
    if cons_raw[hh_id].duplicated().any():
        n_before = len(cons_raw)
        n_unique = cons_raw[hh_id].nunique()
        print(f"WARNING: consumption_file has {n_before:,} rows but only")
        print(f"  {n_unique:,} unique {hh_id} values. Deduplicating by")
        print(f"  keeping the first row per household id.")
        cons_raw = cons_raw.drop_duplicates(subset=[hh_id], keep="first")
    cons = cons_raw[[hh_id, cons_col]]

    df = hh.merge(health[[hh_id, "total_oop"]], on=hh_id, how="left")
    df = df.merge(cons, on=hh_id, how="left")
    df["total_oop"] = df["total_oop"].fillna(0)
    df = df.dropna(subset=[cons_col, weight_col])
    df = df[df[cons_col] > 0]

    df["oop_share"] = df["total_oop"] / df[cons_col]
    w = df[weight_col]

    print(f"\nAnalysis sample: {len(df):,} households (after merge and cleaning)\n")

    weighted_mean = np.average(df["oop_share"], weights=w)
    sorted_df = df.sort_values("oop_share")
    cum_w = sorted_df[weight_col].cumsum() / sorted_df[weight_col].sum()
    # Use positional argmax (via .values), not idxmax -- idxmax returns an
    # index LABEL, but the sorted/merged DataFrame's index is no longer a
    # clean 0..n-1 range at this point, so pairing idxmax with .iloc (which
    # is positional) reads the wrong row or crashes outright.
    median_pos = np.argmax((cum_w >= 0.5).to_numpy())
    weighted_median = sorted_df["oop_share"].iloc[median_pos]

    print(f"Weighted mean OOP share of consumption/income:   {weighted_mean*100:.2f}%")
    print(f"Weighted median OOP share of consumption/income: {weighted_median*100:.2f}%\n")

    for t in cfg["che_thresholds"]:
        che_incidence = np.average(df["oop_share"] > t, weights=w)
        print(f"Catastrophic health expenditure incidence (>{int(t*100)}% threshold): {che_incidence*100:.2f}% of households")

    print("\nBy region:")
    by_region = df.groupby(region_col).apply(
        lambda g: pd.Series({
            "weighted_mean_oop_share_pct": np.average(g["oop_share"], weights=g[weight_col]) * 100,
            "n_households": len(g),
        })
    )
    print(by_region.round(2).to_string())

    print("\nBy consumption decile (regressivity check -- lower deciles should")
    print("show HIGHER oop_share if the pattern from the literature holds):")
    df["decile"] = pd.qcut(df[cons_col], 10, labels=range(1, 11))
    by_decile = df.groupby("decile").apply(
        lambda g: pd.Series({
            "weighted_mean_oop_share_pct": np.average(g["oop_share"], weights=g[weight_col]) * 100,
        })
    )
    print(by_decile.round(2).to_string())

    if age_col and age_col in df.columns:
        print(f"\nBy age group (using '{age_col}' from household_file):")
        df["age_group"] = pd.cut(df[age_col], bins=AGE_BINS, labels=AGE_LABELS, right=False)
        by_age = df.groupby("age_group", observed=True).apply(
            lambda g: pd.Series({
                "weighted_mean_oop_share_pct": np.average(g["oop_share"], weights=g[weight_col]) * 100 if len(g) else np.nan,
                "n_households": len(g),
            })
        )
        print(by_age.round(2).to_string())
        by_age.to_csv(os.path.join(out_dir, "results_by_age.csv"))
        top_age = by_age["weighted_mean_oop_share_pct"].idxmax() if not by_age.empty else None
        if top_age:
            print(f"\nHighest mean OOP burden: age group {top_age} "
                  f"({by_age.loc[top_age, 'weighted_mean_oop_share_pct']:.2f}% of income)")

    if sex_col and sex_col in df.columns:
        print(f"\nBy sex (using '{sex_col}' from household_file):")
        by_sex = df.groupby(sex_col, observed=True).apply(
            lambda g: pd.Series({
                "weighted_mean_oop_share_pct": np.average(g["oop_share"], weights=g[weight_col]) * 100 if len(g) else np.nan,
                "n_households": len(g),
            })
        )
        print(by_sex.round(2).to_string())
        sex_labels_found = None
        try:
            ext = os.path.splitext(cfg["household_file"])[1].lower()
            if ext == ".dta":
                _, hh_meta = pyreadstat.read_dta(cfg["household_file"], metadataonly=True)
            else:
                _, hh_meta = pyreadstat.read_sav(cfg["household_file"], metadataonly=True)
            vl = getattr(hh_meta, "variable_value_labels", {}) or {}
            sex_labels_found = vl.get(sex_col)
        except Exception:
            pass
        if sex_labels_found:
            print(f"Confirmed via file's own value labels: {sex_labels_found}")
        else:
            print("(No value labels found for this column in the file's metadata --")
            print(" check the survey codebook to confirm which code is male/female;")
            print(" GLSS typically codes 1=male, 2=female, but this was not verified.)")
        by_sex.to_csv(os.path.join(out_dir, "results_by_sex.csv"))

    if literacy_col and literacy_col in df.columns:
        print(f"\nBy literacy status (using '{literacy_col}' from household_file):")
        by_lit = df.groupby(literacy_col, observed=True).apply(
            lambda g: pd.Series({
                "weighted_mean_oop_share_pct": np.average(g["oop_share"], weights=g[weight_col]) * 100 if len(g) else np.nan,
                "n_households": len(g),
            })
        )
        print(by_lit.round(2).to_string())
        lit_labels_found = None
        try:
            ext = os.path.splitext(cfg["household_file"])[1].lower()
            hh_meta_fn = pyreadstat.read_dta if ext == ".dta" else pyreadstat.read_sav
            _, hh_meta = hh_meta_fn(cfg["household_file"], metadataonly=True)
            vl = getattr(hh_meta, "variable_value_labels", {}) or {}
            lit_labels_found = vl.get(literacy_col)
        except Exception:
            pass
        if lit_labels_found:
            print(f"Confirmed via file's own value labels: {lit_labels_found}")
        else:
            print("(No value labels found -- check codebook to confirm which code means literate.)")
        by_lit.to_csv(os.path.join(out_dir, "results_by_literacy.csv"))

    if education_col and education_col in df.columns:
        print(f"\nBy education level (using '{education_col}' from household_file):")
        by_edu = df.groupby(education_col, observed=True).apply(
            lambda g: pd.Series({
                "weighted_mean_oop_share_pct": np.average(g["oop_share"], weights=g[weight_col]) * 100 if len(g) else np.nan,
                "n_households": len(g),
            })
        )
        print(by_edu.round(2).to_string())
        by_edu.to_csv(os.path.join(out_dir, "results_by_education.csv"))

    if age_col and age_col in df.columns and literacy_col and literacy_col in df.columns:
        print(f"\nAge x literacy interaction (tests whether the literacy gap in OOP")
        print(f"burden WIDENS with age -- i.e. whether illiteracy's effect compounds")
        print(f"over a lifetime, rather than just existing on average):")
        df["age_group"] = pd.cut(df[age_col], bins=AGE_BINS, labels=AGE_LABELS, right=False)
        crosstab = df.groupby(["age_group", literacy_col], observed=True).apply(
            lambda g: pd.Series({
                "weighted_mean_oop_share_pct": np.average(g["oop_share"], weights=g[weight_col]) * 100 if len(g) else np.nan,
                "n_households": len(g),
            })
        )
        pivot = crosstab["weighted_mean_oop_share_pct"].unstack(literacy_col)
        n_pivot = crosstab["n_households"].unstack(literacy_col)
        print(pivot.round(2).to_string())
        if pivot.shape[1] == 2:
            codes = list(pivot.columns)
            gap = pivot[codes[1]] - pivot[codes[0]]
            print(f"\nGap between literacy code {codes[1]} and code {codes[0]}, by age group")
            print("(a rising trend here would support the 'compounds over time' hypothesis;")
            print("a flat or noisy trend would not -- check the n_households table too,")
            print("since a gap based on a small cell size is not reliable):")
            print(gap.round(3).to_string())
        pivot.to_csv(os.path.join(out_dir, "results_age_literacy_interaction.csv"))
        n_pivot.to_csv(os.path.join(out_dir, "results_age_literacy_interaction_n.csv"))

    if budget_category_columns:
        print(f"\nHousehold budget breakdown (weighted mean share of {cons_col}, all categories):")
        cons_full = load_dta_or_sav(cfg["consumption_file"])
        cat_cols = [c for c in budget_category_columns if c in cons_full.columns]
        cat_df = cons_full[[hh_id] + cat_cols].drop_duplicates(subset=[hh_id], keep="first")
        merged_budget = df[[hh_id, weight_col]].merge(cat_df, on=hh_id, how="left")
        w2 = merged_budget[weight_col]
        budget_shares = {}
        total_col_sum = merged_budget[cat_cols].sum(axis=1)
        for c in cat_cols:
            share = merged_budget[c] / total_col_sum.replace(0, np.nan)
            budget_shares[budget_category_columns[c]] = np.average(share.fillna(0), weights=w2) * 100
        budget_series = pd.Series(budget_shares).sort_values(ascending=False)
        print(budget_series.round(2).to_string())
        budget_series.to_csv(os.path.join(out_dir, "results_budget_shares.csv"), header=["mean_share_pct"])

    by_region.to_csv(os.path.join(out_dir, "results_by_region.csv"))
    by_decile.to_csv(os.path.join(out_dir, "results_by_decile.csv"))
    print(f"\nDetailed results written to results_by_region.csv and results_by_decile.csv")


def cmd_analyze(args):
    if not os.path.exists(args.config):
        with open(args.config, "w") as f:
            json.dump(CONFIG_TEMPLATE, f, indent=2)
        print(f"No config found. Template written to: {args.config}")
        print("Fill it in using scan_report.csv, then re-run this command.")
        return

    with open(args.config) as f:
        cfg = json.load(f)

    # Resolve relative paths in the config relative to the config file's own
    # location, not the current working directory -- avoids "file not found"
    # errors depending on where you happen to run the script from.
    config_dir = os.path.dirname(os.path.abspath(args.config))
    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(config_dir, p)
    cfg["household_file"] = resolve(cfg["household_file"])
    cfg["health_file"] = resolve(cfg["health_file"])
    cfg["consumption_file"] = resolve(cfg["consumption_file"])

    run_analysis(cfg, config_dir)


# ---------------------------------------------------------------------------
# AUTO MODE -- self-discovers files and columns, no manual config editing.
#
# Uses two layers of heuristics:
#   1. Filename priority rules specific to how Ghana Statistical Service
#      names GLSS7 files (e.g. the numbered EXPxxxx aggregate files, the
#      POV welfare file, the secN module files). These are the strongest,
#      most reliable signal when present.
#   2. Falls back to the keyword-match scoring from `scan` when filename
#      rules don't apply (e.g. for AHIES files, which are named differently).
# ---------------------------------------------------------------------------

FILENAME_PRIORITY = {
    # (regex pattern, category this file is a strong candidate for)
    "household_file": [r"sec1_5", r"sec0\b", r"loc_upd", r"sec1\b"],
    "health_file": [r"exphlth", r"sec3b", r"sec3a", r"sec3c", r"sec3d", r"sec3e", r"sec3f"],
    "consumption_file": [r"_pov_", r"e_final", r"ghana_2017_e\b"],
}

# Maps the "file role" names above to the keyword category used for scoring
# columns within that file (see CATEGORY_KEYWORDS).
FILE_ROLE_TO_KEYWORD_CATEGORY = {
    "household_file": "household_id",
    "health_file": "health_spending",
    "consumption_file": "consumption_income",
}


def discover_files(root, ext="dta"):
    """Walk root, read metadata for every file of the given extension, and
    return a dict: {filepath: {"columns": [...], "labels": [...], "types": {...}}}
    """
    inventory = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(f".{ext}"):
                continue
            full_path = os.path.join(dirpath, fn)
            try:
                _, meta = pyreadstat.read_dta(full_path, metadataonly=True) if ext == "dta" \
                    else pyreadstat.read_sav(full_path, metadataonly=True)
            except Exception as e:
                print(f"  (skipped, could not read: {full_path} -- {e})")
                continue
            col_names = meta.column_names
            col_labels = meta.column_labels or [""] * len(col_names)
            try:
                types = dict(zip(meta.column_names, meta.readstat_variable_types.values())) \
                    if hasattr(meta, "readstat_variable_types") else {}
            except Exception:
                types = {}
            try:
                value_labels = dict(meta.variable_value_labels) if hasattr(meta, "variable_value_labels") else {}
            except Exception:
                value_labels = {}
            inventory[full_path] = {
                "columns": col_names, "labels": col_labels, "types": types,
                "value_labels": value_labels,
            }
    return inventory


def score_file(info, category):
    """Count how many columns in this file match a category's keywords."""
    count = 0
    matched_cols = []
    for col, label in zip(info["columns"], info["labels"]):
        text = f"{col} {label or ''}".lower()
        if any(kw in text for kw in CATEGORY_KEYWORDS[category]):
            count += 1
            matched_cols.append(col)
    return count, matched_cols


def pick_file_for_category(inventory, file_role, exclude=()):
    """Pick the best file for a file role (household_file/health_file/
    consumption_file): filename priority rules first, then fall back to
    keyword-match score using the mapped keyword category."""
    keyword_category = FILE_ROLE_TO_KEYWORD_CATEGORY.get(file_role, file_role)
    candidates = {p: info for p, info in inventory.items() if p not in exclude}

    # Layer 1: filename priority rules
    for pattern in FILENAME_PRIORITY.get(file_role, []):
        for path in candidates:
            if re.search(pattern, os.path.basename(path), re.IGNORECASE):
                score, cols = score_file(candidates[path], keyword_category)
                if score > 0 or file_role == "household_file":
                    return path, cols

    # Layer 2: keyword-score fallback -- highest match count wins
    best_path, best_score, best_cols = None, 0, []
    for path, info in candidates.items():
        score, cols = score_file(info, keyword_category)
        if score > best_score:
            best_path, best_score, best_cols = path, score, cols
    return best_path, best_cols


def pick_age_column(info):
    """Find a column that represents a person's age (household head or
    member). Uses whole-word matching on the label to avoid false positives
    like matching 'age' inside 'marriage' or 'average'."""
    candidates = []
    for col, label in zip(info["columns"], info["labels"]):
        label_l = (label or "").lower()
        col_l = col.lower()
        # whole-word match only, not substring -- "marriage" contains "age"
        # but is not an age column
        label_words = re.findall(r"[a-z]+", label_l)
        if col_l == "age" or "age" in label_words:
            candidates.append(col)
    if not candidates:
        return None
    # prefer the exact column name "age" if present
    for c in candidates:
        if c.lower() == "age":
            return c
    return candidates[0]


SEX_KEYWORDS = ["sex of", "gender of", "sex", "gender", "male/female", "male or female"]


def pick_sex_column(info):
    """Find a column representing sex/gender of household head or member."""
    candidates = []
    for col, label in zip(info["columns"], info["labels"]):
        label_l = (label or "").lower()
        col_l = col.lower()
        if col_l in ("sex", "gender", "s1q2") or any(kw in label_l for kw in SEX_KEYWORDS):
            candidates.append(col)
    if not candidates:
        return None
    for c in candidates:
        if c.lower() in ("sex", "gender"):
            return c
    return candidates[0]


LITERACY_KEYWORDS = [
    "literate", "literacy", "read and write", "able to read", "can read",
]
EDUCATION_LEVEL_KEYWORDS = [
    "highest level of education", "highest educational", "educational attainment",
    "level of education", "educational level",
]
YEARS_EDUCATION_KEYWORDS = [
    "years of education", "years of schooling", "years spent in school",
]


def pick_literacy_column(info):
    """Find a column representing literacy status specifically (distinct
    from years of education or educational attainment level)."""
    for col, label in zip(info["columns"], info["labels"]):
        label_l = (label or "").lower()
        if any(kw in label_l for kw in LITERACY_KEYWORDS):
            return col
    return None


def pick_education_column(info):
    """Find a column representing educational attainment level or years of
    education, as a fallback/complement to literacy status specifically."""
    for col, label in zip(info["columns"], info["labels"]):
        label_l = (label or "").lower()
        if any(kw in label_l for kw in EDUCATION_LEVEL_KEYWORDS + YEARS_EDUCATION_KEYWORDS):
            return col
    return None


# Common GLSS/AHIES expenditure category column names (from the POV/welfare
# aggregate file's TOTxxx-style columns) mapped to readable labels. Not every
# file will have every column -- only present ones are used.
BUDGET_CATEGORY_COLUMNS = {
    "TOTFOOD": "Food", "TOTALCH": "Alcohol/Tobacco", "TOTCLTH": "Clothing & Footwear",
    "TOTHOUS": "Housing & Utilities", "TOTFURN": "Furnishings", "TOTHLTH": "Health",
    "TOTTRSP": "Transport", "TOTCMNQ": "Communication", "TOTRCRE": "Recreation & Culture",
    "TOTEDUC": "Education", "TOTHOTL": "Restaurants & Hotels", "TOTMISC": "Miscellaneous",
}


RELATIONSHIP_HEAD_KEYWORDS = ["relationship to head", "relation to head", "relationship with head", "reltohead", "relhead"]
HEAD_VALUE_LABEL_HINTS = ["head"]  # matches "Head", "Household head", etc.


def pick_relationship_column(info):
    """Find a column recording each person's relationship to the household
    head (e.g. 1=Head, 2=Spouse, 3=Child...). Used to correctly identify the
    head's row in a person-level roster, instead of guessing via row order."""
    for col, label in zip(info["columns"], info["labels"]):
        label_l = (label or "").lower()
        col_l = col.lower()
        if col_l in ("reltohead", "relhead") or any(kw in label_l for kw in RELATIONSHIP_HEAD_KEYWORDS):
            return col
    return None


def find_head_code(info, rel_col):
    """Given a relationship-to-head column, find which numeric code means
    'head', using the file's own value labels if present. Falls back to the
    conventional code 1 with a printed caveat if no value labels exist."""
    value_labels = info.get("value_labels", {})
    mapping = value_labels.get(rel_col)
    if mapping:
        for code, label in mapping.items():
            if any(hint in str(label).lower() for hint in HEAD_VALUE_LABEL_HINTS):
                return code, True  # confirmed via value labels
    return 1, False  # unconfirmed guess -- conventional GLSS code


def pick_id_column(info):
    """Among household_id matches, prefer the shortest column name --
    ID variables are almost always named tersely (hid, hhid) vs. long
    descriptive labels that happen to mention 'id'."""
    _, cols = score_file(info, "household_id")
    if not cols:
        return None
    return min(cols, key=len)


# Column labels containing any of these words are almost certainly a
# SUBCATEGORY of expenditure (food, housing, health, alcohol, etc.), not the
# household's true grand total -- even if the label also happens to contain
# the word "total". Files like GLSS7's POV aggregate have a dozen "TOTxxx"
# subcategory columns (TOTFOOD, TOTCLTH, TOTHLTH, TOTALCH...) sitting right
# next to the one or two genuine grand-total columns, and a plain keyword
# match can't tell them apart -- this exclusion list can.
SUBCATEGORY_EXCLUDE_WORDS = [
    "food", "alcohol", "tobacco", "narcotic", "cloth", "footwear", "footware",
    "housing", "furnish", "furniture", "health", "transport", "communication",
    "telephone", "recreation", "cultural", "education", "hotel", "restaurant",
    "miscell", "personal care", "insurance", "rental", "beverage",
]

# Strong positive signals that a column IS the genuine household grand total.
GRAND_TOTAL_LABEL_HINTS = [
    "total annual nominal household expenditure",
    "total annual household expenditure",
    "total household expenditure",
    "total real household expenditure",
    "total annual real household expenditure",
]
GRAND_TOTAL_NAME_HINTS = ["hhexp_n", "hhexp_r", "hhexp", "welfare", "totexp"]


POVERTY_THRESHOLD_HINTS = ["poverty line", "povline", "pl_abs", "pl_ext"]


def pick_total_consumption_column(info):
    """Pick the column most likely to be the TRUE household consumption/
    expenditure grand total, distinguishing it from same-file subcategory
    totals (TOTFOOD, TOTHLTH, TOTALCH, etc.) and from poverty-line threshold
    columns (constants, not per-household totals) that a plain keyword
    match can't tell apart."""
    cols_labels = list(zip(info["columns"], info["labels"]))

    def is_poverty_threshold(col_l, label_l):
        return any(h in label_l or h in col_l for h in POVERTY_THRESHOLD_HINTS)

    def is_subcategory(label_l):
        return any(w in label_l for w in SUBCATEGORY_EXCLUDE_WORDS)

    # Priority 1: grand-total label hints, checked against EVERY column --
    # not pre-filtered by the generic keyword list, since a real grand-total
    # label (e.g. "Total annual nominal household expenditures") often
    # doesn't contain any single short keyword phrase as an exact substring.
    for col, label in cols_labels:
        label_l, col_l = (label or "").lower(), col.lower()
        if is_poverty_threshold(col_l, label_l):
            continue
        if any(hint in label_l for hint in GRAND_TOTAL_LABEL_HINTS):
            return col

    # Priority 2: column name matches a known grand-total naming convention
    for col, label in cols_labels:
        col_l = col.lower()
        if is_poverty_threshold(col_l, ""):
            continue
        if any(hint in col_l for hint in GRAND_TOTAL_NAME_HINTS):
            return col

    # Priority 3: label contains "household" + ("expenditure" or "income")
    # + "total" as separate words (not necessarily adjacent), and isn't a
    # subcategory or poverty-threshold column.
    for col, label in cols_labels:
        label_l, col_l = (label or "").lower(), col.lower()
        if is_poverty_threshold(col_l, label_l) or is_subcategory(label_l):
            continue
        if "household" in label_l and ("expenditure" in label_l or "income" in label_l) and "total" in label_l:
            return col

    # Priority 4: fall back to the generic keyword list, excluding
    # subcategory and poverty-threshold columns
    for col, label in cols_labels:
        label_l, col_l = (label or "").lower(), col.lower()
        text = f"{col_l} {label_l}"
        if is_poverty_threshold(col_l, label_l) or is_subcategory(label_l):
            continue
        if any(kw in text for kw in CATEGORY_KEYWORDS["consumption_income"]):
            return col

    # Last resort: any keyword match at all, even a subcategory one --
    # better than returning nothing.
    for col, label in cols_labels:
        label_l, col_l = (label or "").lower(), col.lower()
        text = f"{col_l} {label_l}"
        if any(kw in text for kw in CATEGORY_KEYWORDS["consumption_income"]):
            return col
    return None


def pick_age_column(info):
    """Find a column that represents a person's age (household head or
    member). Uses whole-word matching on the label to avoid false positives
    like matching 'age' inside 'marriage' or 'average'."""
    candidates = []
    for col, label in zip(info["columns"], info["labels"]):
        label_l = (label or "").lower()
        col_l = col.lower()
        # whole-word match only, not substring -- "marriage" contains "age"
        # but is not an age column
        label_words = re.findall(r"[a-z]+", label_l)
        if col_l == "age" or "age" in label_words:
            candidates.append(col)
    if not candidates:
        return None
    # prefer the exact column name "age" if present
    for c in candidates:
        if c.lower() == "age":
            return c
    return candidates[0]


def pick_single_column(info, category, prefer_substrings=()):
    """Pick one best column for a category, preferring names containing
    any of prefer_substrings (e.g. 'total', 'percap') when available."""
    _, cols = score_file(info, category)
    if not cols:
        return None
    for sub in prefer_substrings:
        for c in cols:
            if sub in c.lower():
                return c
    return cols[0]


def cmd_auto(args):
    root = args.root
    print(f"Discovering .dta files under: {root}\n(This reads metadata only -- fast, even for large files.)\n")
    inventory = discover_files(root, ext="dta")
    print(f"Found {len(inventory)} .dta files.\n")

    if not inventory:
        print("No .dta files found. If your files are .sav only, convert or")
        print("point --root at the folder containing the .dta versions.")
        return

    print("Selecting household file...")
    hh_path, _ = pick_file_for_category(inventory, "household_file")
    print("Selecting health file...")
    health_path, _ = pick_file_for_category(inventory, "health_file", exclude=(hh_path,))
    print("Selecting consumption file...")
    cons_path, _ = pick_file_for_category(inventory, "consumption_file", exclude=(hh_path, health_path))

    if not (hh_path and health_path and cons_path):
        print("\nCould not confidently identify all three required files.")
        print(f"  household_file: {hh_path}")
        print(f"  health_file:    {health_path}")
        print(f"  consumption_file: {cons_path}")
        print("\nRun `scan` instead and pick files manually -- this dataset's")
        print("naming conventions may not match the built-in heuristics.")
        return

    hh_id = pick_id_column(inventory[hh_path])
    weight_col = pick_single_column(inventory[hh_path], "weight")
    region_col = pick_single_column(inventory[hh_path], "region")
    age_col = pick_age_column(inventory[hh_path])
    sex_col = pick_sex_column(inventory[hh_path])
    literacy_col = pick_literacy_column(inventory[hh_path])
    education_col = pick_education_column(inventory[hh_path])
    rel_col = pick_relationship_column(inventory[hh_path])
    head_code, head_code_confirmed = (None, False)
    if rel_col:
        head_code, head_code_confirmed = find_head_code(inventory[hh_path], rel_col)

    health_info = inventory[health_path]
    is_official_exphlth_file = bool(re.search(r"exphlth", os.path.basename(health_path), re.IGNORECASE))

    if is_official_exphlth_file:
        # This is GSS's own pre-aggregated health-expenditure file. It often
        # contains BOTH the raw cost components (e.g. HOSP, OPD, DRUG) AND a
        # pre-computed TOTAL column that already sums them (e.g. TOTHLTH).
        # Summing everything naively double-counts: total + its own parts.
        # So: if a column name looks like a total/aggregate, trust GSS's own
        # total and use ONLY that column. Otherwise, sum all non-id numeric
        # columns as components.
        numeric_types = {"double", "float", "int8", "int16", "int32", "long"}
        all_numeric_cols = [
            c for c in health_info["columns"]
            if c != hh_id and (
                not health_info["types"] or health_info["types"].get(c, "").lower() in numeric_types
                or health_info["types"].get(c, "") == ""
            )
        ]
        total_col_candidates = [c for c in all_numeric_cols if re.search(r"^tot|total", c, re.IGNORECASE)]

        if total_col_candidates:
            oop_cols = [total_col_candidates[0]]
            print(f"\n(Health file matched the official EXPHLTH aggregate pattern.")
            print(f" Found a pre-computed total column: '{oop_cols[0]}'.")
            print(f" Using ONLY this column, since the other columns")
            print(f" ({[c for c in all_numeric_cols if c not in oop_cols]}) are likely")
            print(f" its own sub-components -- summing everything would double-count.)")
        else:
            oop_cols = all_numeric_cols
            print(f"\n(Health file matched the official EXPHLTH aggregate pattern --")
            print(f" no obvious pre-computed total column found, so using ALL")
            print(f" {len(oop_cols)} non-id columns as OOP components: {oop_cols})")
    else:
        _, oop_cols = score_file(health_info, "health_spending")
        if hh_id in oop_cols:
            oop_cols.remove(hh_id)
        if not oop_cols:
            # health aggregate files are often narrow (just id + one value column)
            oop_cols = [c for c in health_info["columns"] if c != hh_id][:1]

    hh_id_in_health = hh_id if hh_id in health_info["columns"] else pick_id_column(health_info)

    cons_info = inventory[cons_path]
    cons_col = pick_total_consumption_column(cons_info)
    hh_id_in_cons = hh_id if hh_id in cons_info["columns"] else pick_id_column(cons_info)
    budget_cols_found = {c: label for c, label in BUDGET_CATEGORY_COLUMNS.items() if c in cons_info["columns"]}

    print("\n" + "=" * 70)
    print("AUTO-SELECTED CONFIGURATION -- review before trusting the results:")
    print("=" * 70)
    print(f"household_file:    {hh_path}")
    print(f"  hh_id:    {hh_id}")
    print(f"  weight:   {weight_col}")
    print(f"  region:   {region_col}")
    print(f"  age:      {age_col if age_col else '(not found -- age breakdown will be skipped)'}")
    print(f"  sex:      {sex_col if sex_col else '(not found -- gender breakdown will be skipped)'}")
    print(f"  literacy: {literacy_col if literacy_col else '(not found)'}")
    print(f"  education: {education_col if education_col else '(not found)'}")
    if rel_col:
        confirmed_str = "confirmed via file's own value labels" if head_code_confirmed else "UNCONFIRMED -- assuming conventional GLSS code 1, verify against codebook"
        print(f"  relationship-to-head: {rel_col} (head = code {head_code}, {confirmed_str})")
    else:
        print(f"  relationship-to-head: (not found -- age/sex will still use first-row-per-household, not confirmed head)")
    print(f"health_file:       {health_path}")
    print(f"  hh_id (in this file): {hh_id_in_health}")
    print(f"  oop_components:   {oop_cols}")
    print(f"consumption_file:  {cons_path}")
    print(f"  hh_id (in this file): {hh_id_in_cons}")
    print(f"  total_consumption: {cons_col}")
    if budget_cols_found:
        print(f"  budget categories found: {list(budget_cols_found.keys())}")
    print("=" * 70)

    missing = [name for name, val in [
        ("hh_id", hh_id), ("weight", weight_col), ("region", region_col),
        ("oop_components", oop_cols), ("total_consumption", cons_col),
    ] if not val]
    if missing or hh_id_in_health != hh_id or hh_id_in_cons != hh_id:
        print("\nWARNING: some fields could not be auto-matched, or the")
        print("household ID column has a different name across files (common")
        print("in GLSS -- e.g. 'hid' vs 'hhid'). The config below still gets")
        print("written so you can hand-fix just the mismatched fields instead")
        print("of starting from a blank template.")

    cfg = {
        "household_file": hh_path,
        "health_file": health_path,
        "consumption_file": cons_path,
        "columns": {
            "hh_id": hh_id,
            "weight": weight_col,
            "region": region_col,
            "age": age_col,
            "sex": sex_col,
            "literacy": literacy_col,
            "education": education_col,
            "relationship_to_head": rel_col,
            "head_code": head_code,
            "budget_category_columns": budget_cols_found,
            "oop_components": oop_cols,
            "total_consumption": cons_col,
        },
        "che_thresholds": [0.10, 0.20, 0.25],
    }
    config_path = os.path.join(root, "config_auto.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nConfig written to: {config_path}")
    print("(Edit this file directly if any field above looks wrong, then")
    print(" re-run with: python ghana_survey_toolkit.py analyze --config \"" + config_path + "\")")

    if missing:
        print("\nStopping here -- fix the missing fields above before analysis can run.")
        return

    print("\nProceeding to run the analysis now...\n")
    try:
        run_analysis(cfg, root)
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        print("This usually means one of the auto-picked columns isn't quite")
        print(f"right. Open {config_path}, fix the field the error points to,")
        print("and re-run with `analyze --config` instead of starting over.")


def cmd_inspect(args):
    path = args.file
    print(f"Inspecting: {path}\n")
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".dta":
            df_full, meta = pyreadstat.read_dta(path)
        elif ext == ".sav":
            df_full, meta = pyreadstat.read_sav(path)
        else:
            print("Unsupported file type -- expected .dta or .sav")
            return
    except Exception as e:
        print(f"Could not read file: {e}")
        return

    col_names = meta.column_names
    col_labels = meta.column_labels or [""] * len(col_names)
    value_labels = getattr(meta, "variable_value_labels", {}) or {}
    print(f"{len(df_full):,} rows, {len(col_names)} columns\n")
    print(f"{'column':<20} {'label':<55} {'sample values'}")
    print("-" * 100)
    for c, lbl in zip(col_names, col_labels):
        sample = df_full[c].dropna().unique()[:4]
        sample_str = ", ".join(str(s) for s in sample)
        print(f"{c:<20} {(lbl or '')[:55]:<55} {sample_str}")

    if value_labels:
        print("\nVALUE LABELS FOUND (this is the authoritative answer for what codes")
        print("like 1/2 actually mean -- no need to consult the codebook PDF for these):")
        for col, mapping in value_labels.items():
            if col in df_full.columns:
                print(f"  {col}: {mapping}")
    else:
        print("\nNo value labels found in this file's metadata. If a column like 'sex'")
        print("uses numeric codes, you'll need the codebook PDF to confirm what each")
        print("code means -- this file doesn't carry that information internally.")

    if args.id_col and args.id_col in df_full.columns:
        n_rows = len(df_full)
        n_unique = df_full[args.id_col].nunique()
        print(f"\n--id-col '{args.id_col}': {n_rows:,} rows, {n_unique:,} unique values", end="")
        if n_rows != n_unique:
            print(f"  -- NOT unique per row (this file is not one-row-per-{args.id_col})")
        else:
            print("  -- unique per row, good")


def interactive_mode():
    """Fallback when the script is launched with no CLI arguments (e.g. via
    VS Code's Run/Debug button, which doesn't pass args by default). Prompts
    for what to do instead of crashing with an argparse error."""
    print("=" * 70)
    print("Ghana Survey Toolkit -- no command-line arguments were given.")
    print("(This happens when you click Run/Debug in an editor instead of")
    print(" running from a terminal with arguments.)")
    print("=" * 70)
    print()
    print("1) auto    -- (RECOMMENDED) auto-discover files/columns and run analysis")
    print("2) scan    -- scan a folder of .dta/.sav files for relevant columns")
    print("3) analyze -- run the analysis using a config.json")
    print("4) quit")
    choice = input("\nChoose 1, 2, 3, or 4: ").strip()

    if choice == "1":
        root = input("Path to the folder (e.g. C:\\Users\\steve\\Downloads\\health reaserch): ").strip().strip('"')
        if not root:
            print("No path given, exiting.")
            return
        args = argparse.Namespace(root=root)
        cmd_auto(args)
    elif choice == "2":
        root = input("Path to the folder to scan (e.g. C:\\Users\\steve\\Downloads\\health reaserch): ").strip().strip('"')
        if not root:
            print("No path given, exiting.")
            return
        args = argparse.Namespace(root=root)
        cmd_scan(args)
    elif choice == "3":
        config = input("Path to config.json [default: config.json]: ").strip().strip('"')
        if not config:
            config = "config.json"
        args = argparse.Namespace(config=config)
        cmd_analyze(args)
    else:
        print("Exiting.")


def main():
    parser = argparse.ArgumentParser(description="Ghana GLSS/AHIES survey toolkit")
    sub = parser.add_subparsers(dest="command", required=False)

    p_inspect = sub.add_parser("inspect", help="Show ALL columns/labels/sample values for one file")
    p_inspect.add_argument("--file", required=True, help="Path to a single .dta/.sav file")
    p_inspect.add_argument("--id-col", default=None, help="Optional: check this column for duplicate rows (e.g. hid)")
    p_inspect.set_defaults(func=cmd_inspect)

    p_auto = sub.add_parser("auto", help="Auto-discover files/columns and run the analysis in one step")
    p_auto.add_argument("--root", required=True, help="Root folder containing the .dta files")
    p_auto.set_defaults(func=cmd_auto)

    p_scan = sub.add_parser("scan", help="Scan all files under a folder for relevant columns")
    p_scan.add_argument("--root", required=True, help="Root folder to scan (e.g. the 'health reaserch' folder)")
    p_scan.set_defaults(func=cmd_scan)

    p_analyze = sub.add_parser("analyze", help="Run the analysis using a filled-in config.json")
    p_analyze.add_argument("--config", default="config.json", help="Path to config.json")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()

    if args.command is None:
        interactive_mode()
        return

    args.func(args)


if __name__ == "__main__":
    main()
