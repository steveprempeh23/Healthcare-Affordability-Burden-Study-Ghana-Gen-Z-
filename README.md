# Healthcare Affordability Burden Study: Ghana's Generation Z

A data science study modeling the healthcare affordability burden faced by Generation Z in Ghana, using nationally representative microdata and large-scale Monte Carlo simulation.

> 📄 Manuscript submitted to a Springer journal (under review). A preprint will be linked here once posted.

## Overview

This project quantifies how healthcare costs burden Ghanaian Gen Z households relative to their income and consumption capacity, using the **Ghana Living Standards Survey Round 7 (GLSS7)** microdata. The analysis combines household-level survey data with a large-scale simulation to estimate affordability burden distributions across the population, including a breakdown by urban and rural residence.

### Key components

- **Data source:** GLSS7 / AHIES (Annual Household Income and Expenditure Survey) microdata
- **Simulation:** Monte Carlo simulation across a 10,000-household sample to model affordability burden under uncertainty
- **Disaggregation:** Urban vs. rural analysis to capture geographic disparities in affordability
- **Toolkit:** A custom Python toolkit for extracting and processing GLSS7/AHIES survey data, usable independently for other GLSS7-based research

## Repository contents

- [`healthcare_affordability_gap_genz_ghana_revised_v3.pdf`](./healthcare_affordability_gap_genz_ghana_revised_v3.pdf) — the full manuscript (revised draft, submitted to a Springer journal, currently under review)

**Code, notebooks, and the extraction toolkit are being prepared for release** and will be added to this repository to make the analysis fully reproducible. Planned structure:

\`\`\`
.
├── src/
│   ├── extraction/      # GLSS7/AHIES extraction toolkit
│   ├── simulation/      # Monte Carlo simulation code
│   └── analysis/        # Urban/rural disaggregation and results
├── notebooks/           # Exploratory and results notebooks
├── figures/             # Generated figures used in the manuscript
├── requirements.txt
└── README.md
\`\`\`

## Methodology (summary)

1. Extract and clean relevant household income, expenditure, and health spending variables from GLSS7/AHIES microdata.
2. Construct an affordability burden metric (e.g., out-of-pocket health spending as a share of household capacity to pay).
3. Run a Monte Carlo simulation (n = 10,000 households) to model the distribution of affordability burden under income and expenditure uncertainty.
4. Disaggregate results by urban/rural residence to identify geographic disparities.
5. Summarize findings and generate figures for the manuscript.

Full methodological detail is in the manuscript PDF above.

## Getting started (once code is added)

\`\`\`bash
git clone https://github.com/steveprempeh23/Healthcare-Affordability-Burden-Study-Ghana-Gen-Z-.git
cd Healthcare-Affordability-Burden-Study-Ghana-Gen-Z-
pip install -r requirements.txt
\`\`\`

## Data access

GLSS7/AHIES microdata is distributed by the Ghana Statistical Service (GSS) and typically requires registration/approval to access. This repository does not redistribute raw survey data.

## Citation

If you reference this research, please cite:

\`\`\`
Prempeh, N. O. A. (2026). Healthcare Affordability Burden Study: Ghana's Generation Z.
[Journal details to be added upon publication]
\`\`\`

A preprint of this manuscript will be made available on SSRN — link to be added here once posted.

## Status

- [x] Manuscript drafted and submitted (Springer journal, under review)
- [ ] Code and extraction toolkit uploaded to this repository
- [ ] Preprint posted (SSRN)
- [ ] Repository description/topics added on GitHub

## Author

**Nana Owusu Achiaw Prempeh**
MSc Data Science, Academic City University College
GitHub: [@steveprempeh23](https://github.com/steveprempeh23)

## License

Code in this repository will be released under [MIT License — confirm before adding]. The manuscript text and figures remain under copyright pending journal publication; please do not redistribute the PDF without permission.
