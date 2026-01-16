# Nonlinear Difference-in-Differences with Neural Networks

This repository implements a nonlinear difference-in-differences estimator using neural networks for causal inference. The methodology combines cross-fitted neural networks with debiased influence functions to estimate treatment effects in multi-cohort settings.

## Overview

The estimator handles ratio-based treatment effects of the form:

```
τ = E[Y_post × μ₀_pre / (μ₁_pre × μ₀_post)] - 1
```

where:
- `μ₀`, `μ₁` are conditional outcome means under control and treatment
- Subscripts indicate pre/post treatment periods

## Features

- **Cross-fitted neural networks** for nuisance parameter estimation
- **Influence functions** with Neyman-orthogonal corrections for √n-consistency
- **Multi-cohort event studies** with proper aggregation weights
- **Clustered inference** using sandwich standard errors
- **Modular, production-ready code** with comprehensive documentation

## Repository Structure

```
.
├── code/
│   └── r/
│       ├── utilities.R             # Core utilities and cross-validation
│       ├── neural_network.R        # MLP architectures and loss functions
│       ├── training.R              # Model training and cross-fitting
│       ├── influence_functions.R   # Debiased influence function computation
│       ├── event_study.R           # Multi-cohort aggregation and visualization
│       └── analysis.R              # Main execution script
├── data/
│   ├── raw/                        # Original data files
│   ├── intermediate/               # Processed intermediate files
│   └── final/                      # Analysis-ready datasets
├── CLAUDE.md                       # Detailed technical documentation
└── README.md                       # This file
```

## Getting Started

### Prerequisites

Install required R packages:

```r
install.packages(c("torch", "data.table", "dplyr", "stringr",
                   "ggplot2", "scales", "sandwich", "fixest"))
```

### Running the Analysis

```r
setwd("~/path/to/Paper-NonlinearDiD/code/r")
source("analysis.R")
results = run_analysis()
```

### Configuration

All parameters are controlled through two configuration lists in `analysis.R`:

- **CONFIG**: Data paths, cohorts, loss type, inference settings
- **HYPERPARAMS**: Neural network architecture, training parameters

See `CLAUDE.md` for detailed configuration options.

## Data Format

Each cohort CSV file must contain:

- **Outcome columns**: Prefixed with `Y_` (e.g., `Y_lead_pre_1`, `Y_lag_post_3`)
- **Treatment column**: One column prefixed with `D_` (e.g., `D_treat`)
- **Covariate columns**: Prefixed with `X_` (e.g., `X_pop_1990`, `X_sfr`)
- **Cluster variable**: `county_fips` for clustered standard errors

## Methodology

The estimation proceeds in five stages:

1. **Cohort-specific models**: Fit cross-fitted neural networks for each cohort
2. **Influence functions**: Compute debiased IFs with Neyman orthogonality
3. **Multi-cohort aggregation**: Combine estimates with proper weighting
4. **Event study**: Construct estimates with clustered standard errors
5. **Visualization**: Generate publication-ready plots

See `CLAUDE.md` for architectural details and theoretical background.

## Key Functions

- `run_joint_structural()` - Fit cross-fitted neural networks for one cohort
- `compute_influence_function()` - Core IF computation with debiasing
- `aggregate_cohort_ifs()` - Aggregate across cohorts with proper weighting
- `build_event_study()` - Construct event study table with clustered SEs
- `plot_event_study()` - Generate publication-ready plots

## Citation

If you use this code in your research, please cite:

```
@software{nonlinear_did_2026,
  title = {Semi-parametric Non-linear Difference-in-Differences for Policy Evaluation: New Evidence on the Entrepreneurial Impacts of State Business Tax Credits},
  author = {Spencer Sween},
  year = {2026},
  url = {https://github.com/spencersween/Paper-NonlinearDiD}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact spencersween@ucsb.edu.
