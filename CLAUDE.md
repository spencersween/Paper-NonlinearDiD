# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a nonlinear difference-in-differences (DiD) estimator using neural networks for causal inference. The methodology combines:

- **Cross-fitted neural networks** to estimate conditional outcome means E[Y|X,D] and propensity scores P(D=1|X)
- **Influence functions** for debiased estimation with Neyman-orthogonal corrections
- **Multi-cohort event studies** with proper aggregation weights and clustered standard errors

The estimator handles ratio-based treatment effects of the form:
```
tau = E[Y_post * mu_0_pre / (mu_1_pre * mu_0_post)] - 1
```

## Code Structure

The R code is organized into modular components in `code/r/`:

1. **utilities.R** - Core utilities, constants, gradient clipping, cross-validation fold construction
2. **neural_network.R** - MLP architectures, bounded transformations, joint models for outcomes and propensity, loss functions
3. **training.R** - Model training with early stopping, cross-fitting driver, data preprocessing
4. **influence_functions.R** - Debiased influence function computation with Neyman orthogonality
5. **event_study.R** - Multi-cohort aggregation, event study construction, visualization
6. **analysis.R** - Main execution script that orchestrates the full pipeline

## Running the Analysis

### Basic Execution

```r
# From the code/r directory
setwd("~/Dropbox/Paper -- Nonlinear DiD/code/r")
source("analysis.R")
results = run_analysis()
```

### Configuration

All analysis parameters are controlled through two configuration lists in `analysis.R`:

**CONFIG** - Data and inference settings:
- `data_dir`: Path to cohort CSV files
- `cohorts`: Vector of cohort years to analyze
- `loss_type`: "poisson" or "mse"
- `bounds`: Prediction bounds for neural networks
- `alpha`, `bonferroni`: Inference parameters

**HYPERPARAMS** - Neural network training:
- `k_folds`: Number of cross-fitting folds (typically 3-5)
- `val_frac`: Validation split within training folds
- `hidden_sizes`: Vector of hidden layer dimensions (empty = linear model)
- `l2_lambda`: L2 (ridge) regularization strength
- `l1_lambda`: L1 (lasso) regularization strength
- `use_explicit_l2`: TRUE to add L2 penalty to loss; FALSE to use optimizer weight_decay
- `optimizer`: "adamw" or "lbfgs"
- `max_epochs`, `early_patience`: Training duration controls
- `batch_size`: Mini-batch size (NULL for full batch)

### Data Requirements

Each cohort CSV file must contain:
- **Outcome columns**: Prefixed with `Y_` (e.g., `Y_lead_pre_1`, `Y_lag_post_3`)
- **Treatment column**: Exactly one column prefixed with `D_` (e.g., `D_treat`)
- **Covariate columns**: Prefixed with `X_` (e.g., `X_pop_1990`, `X_sfr`)
- **Cluster variable**: `county_fips` for clustered standard errors

## Key Functions

### Model Estimation
- `run_joint_structural()` - Fit cross-fitted neural networks for one cohort
- `crossfit_train_joint()` - K-fold cross-fitting driver
- `train_one_fold_joint()` - Train model on a single fold

### Influence Functions
- `compute_influence_function()` - Core IF computation with debiasing
- `compute_cohort_ifs()` - Compute all IFs for a cohort

### Event Study
- `aggregate_cohort_ifs()` - Aggregate across cohorts with proper weighting
- `build_event_study()` - Construct event study table with clustered SEs
- `plot_event_study()` - Generate publication-ready plots

## Architectural Notes

### Neural Network Design

The joint model architecture outputs:
- `a(X)`: E[Y|X, D=0] for each outcome (control potential outcome)
- `b(X)`: E[Y|X, D=1] for each outcome (treatment potential outcome)
- `p(X)`: P(D=1|X) via logit (propensity score)

Predictions use: `yhat = a*(1-D) + b*D`

All outcome predictions are bounded via smooth sigmoid transformations to ensure numerical stability.

### Cross-Fitting

Group-aware K-fold cross-fitting ensures:
1. All observations from the same cluster (county) are in the same fold
2. Out-of-fold predictions avoid overfitting bias
3. Each fold uses separate train/validation splits for early stopping

### Influence Function Theory

The influence function achieves √n-consistency even when neural network nuisance functions converge at slower rates. The debiasing correction uses:

```
psi = plugin_estimate + debiasing_term + propensity_centering
```

where the debiasing term involves inverse propensity weighted residuals to achieve Neyman orthogonality.

## Common Modifications

### Changing Network Architecture

In `analysis.R`, modify `HYPERPARAMS$hidden_sizes`:
```r
hidden_sizes = c(64, 64)     # Two hidden layers with 64 units each
hidden_sizes = c()           # Linear model (no hidden layers)
```

### Regularization Options

The code supports three approaches to regularization:

1. **Explicit L2 (Ridge) penalty** - Added directly to the loss function:
```r
l2_lambda = 1000          # Regularization strength
use_explicit_l2 = TRUE    # Add L2 to loss
```

2. **Optimizer weight decay** - L2 regularization via AdamW optimizer:
```r
l2_lambda = 0.01          # Weight decay parameter
use_explicit_l2 = FALSE   # Use optimizer weight_decay instead
optimizer = "adamw"       # Required for this approach
```

3. **L1 (Lasso) penalty** - Always added directly to loss:
```r
l1_lambda = 100           # L1 regularization strength
```

**For linear models** (`hidden_sizes = c()`):
- Use explicit L2 regularization for ridge regression
- Set `use_explicit_l2 = TRUE` and `l2_lambda > 0`
- L-BFGS optimizer always uses explicit L2 regardless of `use_explicit_l2`

**For neural networks**:
- Both explicit L2 and optimizer weight decay are available
- Explicit L2 applies to all parameters including biases
- Optimizer weight decay (AdamW) applies differently in optimization

### Adding Covariates

Edit the column selection in `load_cohort_data()`:
```r
dplyr::select(
  county_fips,
  starts_with("Y"),
  starts_with("D"),
  X_pop_1990, X_sfr, X_eqi,
  X_new_covariate  # Add new covariates here
)
```

### Adjusting Loss Functions

The loss type is specified in `CONFIG$loss_type`:
- `"poisson"`: For count outcomes (uses Poisson NLL)
- `"mse"`: For continuous outcomes (uses MSE)

### Modifying Bounds

Prediction bounds prevent extreme values:
```r
bounds = c(lower = 1e-5, upper = 1e+5)  # Default
bounds = c(lower = 0.01, upper = 1000)  # Tighter bounds
```

## Coding Conventions

- **Assignment operator**: Use `=` instead of `<-` throughout
- **Function naming**: snake_case for all functions
- **Comments**: Professional inline comments explaining non-obvious logic
- **Constants**: UPPERCASE for global configuration lists

## Dependencies

Required R packages:
- `torch` - Neural network training
- `data.table` - Fast data loading
- `dplyr`, `stringr` - Data manipulation
- `ggplot2`, `scales` - Visualization
- `sandwich`, `fixest` - Clustered inference

Install with:
```r
install.packages(c("torch", "data.table", "dplyr", "stringr",
                   "ggplot2", "scales", "sandwich", "fixest"))
```
