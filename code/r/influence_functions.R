################################################################################
#
# Influence Function Computation
#
# Implements debiased influence functions for ratio-based DiD estimands.
# Computes Neyman-orthogonal corrections for consistent treatment effect
# estimation with neural network nuisance functions.
#
################################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(stringr)
  library(sandwich)
})

################################################################################
# Constants and Utilities
################################################################################

# Default configuration for IF computation
IF_CONFIG = list(
  clamp_floor = 1e-5
)

# Clamp values to a minimum threshold
# @param x Numeric vector
# @param lo Lower bound (default from IF_CONFIG)
clamp = function(x, lo = IF_CONFIG$clamp_floor) pmax(lo, x)

# Extract numeric suffix from column names
# @param cols Character vector of column names
# @param pattern Regex pattern to remove
extract_k = function(cols, pattern) {
  as.numeric(gsub(pattern, "", cols))
}

################################################################################
# Core Influence Function
################################################################################

# Compute influence function for ratio-based DiD estimand
#
# Computes the debiased influence function for:
#   tau = E[Y_post * mu_0_pre / (mu_1_pre * mu_0_post)] - 1
#
# Uses Neyman orthogonality to achieve √n-consistent estimation with
# neural network nuisance functions converging at slower rates.
#
# @param outcome_pre Pre-treatment outcome vector
# @param outcome_post Post-treatment outcome vector
# @param yhat_0_pre Predicted E[Y_pre | D=0, X]
# @param yhat_0_post Predicted E[Y_post | D=0, X]
# @param yhat_1_pre Predicted E[Y_pre | D=1, X]
# @param yhat_1_post Predicted E[Y_post | D=1, X]
# @param treat Treatment indicator
# @param pscore Propensity score P(D=1 | X)
# @param clamp_ratio Maximum allowed ratio (Inf for no capping)
# @return Influence function vector (length = n)
compute_influence_function = function(outcome_pre, outcome_post,
                                       yhat_0_pre, yhat_0_post,
                                       yhat_1_pre, yhat_1_post,
                                       treat, pscore,
                                       clamp_ratio = Inf) {
  n = length(treat)
  q = mean(treat)

  # Inverse propensity weights
  i_1m_ps = 1 / (1 - pscore + IF_CONFIG$clamp_floor)
  i_ps = 1 / (pscore + IF_CONFIG$clamp_floor)

  # Clamp predictions to avoid division issues
  yhat_0_pre  = clamp(yhat_0_pre)
  yhat_0_post = clamp(yhat_0_post)
  yhat_1_pre  = clamp(yhat_1_pre)
  yhat_1_post = clamp(yhat_1_post)

  # Denominators for ratio components
  denom1 = yhat_1_pre * yhat_0_post
  denom2 = yhat_1_pre * clamp(yhat_0_post^2)
  denom3 = clamp(yhat_1_pre^2) * yhat_0_post

  # Numerators
  numer1 = outcome_post * yhat_0_pre
  numer3 = yhat_1_post * yhat_0_pre

  # Plugin estimator (with optional capping)
  ratio = numer1 / denom1 - 1
  plugin = (treat / q) * sign(ratio) * pmin(abs(ratio), clamp_ratio)

  # Debiasing correction terms (Neyman orthogonality)
  EdH00 =  (pscore / q) * (yhat_1_post / denom1)
  EdH01 = -(pscore / q) * (numer3 / denom2)
  EdH10 = -(pscore / q) * (numer3 / denom3)

  # Residuals
  ell00 = (treat == 0) * (outcome_pre  - yhat_0_pre)
  ell01 = (treat == 0) * (outcome_post - yhat_0_post)
  ell10 = (treat == 1) * (outcome_pre  - yhat_1_pre)

  # Debiasing term
  debias = i_1m_ps * (EdH00 * ell00 + EdH01 * ell01) + i_ps * (EdH10 * ell10)

  # Influence function with propensity score centering
  psi_1 = plugin + debias
  psi_2 = (-1 / q) * mean(psi_1) * (treat - q)

  psi_1 + psi_2
}

################################################################################
# Clustered Standard Errors
################################################################################

# Compute clustered standard error for a vector of influence functions
#
# Uses sandwich package to compute robust variance with arbitrary clustering.
#
# @param psi Influence function vector
# @param cluster Cluster identifier vector
# @return Standard error (scalar)
clustered_se = function(psi, cluster) {
  valid = !is.na(psi) & !is.na(cluster)
  if (sum(valid) < 10) return(NA_real_)
  sqrt(sandwich::vcovCL(lm(psi[valid] ~ 1), cluster = cluster[valid])[1, 1])
}

################################################################################
# Cohort-Level IF Computation
################################################################################

# Compute all influence functions for a single cohort
#
# Extracts pre- and post-treatment outcome pairs from the data frame
# and computes influence functions for each event-time period.
#
# @param df Cohort data frame
# @param results Fitted model results (from run_joint_structural)
# @param clamp_ratio Maximum allowed ratio for IF computation
# @return Data frame with IF columns plus county_fips and cohort
compute_cohort_ifs = function(df, results, clamp_ratio = Inf) {
  treat = as.numeric(df$D_treat)
  pscore = as.numeric(results$outputs$p_scores)

  if_list = list()

  # --- Pre-treatment IFs: Y_lead_pre_k vs Y_lead_post_k ---
  lead_pre_k = extract_k(grep("^Y_lead_pre_", names(df), value = TRUE), "Y_lead_pre_")
  lead_post_k = extract_k(grep("^Y_lead_post_", names(df), value = TRUE), "Y_lead_post_")
  matching_k = sort(intersect(lead_pre_k, lead_post_k))

  for (k in matching_k) {
    pre_col = paste0("Y_lead_pre_", k)
    post_col = paste0("Y_lead_post_", k)
    pre_idx = match(pre_col, results$data$y_cols)
    post_idx = match(post_col, results$data$y_cols)

    if (is.na(pre_idx) || is.na(post_idx)) next

    psi = compute_influence_function(
      outcome_pre = as.numeric(df[[pre_col]]),
      outcome_post = as.numeric(df[[post_col]]),
      yhat_0_pre = as.numeric(results$outputs$a[, pre_idx]),
      yhat_0_post = as.numeric(results$outputs$a[, post_idx]),
      yhat_1_pre = as.numeric(results$outputs$b[, pre_idx]),
      yhat_1_post = as.numeric(results$outputs$b[, post_idx]),
      treat = treat,
      pscore = pscore,
      clamp_ratio = clamp_ratio
    )

    if_list[[sprintf("IF_pre_k%d", k)]] = psi
  }

  # --- Post-treatment IFs: Y_lag_pre_base vs Y_lag_post_* ---
  base_col = "Y_lag_pre_base"
  base_idx = match(base_col, results$data$y_cols)

  if (!is.na(base_idx)) {
    outcome_pre_base = as.numeric(df[[base_col]])
    yhat_0_pre_base = clamp(as.numeric(results$outputs$a[, base_idx]))
    yhat_1_pre_base = clamp(as.numeric(results$outputs$b[, base_idx]))

    post_cols = grep("^Y_lag_post_", names(df), value = TRUE)

    for (post_col in post_cols) {
      post_idx = match(post_col, results$data$y_cols)
      if (is.na(post_idx)) next

      suffix = sub("^Y_lag_post_", "", post_col)
      if (is.na(suppressWarnings(as.numeric(suffix)))) next

      psi = compute_influence_function(
        outcome_pre = outcome_pre_base,
        outcome_post = as.numeric(df[[post_col]]),
        yhat_0_pre = yhat_0_pre_base,
        yhat_0_post = as.numeric(results$outputs$a[, post_idx]),
        yhat_1_pre = yhat_1_pre_base,
        yhat_1_post = as.numeric(results$outputs$b[, post_idx]),
        treat = treat,
        pscore = pscore,
        clamp_ratio = clamp_ratio
      )

      if_list[[sprintf("IF_post_k%s", suffix)]] = psi
    }
  }

  as.data.frame(if_list)
}
