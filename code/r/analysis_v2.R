################################################################################
#
# Nonlinear Difference-in-Differences: Multi-Cohort Event Study Analysis
#
# Main execution script that orchestrates:
#   1. Cohort-specific structural model fitting
#   2. Influence function computation
#   3. Multi-cohort aggregation with proper weighting
#   4. Event study estimation with clustered inference
#
# Usage:
#   From the project root directory:
#     source("code/r/analysis_v2.R")
#     results = run_analysis()
#
#   Or from within code/r/:
#     source("analysis_v2.R")
#     results = run_analysis()
#
################################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(stringr)
  library(ggplot2)
})

################################################################################
# Directory Setup
################################################################################

# Detect project root directory
# This works whether script is run from project root or from code/r/
get_project_root = function() {
  # Get the directory containing this script
  script_dir = tryCatch({
    # Works when sourced
    dirname(sys.frame(1)$ofile)
  }, error = function(e) {
    # Fallback to current working directory
    getwd()
  })

  # If we're in code/r/, go up two levels to project root
  if (basename(script_dir) == "r" && basename(dirname(script_dir)) == "code") {
    return(normalizePath(file.path(script_dir, "..", "..")))
  }

  # If we're in code/, go up one level
  if (basename(script_dir) == "code") {
    return(normalizePath(file.path(script_dir, "..")))
  }

  # Otherwise assume we're at project root
  return(normalizePath(script_dir))
}

# Set up all paths relative to project root
PROJECT_ROOT = get_project_root()
CODE_DIR = file.path(PROJECT_ROOT, "code", "r")
DATA_DIR = file.path(PROJECT_ROOT, "data", "final", "csv")

# Print paths for verification
cat("\n")
cat("Directory Setup:\n")
cat("  Project root:", PROJECT_ROOT, "\n")
cat("  Code dir:    ", CODE_DIR, "\n")
cat("  Data dir:    ", DATA_DIR, "\n")
cat("\n")

# Source all component modules using absolute paths
source(file.path(CODE_DIR, "utilities.R"))
source(file.path(CODE_DIR, "neural_network.R"))
source(file.path(CODE_DIR, "training.R"))
source(file.path(CODE_DIR, "influence_functions.R"))
source(file.path(CODE_DIR, "event_study.R"))

################################################################################
# Configuration
################################################################################

CONFIG = list(
  # Data - now uses the detected DATA_DIR
  data_dir = DATA_DIR,
  cohorts = c(1991, 1994, 1996, 1997, 1998, 2000, 2001, 2002, 2003, 2004, 2006, 2008, 2010),

  # Model settings
  loss_type = "poisson",
  propensity_weight = 1.0,
  bounds = c(lower = 1e-5, upper = 1e+5),

  # Influence function computation
  clamp_floor = 1e-5,
  clamp_ratio = Inf,

  # Inference
  alpha = 0.05,
  bonferroni = TRUE,

  # Output
  y_limits = NULL
)

HYPERPARAMS = list(
  seed = 42,
  k_folds = 3,
  val_frac = 0.20,
  hidden_sizes = c(),
  activation = "relu",
  dropout = 0.00,
  weight_decay_lambda = 0,
  optimizer = "adamw",
  optimizer_args = list(lr = 0.10),
  max_epochs = 2000L,
  batch_size = 2^16,
  use_early_stopping = TRUE,
  keep_best_model = TRUE,
  early_patience = 100,
  early_min_delta = 0.0,
  grad_clip_norm = 1.0,
  lr_step = 1000,
  lr_gamma = 0.1,
  device = "cpu",
  verbose = TRUE,
  make_plots = TRUE
)

################################################################################
# Data Loading
################################################################################

# Load and prepare cohort data
#
# @param cohort Cohort year
# @param data_dir Directory containing CSV files (defaults to CONFIG$data_dir)
# @return Data frame with selected columns
load_cohort_data = function(cohort, data_dir = CONFIG$data_dir) {
  fpath = file.path(data_dir, sprintf("cohort_%d.csv", cohort))

  # Check if file exists
  if (!file.exists(fpath)) {
    stop(sprintf("Data file not found: %s", fpath))
  }

  data.table::fread(fpath) |>
    as.data.frame() |>
    dplyr::select(
      county_fips,
      starts_with("Y"),
      starts_with("D"),
      X_pop_1990
    )
}

################################################################################
# Main Analysis Pipeline
################################################################################

# Run complete nonlinear DiD analysis
#
# Orchestrates the full analysis pipeline:
#   - Stage 1: Fit structural models for each cohort
#   - Stage 2: Aggregate influence functions across cohorts
#   - Stage 3: Build event study with clustered inference
#   - Stage 4: Compute average pre/post treatment effects
#   - Stage 5: Generate publication-ready plots
#
# @return List with complete analysis results
run_analysis = function() {

  cat("\n", strrep("=", 70), "\n")
  cat(" Nonlinear DiD Event Study Analysis\n")
  cat(strrep("=", 70), "\n\n")

  cat("Data directory:", CONFIG$data_dir, "\n")
  cat("Number of cohorts:", length(CONFIG$cohorts), "\n\n")

  # ========================================================================
  # Stage 1: Fit models and compute IFs for each cohort
  # ========================================================================
  cat("Stage 1: Fitting cohort-specific models\n")
  cat(strrep("-", 50), "\n")

  if_list = list()
  model_list = list()

  for (cohort in CONFIG$cohorts) {
    cat(sprintf("\n[Cohort %d] ", cohort))

    # Load data
    df = load_cohort_data(cohort)
    cat(sprintf("n = %d ... ", nrow(df)))

    # Fit structural model
    results = run_joint_structural(
      df = df,
      hyperparams = HYPERPARAMS,
      group_vec = df$county_fips,
      loss_type = CONFIG$loss_type,
      propensity_weight = CONFIG$propensity_weight,
      lower_bound = CONFIG$bounds["lower"],
      upper_bound = CONFIG$bounds["upper"]
    )

    # Compute influence functions
    if_df = compute_cohort_ifs(df, results, clamp_ratio = CONFIG$clamp_ratio)
    if_df$county_fips = df$county_fips
    if_df$cohort = cohort

    if_list[[as.character(cohort)]] = if_df
    model_list[[as.character(cohort)]] = results

    cat(sprintf("done (%d estimands)", ncol(if_df) - 2))
  }

  # ========================================================================
  # Stage 2: Aggregate across cohorts
  # ========================================================================
  cat("\n\n")
  cat(strrep("-", 50), "\n")
  cat("Stage 2: Aggregating across cohorts\n")
  cat(strrep("-", 50), "\n")

  agg = aggregate_cohort_ifs(if_list)

  cat(sprintf(
    "\nTotal observations: %s\n",
    format(agg$N_total, big.mark = ",")
  ))
  cat(sprintf("Unique estimands: %d\n", nrow(agg$estimates)))

  # ========================================================================
  # Stage 3: Build event study
  # ========================================================================
  cat("\n")
  cat(strrep("-", 50), "\n")
  cat("Stage 3: Event study estimation\n")
  cat(strrep("-", 50), "\n\n")

  es_df = build_event_study(agg, alpha = CONFIG$alpha, bonferroni = CONFIG$bonferroni)

  print(
    es_df |>
      select(event_time, type, tau_k, se_clustered, t_stat, pval, ci_lo, ci_hi, n_cohorts, N_k) |>
      mutate(across(where(is.numeric), ~round(., 3)))
  )

  # ========================================================================
  # Stage 4: Average effects
  # ========================================================================
  cat("\n")
  cat(strrep("-", 50), "\n")
  cat("Stage 4: Average treatment effects\n")
  cat(strrep("-", 50), "\n")

  avg_effects = compute_average_effects(es_df, agg)

  if (!is.null(avg_effects$pre)) {
    cat(sprintf(
      "\nPre-treatment average (k = -%d to -1): %.4f\n",
      avg_effects$pre$n_periods,
      avg_effects$pre$mean
    ))
    print(avg_effects$pre$fit)
  }

  if (!is.null(avg_effects$post)) {
    cat(sprintf(
      "\nPost-treatment average (k = 1 to %d): %.4f\n",
      avg_effects$post$n_periods,
      avg_effects$post$mean
    ))
    print(avg_effects$post$fit)
  }

  # ========================================================================
  # Stage 5: Plot
  # ========================================================================
  cat("\n")
  cat(strrep("-", 50), "\n")
  cat("Stage 5: Generating plot\n")
  cat(strrep("-", 50), "\n")

  p = plot_event_study(
    es_df,
    title = "Nonlinear DiD Event Study",
    alpha = CONFIG$alpha,
    bonferroni = CONFIG$bonferroni,
    n_cohorts = length(CONFIG$cohorts),
    y_limits = CONFIG$y_limits
  )
  print(p)

  # ========================================================================
  # Return results
  # ========================================================================
  invisible(list(
    if_list = if_list,
    model_list = model_list,
    aggregation = agg,
    event_study = es_df,
    average_effects = avg_effects,
    plot = p,
    paths = list(
      project_root = PROJECT_ROOT,
      code_dir = CODE_DIR,
      data_dir = DATA_DIR
    )
  ))
}

################################################################################
# Execute Analysis
################################################################################

# Uncomment to run automatically when sourced:
# results = run_analysis()

# Or run manually in your R session:
# source("code/r/analysis_v2.R")
# results = run_analysis()
