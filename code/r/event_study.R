################################################################################
#
# Event Study Aggregation and Visualization
#
# Multi-cohort event study estimation with:
#   - Proper cohort weighting
#   - Clustered standard errors
#   - Pre/post treatment effect aggregation
#   - Professional publication-ready plots
#
################################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(stringr)
  library(ggplot2)
  library(scales)
  library(fixest)
})

source("code/r/influence_functions.R")

################################################################################
# Multi-Cohort Aggregation
################################################################################

# Aggregate influence functions across cohorts with proper weighting
#
# For each event-time k, computes:
#   tau_k = sum_g P(G=g | G in G_k) * tau_k^(g)
#
# where G_k is the set of cohorts contributing to event-time k.
#
# @param if_list Named list of IF data frames (one per cohort)
# @return List with aggregated estimates and reweighted IFs
aggregate_cohort_ifs = function(if_list) {

  # Identify all IF columns
  all_if_cols = unique(unlist(lapply(if_list, function(x) {
    grep("^IF_", names(x), value = TRUE)
  })))

  # Cohort sizes
  cohort_n = sapply(if_list, nrow)
  N_total = sum(cohort_n)

  # Compute cohort-level means
  cohort_means = do.call(rbind, lapply(names(if_list), function(g) {
    df = if_list[[g]]
    if_cols = intersect(names(df), all_if_cols)

    data.frame(
      cohort = as.integer(g),
      estimand = if_cols,
      n_g = nrow(df),
      tau_g = sapply(if_cols, function(col) mean(df[[col]], na.rm = TRUE)),
      row.names = NULL
    )
  }))

  # Compute weights and aggregate
  estimand_weights = cohort_means |>
    filter(!is.na(tau_g)) |>
    group_by(estimand) |>
    mutate(
      N_k = sum(n_g),
      w_g = n_g / N_k
    ) |>
    ungroup()

  agg_estimates = estimand_weights |>
    group_by(estimand) |>
    summarise(
      tau_k = sum(w_g * tau_g),
      N_k = first(N_k),
      n_cohorts = n(),
      .groups = "drop"
    )

  # Stack and reweight IFs for variance estimation
  stacked_if = bind_rows(lapply(names(if_list), function(g) {
    df = if_list[[g]]
    df$cohort = as.integer(g)
    df
  }))

  reweighted_if = matrix(0, nrow = nrow(stacked_if), ncol = length(all_if_cols))
  colnames(reweighted_if) = all_if_cols

  for (col in all_if_cols) {
    if (col %in% names(stacked_if)) {
      N_k = agg_estimates$N_k[agg_estimates$estimand == col]
      if (length(N_k) == 1 && !is.na(N_k)) {
        vals = stacked_if[[col]] * (N_total / N_k)
        vals[is.na(vals)] = 0
        reweighted_if[, col] = vals
      }
    }
  }

  reweighted_if_df = as.data.frame(reweighted_if)
  reweighted_if_df$county_fips = stacked_if$county_fips
  reweighted_if_df$cohort = stacked_if$cohort

  list(
    estimates = agg_estimates,
    weights = estimand_weights,
    reweighted_if = reweighted_if_df,
    N_total = N_total
  )
}

################################################################################
# Event Study Construction
################################################################################

# Build event study table from aggregated estimates
#
# Computes clustered standard errors and confidence intervals for each
# event-time estimate with optional Bonferroni correction.
#
# @param agg Aggregation results from aggregate_cohort_ifs()
# @param alpha Significance level (default 0.05)
# @param bonferroni Apply Bonferroni correction across all tests?
# @return Data frame with event study estimates
build_event_study = function(agg, alpha = 0.05, bonferroni = TRUE) {

  # Compute clustered SEs
  results_agg = agg$estimates |>
    rowwise() |>
    mutate(
      se_clustered = clustered_se(
        agg$reweighted_if[[estimand]],
        agg$reweighted_if$county_fips
      )
    ) |>
    ungroup()

  # Parse event time from estimand names
  es_df = results_agg |>
    mutate(
      type = case_when(
        str_detect(estimand, "^IF_pre_k")  ~ "pre",
        str_detect(estimand, "^IF_post_k") ~ "post",
        TRUE ~ "other"
      ),
      k = as.numeric(str_extract(estimand, "\\d+$"))
    ) |>
    filter(type %in% c("pre", "post"), !is.na(k)) |>
    mutate(
      event_time = if_else(type == "pre", -k, k)
    ) |>
    arrange(event_time)

  # Confidence intervals
  n_tests = if (bonferroni) nrow(es_df) else 1
  z_crit = qnorm(1 - (alpha / n_tests) / 2)

  es_df |>
    mutate(
      t_stat = tau_k / se_clustered,
      pval = 2 * pnorm(-abs(t_stat)),
      ci_lo = tau_k - z_crit * se_clustered,
      ci_hi = tau_k + z_crit * se_clustered
    )
}

################################################################################
# Visualization
################################################################################

# Create event study plot
#
# Generates publication-ready event study plot with confidence bands,
# properly formatted axes, and informative caption.
#
# @param es_df Event study data frame from build_event_study()
# @param title Plot title
# @param alpha Significance level (for caption)
# @param bonferroni Whether Bonferroni correction was applied
# @param n_cohorts Number of cohorts in the analysis
# @param y_limits Optional y-axis limits
# @return ggplot object
plot_event_study = function(es_df, title = "Event Study",
                            alpha = 0.05, bonferroni = TRUE,
                            n_cohorts = NA, y_limits = NULL) {

  n_tests = nrow(es_df)
  x_breaks = sort(unique(es_df$event_time))
  if (length(x_breaks) > 20) {
    x_breaks = pretty(range(es_df$event_time), n = 10)
  }

  ggplot(es_df, aes(x = event_time, y = tau_k)) +
    geom_hline(yintercept = 0, linewidth = 0.4) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.5, color = "gray40") +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.15, fill = "steelblue") +
    geom_line(linewidth = 0.6, color = "steelblue4") +
    geom_point(aes(shape = type), size = 2.5, color = "steelblue4") +
    scale_x_continuous(breaks = x_breaks) +
    scale_y_continuous(
      n.breaks = 10,
      limits = y_limits,
      labels = label_number(accuracy = 0.01)
    ) +
    scale_shape_manual(
      values = c(pre = 16, post = 17),
      labels = c(pre = "Pre-treatment", post = "Post-treatment")
    ) +
    labs(
      title = title,
      x = "Event time (years relative to treatment)",
      y = expression(hat(tau)[k]),
      shape = NULL,
      caption = sprintf(
        "Notes: %s%% CI%s. Standard errors clustered by county. N = %s observations%s.",
        round((1 - alpha) * 100),
        if (bonferroni) sprintf(" (Bonferroni-corrected, %d tests)", n_tests) else "",
        format(sum(es_df$N_k[!duplicated(es_df$event_time)]), big.mark = ","),
        if (!is.na(n_cohorts)) sprintf(" across %d cohorts", n_cohorts) else ""
      )
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.caption = element_text(hjust = 0, size = 9, color = "gray40"),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.margin = margin(t = -5)
    )
}

################################################################################
# Summary Statistics
################################################################################

# Compute averaged pre/post treatment effects
#
# Aggregates event-time estimates within pre- and post-treatment periods
# and computes clustered standard errors using fixest.
#
# @param es_df Event study data frame
# @param agg Aggregation results (for reweighted IFs)
# @return List with pre and post average estimates
compute_average_effects = function(es_df, agg) {

  results = list()

  for (period in c("pre", "post")) {
    rows = es_df |> filter(type == period)

    if (nrow(rows) == 0) next

    cols = rows$estimand
    avg_if = rowMeans(agg$reweighted_if[, cols, drop = FALSE])

    fit = feols(
      psi ~ 1,
      data = data.frame(psi = avg_if, cl = agg$reweighted_if$county_fips),
      cluster = ~cl
    )

    results[[period]] = list(
      mean = mean(rows$tau_k),
      fit = fit,
      n_periods = nrow(rows)
    )
  }

  results
}
