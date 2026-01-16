################################################################################
#
# Model Training and Cross-Fitting
#
# Implements cross-fitted neural network estimation with:
#   - Early stopping with model checkpointing
#   - Group-aware cross-validation for clustered data
#   - Batch training and L-BFGS optimization
#   - Training history visualization
#
################################################################################

suppressPackageStartupMessages({
  library(torch)
  library(ggplot2)
  library(data.table)
  library(dplyr)
})

source("code/r/utilities.R")
source("code/r/neural_network.R")

################################################################################
# Early Stopping
################################################################################

# Initialize early stopping state
# @param patience Epochs to wait for improvement
# @param min_delta Minimum improvement threshold
early_stopper_init = function(patience = 20L, min_delta = 0.0) {
  list(
    best = Inf,
    best_epoch = 0L,
    patience = as.integer(patience),
    min_delta = min_delta,
    wait = 0L,
    best_state = NULL
  )
}

# Update early stopping state
# @param stopper Early stopper state
# @param model Current model
# @param val_loss Current validation loss
# @param epoch Current epoch
# @return List with updated stopper and should_stop flag
early_stopper_update = function(stopper, model, val_loss, epoch) {
  improved = (stopper$best - val_loss) > stopper$min_delta

  if (improved) {
    stopper$best = val_loss
    stopper$best_epoch = as.integer(epoch)
    stopper$wait = 0L
    stopper$best_state = model$state_dict()
  } else {
    stopper$wait = stopper$wait + 1L
  }

  list(stopper = stopper, should_stop = stopper$wait >= stopper$patience)
}

# Restore best model state
# @param stopper Early stopper state
# @param model Model to restore
early_stopper_restore_best = function(stopper, model) {
  if (!is.null(stopper$best_state)) {
    model$load_state_dict(stopper$best_state)
  }
  model
}

################################################################################
# Evaluation Helpers
################################################################################

# Get model predictions
# @param model Joint model
# @param x Covariate tensor
# @param d Treatment tensor
# @param device Computation device
predict_joint = function(model, x, d, device = "cpu") {
  model$eval()
  with_no_grad({
    model(x$to(device = device), d$to(device = device))
  })
}

# Compute loss on a dataset
# @param model Joint model
# @param x Covariate tensor
# @param d Treatment tensor
# @param y Outcome tensor
# @param total_loss_fn Loss function
# @param device Computation device
# @return Scalar loss value
compute_loss_dataset = function(model, x, d, y, total_loss_fn, device = "cpu") {
  model$eval()

  loss = with_no_grad({
    out = model(x$to(device = device), d$to(device = device))
    total_loss_fn(model, out$yhat, y$to(device = device), out$p_logit, d$to(device = device))
  })

  loss_val = as.numeric(loss$item())

  if (!is.finite(loss_val)) {
    warning("Non-finite loss detected; returning large value.")
    return(DEFAULTS$large_loss)
  }

  loss_val
}

################################################################################
# Single Fold Training
################################################################################

# Train model on one fold
#
# @param Y_tensor Outcome tensor (n x J)
# @param D_tensor Treatment tensor (n x 1)
# @param X_tensor Covariate tensor (n x p)
# @param train_idx Training indices
# @param val_idx Validation indices
# @param test_idx Test indices
# @param net_builder_fn Function to build network architecture
# @param total_loss_fn Loss function
# @param optimizer_spec Optimizer specification list
# @param max_epochs Maximum training epochs
# @param batch_size Mini-batch size (NULL for full batch)
# @param use_early_stopping Enable early stopping
# @param keep_best_model Restore best model after training
# @param early_patience Early stopping patience
# @param early_min_delta Minimum improvement for early stopping
# @param grad_clip_norm Maximum gradient norm (NULL to disable)
# @param lr_step Epochs between LR decay
# @param lr_gamma LR decay factor
# @param loss_type "mse" or "poisson"
# @param lower_bound Lower bound for predictions
# @param upper_bound Upper bound for predictions
# @param device Computation device
# @param verbose Print progress
# @return List with trained model and diagnostics
train_one_fold_joint = function(Y_tensor, D_tensor, X_tensor,
                                 train_idx, val_idx, test_idx,
                                 net_builder_fn, total_loss_fn, optimizer_spec,
                                 max_epochs = 200L, batch_size = NULL,
                                 use_early_stopping = TRUE, keep_best_model = TRUE,
                                 early_patience = 20L, early_min_delta = 0.0,
                                 grad_clip_norm = NULL, lr_step = 1000L, lr_gamma = 0.1,
                                 loss_type = c("mse", "poisson"),
                                 lower_bound = 1e-5, upper_bound = 10000,
                                 device = "cpu", verbose = TRUE) {
  loss_type = match.arg(loss_type)

  assert_torch_tensor(Y_tensor, "Y_tensor")
  assert_torch_tensor(D_tensor, "D_tensor")
  assert_torch_tensor(X_tensor, "X_tensor")

  # Ensure correct dimensions
  if (Y_tensor$dim() == 1) Y_tensor = Y_tensor$unsqueeze(2)
  if (D_tensor$dim() == 1) D_tensor = D_tensor$unsqueeze(2)

  # Extract fold data
  x_train = index_tensor_rows(X_tensor, train_idx)
  d_train = index_tensor_rows(D_tensor, train_idx)
  y_train = index_tensor_rows(Y_tensor, train_idx)

  x_val = index_tensor_rows(X_tensor, val_idx)
  d_val = index_tensor_rows(D_tensor, val_idx)
  y_val = index_tensor_rows(Y_tensor, val_idx)

  x_test = index_tensor_rows(X_tensor, test_idx)
  d_test = index_tensor_rows(D_tensor, test_idx)
  y_test = index_tensor_rows(Y_tensor, test_idx)

  # Build model
  input_dim = as.integer(X_tensor$size(2))
  J = as.integer(Y_tensor$size(2))
  output_dim = 2L * J + 1L

  shared_net = net_builder_fn(input_dim = input_dim, output_dim = output_dim)
  model = make_joint_model(shared_net, J_outcomes = J, loss_type = loss_type,
                            lower_bound = lower_bound, upper_bound = upper_bound)
  model = model$to(device = device)

  # Setup optimizer
  opt_wrap = do.call(make_optimizer, c(list(model = model), optimizer_spec))
  opt_type = opt_wrap$type
  opt = opt_wrap$opt

  # Setup early stopping
  stopper = early_stopper_init(patience = early_patience, min_delta = early_min_delta)
  history = data.frame(epoch = integer(0), train_loss = numeric(0), val_loss = numeric(0))

  # Batch generator
  make_batches = function(n, batch_size) {
    if (is.null(batch_size) || batch_size >= n) return(list(seq_len(n)))
    idx = sample(seq_len(n))
    split(idx, ceiling(seq_along(idx) / batch_size))
  }

  # Training loop
  for (epoch in seq_len(as.integer(max_epochs))) {
    model$train()

    if (opt_type == "lbfgs") {
      # L-BFGS requires closure
      closure = function() {
        opt$zero_grad()
        out = model(x_train$to(device = device), d_train$to(device = device))
        loss = total_loss_fn(model, out$yhat, y_train$to(device = device),
                              out$p_logit, d_train$to(device = device))
        loss$backward()
        if (!is.null(grad_clip_norm)) {
          safe_clip_grad_norm_(model$parameters, max_norm = grad_clip_norm)
        }
        loss
      }

      loss_train_tensor = opt$step(closure)
      train_loss = as.numeric(loss_train_tensor$item())
      if (!is.finite(train_loss)) {
        warning(sprintf("Non-finite training loss at epoch %d", epoch))
        train_loss = DEFAULTS$large_loss
      }

    } else {
      # Mini-batch SGD/Adam
      ntr = x_train$size(1)
      batches = make_batches(as.integer(ntr), batch_size)
      epoch_losses = c()

      for (b in batches) {
        xb = x_train[b, ..]$to(device = device)
        db = d_train[b, ..]$to(device = device)
        yb = y_train[b, ..]$to(device = device)

        opt$zero_grad()
        out = model(xb, db)
        loss = total_loss_fn(model, out$yhat, yb, out$p_logit, db)

        loss_val = as.numeric(loss$item())
        if (!is.finite(loss_val)) {
          warning(sprintf("Non-finite batch loss at epoch %d; skipping.", epoch))
          next
        }

        loss$backward()

        if (!is.null(grad_clip_norm)) {
          safe_clip_grad_norm_(model$parameters, max_norm = grad_clip_norm)
        }

        opt$step()
        epoch_losses = c(epoch_losses, loss_val)
      }

      train_loss = if (length(epoch_losses) > 0) mean(epoch_losses) else DEFAULTS$large_loss
    }

    # Validation
    val_loss = compute_loss_dataset(model, x_val, d_val, y_val, total_loss_fn, device)
    history = rbind(history, data.frame(epoch = epoch, train_loss = train_loss, val_loss = val_loss))

    # Logging
    if (verbose && (epoch == 1L || epoch %% 10L == 0L)) {
      message(sprintf("  Epoch %4d | train: %.4f | val: %.4f", epoch, train_loss, val_loss))
    }

    # Early stopping check
    if (isTRUE(use_early_stopping)) {
      upd = early_stopper_update(stopper, model, val_loss, epoch)
      stopper = upd$stopper

      if (isTRUE(upd$should_stop)) {
        if (verbose) {
          message(sprintf("  Early stop at epoch %d (best: epoch %d, val: %.4f)",
                          epoch, stopper$best_epoch, stopper$best))
        }
        break
      }
    }

    # LR scheduling
    maybe_step_lr(opt, epoch = epoch, lr_step = lr_step, lr_gamma = lr_gamma, verbose = verbose)
  }

  # Restore best model
  if (isTRUE(use_early_stopping) && isTRUE(keep_best_model)) {
    model = early_stopper_restore_best(stopper, model)
  }

  # Test evaluation
  test_loss = compute_loss_dataset(model, x_test, d_test, y_test, total_loss_fn, device)
  test_out = predict_joint(model, x_test, d_test, device)

  list(
    model = model,
    history = history,
    test_loss = test_loss,
    test_yhat = test_out$yhat$to(device = "cpu"),
    test_p_logit = test_out$p_logit$to(device = "cpu"),
    test_a = test_out$a$to(device = "cpu"),
    test_b = test_out$b$to(device = "cpu")
  )
}

################################################################################
# Visualization
################################################################################

# Plot training and validation loss curves
# @param history_df Data frame with epoch, train_loss, val_loss
# @param fold_id Fold identifier for title
# @return ggplot object
plot_loss_curves = function(history_df, fold_id) {
  df = rbind(
    data.frame(epoch = history_df$epoch, loss = history_df$train_loss, series = "Training"),
    data.frame(epoch = history_df$epoch, loss = history_df$val_loss, series = "Validation")
  )

  ggplot(df, aes(x = epoch, y = loss, color = series, linetype = series)) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = c(Training = "steelblue", Validation = "coral")) +
    labs(
      title = sprintf("Loss Curves (Fold %d)", fold_id),
      x = "Epoch",
      y = "Loss",
      color = NULL,
      linetype = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold")
    )+
    scale_y_continuous(limits = c(-100000,100000))
}

################################################################################
# Cross-Fitting Driver
################################################################################

# Cross-fit joint model across all folds
#
# Performs K-fold cross-fitting to obtain out-of-fold predictions for:
#   - yhat: predicted outcomes
#   - a: E[Y|X,D=0]
#   - b: E[Y|X,D=1]
#   - p_scores: P(D=1|X)
#
# @param Y_tensor Outcome tensor
# @param D_tensor Treatment tensor
# @param X_tensor Covariate tensor
# @param folds Fold assignment vector
# @param group_vec Group identifier vector
# @param val_frac Validation fraction within training folds
# @param net_builder_fn Network architecture builder
# @param loss_type "mse" or "poisson"
# @param l2_lambda L2 (ridge) regularization strength
# @param l1_lambda L1 (lasso) regularization strength
# @param use_explicit_l2 Whether to use explicit L2 penalty (vs optimizer weight decay)
# @param propensity_weight Weight on propensity loss
# @param optimizer "adamw" or "lbfgs"
# @param optimizer_args Additional optimizer arguments
# @param max_epochs Maximum epochs per fold
# @param batch_size Mini-batch size
# @param use_early_stopping Enable early stopping
# @param keep_best_model Restore best model
# @param early_patience Early stopping patience
# @param early_min_delta Minimum improvement threshold
# @param grad_clip_norm Maximum gradient norm
# @param lr_step LR decay interval
# @param lr_gamma LR decay factor
# @param lower_bound Prediction lower bound
# @param upper_bound Prediction upper bound
# @param device Computation device
# @param seed Random seed
# @param verbose Print progress
# @param make_plots Generate loss curve plots
# @param y_colnames Names for outcome columns
# @return List with out-of-fold predictions and diagnostics
crossfit_train_joint = function(Y_tensor, D_tensor, X_tensor, folds, group_vec,
                                 val_frac = 0.10, net_builder_fn,
                                 loss_type = c("mse", "poisson"),
                                 l2_lambda = 0.0, l1_lambda = 0.0,
                                 use_explicit_l2 = TRUE,
                                 propensity_weight = 1.0,
                                 optimizer = c("adamw", "lbfgs"), optimizer_args = list(),
                                 max_epochs = 200L, batch_size = NULL,
                                 use_early_stopping = TRUE, keep_best_model = TRUE,
                                 early_patience = 20L, early_min_delta = 0.0,
                                 grad_clip_norm = NULL, lr_step = 1000L, lr_gamma = 0.1,
                                 lower_bound = 1e-5, upper_bound = 10000,
                                 device = "cpu", seed = DEFAULTS$seed,
                                 verbose = TRUE, make_plots = TRUE, y_colnames = NULL) {
  loss_type = match.arg(loss_type)
  optimizer = match.arg(optimizer)

  assert_torch_tensor(Y_tensor, "Y_tensor")
  assert_torch_tensor(D_tensor, "D_tensor")
  assert_torch_tensor(X_tensor, "X_tensor")

  n = as.integer(X_tensor$size(1))
  if (length(folds) != n) stop("folds must have length nrow(X_tensor).")
  if (length(group_vec) != n) stop("group_vec must have length nrow(X_tensor).")

  # Loss function setup
  # For AdamW: use optimizer weight decay if use_explicit_l2=FALSE, otherwise explicit L2
  # For L-BFGS: always use explicit L2
  use_optimizer_weight_decay = (optimizer == "adamw" && !use_explicit_l2 && l2_lambda > 0)

  total_loss_fn = make_joint_loss_fn(
    loss_type = loss_type,
    l2_lambda = l2_lambda,
    l1_lambda = l1_lambda,
    use_explicit_l2 = use_explicit_l2 || optimizer == "lbfgs",
    propensity_weight = propensity_weight
  )

  # Ensure Y is 2D
  if (Y_tensor$dim() == 1) Y_tensor = Y_tensor$unsqueeze(2)
  J = as.integer(Y_tensor$size(2))

  # Initialize output storage
  yhat_oof = matrix(NA_real_, nrow = n, ncol = J)
  a_oof = matrix(NA_real_, nrow = n, ncol = J)
  b_oof = matrix(NA_real_, nrow = n, ncol = J)
  p_logit_oof = rep(NA_real_, n)

  K = max(folds)
  models = vector("list", K)
  histories = vector("list", K)
  test_losses = rep(NA_real_, K)
  plots = vector("list", K)
  splits = vector("list", K)

  # Train each fold
  for (k in sort(unique(folds))) {
    if (verbose) message(sprintf("\n[Fold %d/%d]", k, K))

    test_idx = which(folds == k)
    train_full_idx = which(folds != k)

    # Split train into train/val
    split = split_train_val_by_group(
      train_idx = train_full_idx,
      group_vec = group_vec,
      val_frac = val_frac,
      seed = seed + k,
      shuffle = TRUE
    )

    train_idx = split$train_idx
    val_idx = split$val_idx
    splits[[k]] = list(train_idx = train_idx, val_idx = val_idx, test_idx = test_idx)

    # Optimizer specification
    optimizer_spec = c(
      list(
        optimizer = optimizer,
        lr = optimizer_args$lr %||% 1e-3,
        weight_decay = if (use_optimizer_weight_decay) l2_lambda else 0.0
      ),
      optimizer_args[names(optimizer_args) != "lr"]
    )

    # Train fold
    res = train_one_fold_joint(
      Y_tensor = Y_tensor, D_tensor = D_tensor, X_tensor = X_tensor,
      train_idx = train_idx, val_idx = val_idx, test_idx = test_idx,
      net_builder_fn = net_builder_fn, total_loss_fn = total_loss_fn,
      optimizer_spec = optimizer_spec, max_epochs = max_epochs,
      batch_size = batch_size, use_early_stopping = use_early_stopping,
      keep_best_model = keep_best_model, early_patience = early_patience,
      early_min_delta = early_min_delta, grad_clip_norm = grad_clip_norm,
      lr_step = lr_step, lr_gamma = lr_gamma, loss_type = loss_type,
      lower_bound = lower_bound, upper_bound = upper_bound,
      device = device, verbose = verbose
    )

    # Store out-of-fold predictions
    yhat_oof[test_idx, ] = as.array(res$test_yhat$to(device = "cpu"))
    a_oof[test_idx, ] = as.array(res$test_a$to(device = "cpu"))
    b_oof[test_idx, ] = as.array(res$test_b$to(device = "cpu"))
    p_logit_oof[test_idx] = as.numeric(res$test_p_logit$to(device = "cpu"))

    models[[k]] = res$model
    histories[[k]] = res$history
    test_losses[k] = res$test_loss

    if (make_plots) {
      plots[[k]] = plot_loss_curves(res$history, fold_id = k)
      print(plots[[k]])
    }
  }

  # Set column names
  colnames(yhat_oof) = y_colnames %||% paste0("yhat_", seq_len(J))
  colnames(a_oof) = y_colnames %||% paste0("a_", seq_len(J))
  colnames(b_oof) = y_colnames %||% paste0("b_", seq_len(J))

  # Convert logits to probabilities
  p_scores = plogis(p_logit_oof)

  list(
    models = models,
    oof_yhat = yhat_oof,
    oof_a = a_oof,
    oof_b = b_oof,
    oof_p_logit = p_logit_oof,
    oof_p_scores = p_scores,
    y_names = y_colnames,
    test_losses = test_losses,
    avg_test_loss = mean(test_losses, na.rm = TRUE),
    histories = histories,
    plots = plots,
    folds = folds,
    splits = splits,
    cfg = list(
      loss_type = loss_type,
      l2_lambda = l2_lambda,
      l1_lambda = l1_lambda,
      use_explicit_l2 = use_explicit_l2,
      propensity_weight = propensity_weight,
      optimizer = optimizer,
      optimizer_args = optimizer_args,
      batch_size = batch_size,
      grad_clip_norm = grad_clip_norm,
      lr_step = lr_step,
      lr_gamma = lr_gamma,
      lower_bound = lower_bound,
      upper_bound = upper_bound,
      device = device
    )
  )
}

################################################################################
# Data Interface
################################################################################

# Build torch tensors from data frame
#
# Extracts Y (outcomes), D (treatment), X (covariates) based on column prefixes.
# Standardizes X to zero mean and unit variance.
#
# @param df Data frame
# @param y_regex Regex for outcome columns
# @param d_regex Regex for treatment column
# @param x_regex Regex for covariate columns
# @param enforce_numeric Enforce numeric conversion
# @param check_poisson_assumptions Warn about Poisson requirements
# @return List with tensors and column names
build_tensors_from_df = function(df, y_regex = "^Y_", d_regex = "^D_", x_regex = "^X_",
                                  enforce_numeric = TRUE, check_poisson_assumptions = TRUE) {
  stopifnot(is.data.frame(df))

  nm = names(df)
  y_cols = grep(y_regex, nm, value = TRUE)
  d_cols = grep(d_regex, nm, value = TRUE)
  x_cols = grep(x_regex, nm, value = TRUE)

  if (length(y_cols) == 0) stop("No outcome columns found (expected 'Y_*').")
  if (length(d_cols) != 1) stop("Expected exactly one treatment column ('D_*').")
  if (length(x_cols) == 0) stop("No covariate columns found (expected 'X_*').")

  Y = as.matrix(df[, y_cols, drop = FALSE])
  D = as.matrix(df[, d_cols, drop = FALSE])
  X = scale(as.matrix(df[, x_cols, drop = FALSE]))

  if (isTRUE(enforce_numeric)) {
    to_num_matrix = function(M, label) {
      M2 = suppressWarnings(apply(M, 2, as.numeric))
      if (is.null(dim(M2))) M2 = matrix(M2, ncol = 1)

      if (anyNA(M2)) {
        bad_cols = colnames(M)[apply(M2, 2, anyNA)]
        stop(sprintf("Non-numeric values in %s columns: %s", label, paste(bad_cols, collapse = ", ")))
      }

      storage.mode(M2) = "double"
      M2
    }

    Y = to_num_matrix(Y, "Y")
    D = to_num_matrix(D, "D")
    X = to_num_matrix(X, "X")
  } else {
    storage.mode(Y) = "double"
    storage.mode(D) = "double"
    storage.mode(X) = "double"
  }

  # Check Poisson assumptions
  if (isTRUE(check_poisson_assumptions)) {
    if (any(Y < 0)) {
      warning("Y contains negative values. Poisson loss requires Y >= 0.")
    }

    y_range = range(Y)
    y_mean = mean(Y)
    message(sprintf("Outcome statistics: min=%.2f, max=%.2f, mean=%.2f",
                    y_range[1], y_range[2], y_mean))

    if (y_mean > 10) {
      message("Note: Large Y values may produce negative Poisson NLL (this is expected).")
    }
  }

  # Verify dimensions
  n = nrow(df)
  if (nrow(Y) != n || nrow(D) != n || nrow(X) != n) {
    stop("Row count mismatch among Y/D/X blocks.")
  }

  list(
    Y_tensor = torch_tensor(Y, dtype = torch_float()),
    D_tensor = torch_tensor(D, dtype = torch_float()),
    X_tensor = torch_tensor(X, dtype = torch_float()),
    y_cols = y_cols,
    d_cols = d_cols,
    x_cols = x_cols
  )
}

################################################################################
# Main Entry Point
################################################################################

# Run structural estimation pipeline
#
# Main function to estimate conditional outcome means and propensity scores
# using cross-fitted neural networks.
#
# @param df Data frame with Y_*, D_*, X_* columns
# @param hyperparams List of training hyperparameters
# @param group_vec Vector of group identifiers for clustered CV
# @param loss_type "mse" or "poisson"
# @param propensity_weight Weight on propensity score loss
# @param lower_bound Lower bound for predictions
# @param upper_bound Upper bound for predictions
# @return List with data, model outputs, and configuration
run_joint_structural = function(df, hyperparams, group_vec,
                                 loss_type = c("mse", "poisson"),
                                 propensity_weight = 1.0,
                                 lower_bound = 1e-5,
                                 upper_bound = 10000) {
  loss_type = match.arg(loss_type)

  # Build tensors
  tensors = build_tensors_from_df(df)
  set_all_seeds(hyperparams$seed)

  # Validate group vector
  n = nrow(df)
  if (length(group_vec) != n) {
    stop("group_vec must have length nrow(df).")
  }
  group_vec = as_numeric_vector(group_vec)

  # Create folds
  folds = make_folds_by_group(
    group_vec = group_vec,
    k_folds = hyperparams$k_folds,
    seed = hyperparams$seed,
    shuffle = TRUE
  )

  # Network builder
  net_builder_fn = function(input_dim, output_dim) {
    make_mlp(
      input_dim = input_dim,
      output_dim = output_dim,
      hidden_sizes = hyperparams$hidden_sizes,
      activation = hyperparams$activation,
      dropout = hyperparams$dropout
    )
  }

  # Run cross-fitting
  joint_results = crossfit_train_joint(
    Y_tensor = tensors$Y_tensor,
    D_tensor = tensors$D_tensor,
    X_tensor = tensors$X_tensor,
    folds = folds,
    group_vec = group_vec,
    val_frac = hyperparams$val_frac,
    net_builder_fn = net_builder_fn,
    loss_type = loss_type,
    l2_lambda = hyperparams$l2_lambda,
    l1_lambda = hyperparams$l1_lambda,
    use_explicit_l2 = hyperparams$use_explicit_l2,
    propensity_weight = propensity_weight,
    optimizer = hyperparams$optimizer,
    optimizer_args = hyperparams$optimizer_args,
    max_epochs = hyperparams$max_epochs,
    batch_size = hyperparams$batch_size,
    use_early_stopping = hyperparams$use_early_stopping,
    keep_best_model = hyperparams$keep_best_model,
    early_patience = hyperparams$early_patience,
    early_min_delta = hyperparams$early_min_delta,
    grad_clip_norm = hyperparams$grad_clip_norm,
    lr_step = hyperparams$lr_step,
    lr_gamma = hyperparams$lr_gamma,
    lower_bound = lower_bound,
    upper_bound = upper_bound,
    device = hyperparams$device,
    seed = hyperparams$seed,
    verbose = hyperparams$verbose,
    make_plots = hyperparams$make_plots,
    y_colnames = tensors$y_cols
  )

  list(
    data = list(
      df = df,
      folds = folds,
      y_cols = tensors$y_cols,
      d_cols = tensors$d_cols,
      x_cols = tensors$x_cols
    ),
    joint_stage = joint_results,
    outputs = list(
      yhat = joint_results$oof_yhat,
      a = joint_results$oof_a,
      b = joint_results$oof_b,
      p_logit = joint_results$oof_p_logit,
      p_scores = joint_results$oof_p_scores
    ),
    bounds = list(
      lower = lower_bound,
      upper = upper_bound
    ),
    loss_type = loss_type
  )
}
