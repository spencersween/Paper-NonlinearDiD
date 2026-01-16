################################################################################
#
# Utility Functions and Constants
#
# Core utilities for the nonlinear DiD structural estimation framework.
# Includes constants, null coalescing, tensor operations, gradient utilities,
# learning rate management, and cross-validation fold construction.
#
################################################################################

suppressPackageStartupMessages({
  library(torch)
})

################################################################################
# Constants
################################################################################

DEFAULTS = list(
  seed = 1234L,
  eps = 1e-6,
  large_loss = 1e6,
  poisson_clamp = 1e-6
)

################################################################################
# Basic Utilities
################################################################################

# Null-coalescing operator
# @param a Primary value
# @param b Fallback value if a is NULL
`%||%` = function(a, b) if (!is.null(a)) a else b

# Assert that x is a torch tensor
# @param x Object to check
# @param name Name for error message
assert_torch_tensor = function(x, name = "tensor") {
  if (!inherits(x, "torch_tensor")) {
    stop(sprintf("%s must be a torch_tensor.", name))
  }
  invisible(TRUE)
}

# Set all random seeds for reproducibility
# @param seed Integer seed value
set_all_seeds = function(seed = DEFAULTS$seed) {
  set.seed(seed)
  torch_manual_seed(as.integer(seed))
  invisible(TRUE)
}

# Convert to numeric vector (safely)
# @param x Input vector
as_numeric_vector = function(x) as.numeric(x)

# Index tensor rows, handling empty indices
# @param tensor Torch tensor
# @param idx Integer indices
index_tensor_rows = function(tensor, idx) {
  if (length(idx) == 0) {
    ncol = if (tensor$dim() >= 2) tensor$size(2) else 1
    if (tensor$dim() == 1) return(torch_empty(c(0)))
    return(torch_empty(c(0, ncol)))
  }
  tensor[idx, ..]
}

# Check if tensor contains a finite scalar
# @param x Torch tensor
is_finite_scalar_tensor = function(x) {
  if (!inherits(x, "torch_tensor")) return(FALSE)
  v = as.numeric(x$item())
  is.finite(v)
}

# Stop execution if tensor is non-finite
# @param x Torch tensor
# @param label Description for error message
stop_if_nonfinite = function(x, label = "value") {
  if (!is_finite_scalar_tensor(x)) {
    stop(sprintf("Non-finite %s encountered (NA/NaN/Inf).", label))
  }
  invisible(TRUE)
}

################################################################################
# Gradient Utilities
################################################################################

# Safe gradient clipping that handles NaN/Inf
#
# Clips gradients by global norm while replacing any NaN/Inf values with zeros.
#
# @param parameters List of model parameters
# @param max_norm Maximum allowed gradient norm
# @return Total gradient norm (before clipping)
safe_clip_grad_norm_ = function(parameters, max_norm) {
  total_norm = torch_tensor(0.0)

  for (p in parameters) {
    if (is.null(p$grad)) next

    grad_data = p$grad$data()

    # Check for and replace NaN/Inf gradients
    has_nan = torch_isnan(grad_data)$any()$item()
    has_inf = torch_isinf(grad_data)$any()$item()

    if (has_nan || has_inf) {
      warning("NaN or Inf detected in gradients; replacing with zeros.")
      nan_or_inf_mask = torch_isnan(grad_data) | torch_isinf(grad_data)
      p$grad$data()$masked_fill_(nan_or_inf_mask, 0.0)
      grad_data = p$grad$data()
    }

    param_norm = grad_data$norm(2)
    total_norm = total_norm + param_norm$pow(2)
  }

  total_norm = total_norm$sqrt()
  total_norm_val = total_norm$item()

  if (!is.finite(total_norm_val) || total_norm_val == 0) {
    return(invisible(total_norm))
  }

  clip_coef = max_norm / (total_norm + DEFAULTS$eps)
  clip_coef_val = clip_coef$item()

  if (is.finite(clip_coef_val) && clip_coef_val < 1) {
    for (p in parameters) {
      if (!is.null(p$grad)) {
        p$grad$data()$mul_(clip_coef)
      }
    }
  }

  invisible(total_norm)
}

################################################################################
# Learning Rate Utilities
################################################################################

# Get current learning rate from optimizer
# @param opt Torch optimizer
get_optimizer_lr = function(opt) {
  as.numeric(opt$param_groups[[1]]$lr)
}

# Set learning rate for optimizer
# @param opt Torch optimizer
# @param lr New learning rate
set_optimizer_lr = function(opt, lr) {
  for (i in seq_along(opt$param_groups)) {
    opt$param_groups[[i]]$lr = lr
  }
  invisible(TRUE)
}

# Apply learning rate step decay
# @param opt Torch optimizer
# @param epoch Current epoch
# @param lr_step Epochs between LR reductions
# @param lr_gamma Multiplicative factor
# @param verbose Print LR changes
maybe_step_lr = function(opt, epoch, lr_step = 1000L, lr_gamma = 0.1, verbose = TRUE) {
  lr_step = as.integer(lr_step)
  if (lr_step <= 0L || epoch %% lr_step != 0L) {
    return(invisible(FALSE))
  }

  old_lr = get_optimizer_lr(opt)
  new_lr = old_lr * lr_gamma
  set_optimizer_lr(opt, new_lr)

  if (isTRUE(verbose)) {
    message(sprintf("LR decay at epoch %d: %.2e -> %.2e", epoch, old_lr, new_lr))
  }

  invisible(TRUE)
}

################################################################################
# Cross-Validation Fold Construction
################################################################################

# Create k-fold splits respecting group structure
#
# Ensures all observations from the same group are in the same fold.
#
# @param group_vec Vector of group identifiers
# @param k_folds Number of folds
# @param seed Random seed
# @param shuffle Whether to shuffle groups before assignment
# @return Integer vector of fold assignments
make_folds_by_group = function(group_vec, k_folds, seed = DEFAULTS$seed, shuffle = TRUE) {
  group_vec = as_numeric_vector(group_vec)
  n = length(group_vec)

  if (n < 2) stop("group_vec must have length >= 2.")
  if (k_folds < 2) stop("k_folds must be >= 2.")

  set.seed(seed)
  groups = sort(unique(group_vec))
  if (shuffle) groups = sample(groups)

  fold_id_by_group = rep(1:k_folds, length.out = length(groups))
  names(fold_id_by_group) = groups

  as.integer(fold_id_by_group[as.character(group_vec)])
}

# Split training indices into train/validation by group
#
# @param train_idx Indices designated for training
# @param group_vec Full group vector
# @param val_frac Fraction of groups to use for validation
# @param seed Random seed
# @param shuffle Whether to shuffle groups
# @return List with train_idx and val_idx
split_train_val_by_group = function(train_idx, group_vec, val_frac = 0.10,
                                     seed = DEFAULTS$seed, shuffle = TRUE) {
  if (length(train_idx) < 2) {
    stop("Not enough training points to split train/val.")
  }

  group_vec = as_numeric_vector(group_vec)
  set.seed(seed)

  groups_tr = sort(unique(group_vec[train_idx]))
  if (shuffle) groups_tr = sample(groups_tr)

  n_groups = length(groups_tr)
  n_val_groups = max(1L, floor(n_groups * val_frac))
  if (n_groups - n_val_groups < 1L) {
    n_val_groups = n_groups - 1L
  }

  val_groups = groups_tr[seq_len(n_val_groups)]
  val_idx = train_idx[group_vec[train_idx] %in% val_groups]
  tr_idx = setdiff(train_idx, val_idx)

  list(train_idx = tr_idx, val_idx = val_idx)
}
