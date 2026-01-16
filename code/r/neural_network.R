################################################################################
#
# Neural Network Building Blocks
#
# Defines MLP architectures, bounded transformations, and joint models for
# structural causal inference. The joint model estimates:
#   - Conditional outcome means: E[Y|X,D=0] and E[Y|X,D=1]
#   - Propensity scores: P(D=1|X)
#
################################################################################

suppressPackageStartupMessages({
  library(torch)
})

source("code/r/utilities.R")

################################################################################
# Layer Initialization
################################################################################

# Initialize linear layer weights
# @param layer nn_linear layer
# @param activation Activation function name
init_linear_layer = function(layer, activation) {
  stopifnot(inherits(layer, "nn_linear"))

  if (activation %in% c("relu", "gelu")) {
    nn_init_kaiming_normal_(layer$weight, nonlinearity = "relu")
  } else {
    nn_init_xavier_normal_(layer$weight)
  }

  if (!is.null(layer$bias)) {
    nn_init_constant_(layer$bias, 0)
  }

  invisible(TRUE)
}

################################################################################
# MLP Construction
################################################################################

# Build a multi-layer perceptron
#
# @param input_dim Input dimension
# @param output_dim Output dimension
# @param hidden_sizes Vector of hidden layer sizes
# @param activation Activation function: "relu", "tanh", or "gelu"
# @param dropout Dropout probability
# @return nn_module MLP
make_mlp = function(input_dim, output_dim, hidden_sizes = c(64L, 64L),
                     activation = c("relu", "tanh", "gelu"), dropout = 0.0) {
  activation = match.arg(activation)

  act_layer = switch(activation,
                      relu = nn_relu,
                      tanh = nn_tanh,
                      gelu = nn_gelu
  )

  layers = list()
  prev = as.integer(input_dim)

  for (h in hidden_sizes) {
    h = as.integer(h)
    lin = nn_linear(prev, h)
    init_linear_layer(lin, activation)

    layers = c(layers, list(lin, act_layer()))
    if (dropout > 0) {
      layers = c(layers, list(nn_dropout(p = dropout)))
    }
    prev = h
  }

  out_lin = nn_linear(prev, as.integer(output_dim))
  init_linear_layer(out_lin, activation)
  layers = c(layers, list(out_lin))

  nn_module(
    initialize = function() {
      self$net = nn_sequential(!!!layers)
    },
    forward = function(x) {
      self$net(x)
    }
  )()
}

################################################################################
# Bounded Transformation
################################################################################

# Smooth bounded transformation via sigmoid
#
# Maps unbounded input to [lower_bound, upper_bound] with smooth gradients.
#
# @param x Unbounded tensor
# @param lower_bound Lower bound of output range
# @param upper_bound Upper bound of output range
# @return Bounded tensor
smooth_bounded_transform = function(x, lower_bound = 1e-5, upper_bound = 10000) {
  range_val = upper_bound - lower_bound
  lower_bound + range_val * torch_sigmoid(x)
}

################################################################################
# Joint Model Architecture
################################################################################

# Create joint model for outcomes and propensity
#
# Model outputs:
#   - a(X): E[Y|X, D=0] for each outcome
#   - b(X): E[Y|X, D=1] for each outcome
#   - p(X): P(D=1|X) via logit
#
# Predictions are yhat = a*(1-D) + b*D
#
# @param shared_net Shared feature extraction network
# @param J_outcomes Number of outcome variables
# @param loss_type "mse" or "poisson"
# @param lower_bound Lower bound for predictions
# @param upper_bound Upper bound for predictions
# @return nn_module joint model
make_joint_model = function(shared_net, J_outcomes, loss_type = c("mse", "poisson"),
                             lower_bound = 1e-5, upper_bound = 10000) {
  J_outcomes = as.integer(J_outcomes)
  loss_type = match.arg(loss_type)

  nn_module(
    initialize = function() {
      self$shared_net = shared_net
      self$J = J_outcomes
      self$loss_type = loss_type
      self$lower_bound = lower_bound
      self$upper_bound = upper_bound
      self$output_dim = 2L * J_outcomes + 1L
    },

    forward = function(x, d) {
      if (d$dim() == 1) d = d$unsqueeze(2)

      raw_out = self$shared_net(x)

      # Split outputs: [a_1,...,a_J, b_1,...,b_J, p_logit]
      ab_raw = raw_out[, 1:(2L * self$J)]
      p_logit = raw_out[, (2L * self$J + 1L)]

      ab = ab_raw$view(c(ab_raw$size(1), 2L, self$J))

      # Apply bounded transform for both loss types
      a = smooth_bounded_transform(ab[, 1, ], self$lower_bound, self$upper_bound)
      b = smooth_bounded_transform(ab[, 2, ], self$lower_bound, self$upper_bound)

      # Compute predicted outcomes
      d_expand = d$expand(c(-1L, self$J))
      yhat = a * (1.0 - d_expand) + b * d_expand

      list(yhat = yhat, p_logit = p_logit, a = a, b = b)
    }
  )()
}

################################################################################
# Loss Functions
################################################################################

# Mean squared error loss for multiple outcomes
# @param yhat Predicted values
# @param y True values
mse_loss_multi = function(yhat, y) {
  nnf_mse_loss(yhat, y, reduction = "mean")
}

# Poisson negative log-likelihood loss
#
# Loss = lambda - y * log(lambda)
#
# Note: Minimum occurs at lambda = y. For y > e ≈ 2.718, the minimum
# loss is negative, which is mathematically correct.
#
# @param yhat Predicted rate (lambda)
# @param y Observed counts
poisson_loss_multi = function(yhat, y) {
  yhat_safe = torch_clamp(yhat, min = DEFAULTS$poisson_clamp)
  y_safe = torch_clamp(y, min = 0)

  loss = yhat_safe - y_safe * torch_log(yhat_safe)

  # Handle numerical issues
  nan_mask = torch_isnan(loss) | torch_isinf(loss)
  if (nan_mask$any()$item()) {
    loss = torch_where(nan_mask, torch_full_like(loss, 10.0), loss)
  }

  torch_mean(loss)
}

# Binary cross-entropy loss for propensity
# @param p_logit Logit of propensity
# @param d Treatment indicator
bce_loss_logits = function(p_logit, d) {
  if (d$dim() == 2 && d$size(2) == 1) d = d$squeeze(2)
  nnf_binary_cross_entropy_with_logits(p_logit, d, reduction = "mean")
}

# L2 regularization penalty
# @param model Neural network model
# @param lambda Regularization strength
l2_penalty = function(model, lambda = 0.0) {
  if (lambda <= 0) return(torch_tensor(0.0))

  params = model$parameters
  if (length(params) == 0) return(torch_tensor(0.0))

  accum = torch_tensor(0.0, device = params[[1]]$device)
  for (p in params) {
    accum = accum + torch_sum(p$pow(2))
  }

  lambda * accum
}

# Create joint loss function
#
# Combines outcome loss + propensity loss + optional L2 penalty.
#
# @param loss_type "mse" or "poisson"
# @param weight_decay_lambda L2 penalty strength
# @param add_explicit_penalty Whether to add explicit L2 (vs optimizer weight decay)
# @param propensity_weight Weight on propensity loss
# @return Loss function
make_joint_loss_fn = function(loss_type = c("mse", "poisson"),
                               weight_decay_lambda = 0.0,
                               add_explicit_penalty = TRUE,
                               propensity_weight = 1.0) {
  loss_type = match.arg(loss_type)
  force(weight_decay_lambda)
  force(add_explicit_penalty)
  force(propensity_weight)

  outcome_loss_fn = switch(loss_type,
                            mse = mse_loss_multi,
                            poisson = poisson_loss_multi
  )

  function(model, yhat, y, p_logit, d) {
    outcome_loss = outcome_loss_fn(yhat, y)
    bce = bce_loss_logits(p_logit, d)

    base_loss = outcome_loss + propensity_weight * bce

    if (!isTRUE(add_explicit_penalty) || weight_decay_lambda <= 0) {
      return(base_loss)
    }

    base_loss + l2_penalty(model, lambda = weight_decay_lambda)
  }
}

################################################################################
# Optimizer Factory
################################################################################

# Create optimizer for model
# @param model Neural network model
# @param optimizer "adamw" or "lbfgs"
# @param lr Learning rate
# @param weight_decay Weight decay for AdamW
# @param ... Additional optimizer arguments
# @return List with optimizer type and object
make_optimizer = function(model, optimizer = c("adamw", "lbfgs"),
                           lr = 1e-3, weight_decay = 0.0, ...) {
  optimizer = match.arg(optimizer)

  if (optimizer == "adamw") {
    opt = optim_adamw(model$parameters, lr = lr, weight_decay = weight_decay, ...)
    return(list(type = "adamw", opt = opt))
  }

  opt = optim_lbfgs(model$parameters, lr = lr, ...)
  list(type = "lbfgs", opt = opt)
}
