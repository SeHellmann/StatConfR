#' @importFrom stats model.matrix rnorm
NULL

# Helper function to get default parameter definitions for regression models
# TODO:
# - meta-I: missing?
get_model_defs_reg <- function(model) {

  grid_positive <- seq(0.5, 3, 0.5)
  grid_identity <- seq(-1, 1, 0.5)
  grid_prob <- seq(0.1, 0.9, 0.2)

  defs <- list()

  if (model == "SDT") {
    defs$d <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
    defs$c <- list(link_fun = identity, inv_link_fun = identity, default_formula = ~ 1, grid = grid_identity)
  } else if (model == "GN") {
    defs$d <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
    defs$c <- list(link_fun = identity, inv_link_fun = identity, default_formula = ~ 1, grid = grid_identity)
    defs$sigma <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
  } else if (model == "PDA") {
    defs$d <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
    defs$c <- list(link_fun = identity, inv_link_fun = identity, default_formula = ~ 1, grid = grid_identity)
    defs$b <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
  } else if (model %in% c("IG", "ITGc", "ITGcm")) {
    defs$d <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
    defs$c <- list(link_fun = identity, inv_link_fun = identity, default_formula = ~ 1, grid = grid_identity)
    defs$m <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
  } else if (model == "WEV") {
    defs$d <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
    defs$c <- list(link_fun = identity, inv_link_fun = identity, default_formula = ~ 1, grid = grid_identity)
    defs$sigma <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
    defs$w <- list(link_fun = qlogis, inv_link_fun = plogis, default_formula = ~ 1, grid = grid_prob)
  } else if (model == "logN") {
    defs$d <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
    defs$c <- list(link_fun = identity, inv_link_fun = identity, default_formula = ~ 1, grid = grid_identity)
    defs$sigma <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
  } else if (model == "logWEV") {
    defs$d <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
    defs$c <- list(link_fun = identity, inv_link_fun = identity, default_formula = ~ 1, grid = grid_identity)
    defs$sigma <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
    defs$w <- list(link_fun = qlogis, inv_link_fun = plogis, default_formula = ~ 1, grid = grid_prob)
  } else if (model == "RCE") {
    defs$d <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
    defs$c <- list(link_fun = identity, inv_link_fun = identity, default_formula = ~ 1, grid = grid_identity)
  } else if (model == "CAS") {
    defs$d <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
    defs$c <- list(link_fun = identity, inv_link_fun = identity, default_formula = ~ 1, grid = grid_identity)
    defs$sigma <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
  #} else if (model == "meta-d'" || model == "metaSDT") {
  #  defs$d <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
  #  defs$c <- list(link_fun = identity, inv_link_fun = identity, default_formula = ~ 1, grid = grid_identity)
  #  defs$meta_d <- list(link_fun = log, inv_link_fun = exp, default_formula = ~ 1, grid = grid_positive)
  } else {
    stop(paste("Unknown model:", model))
  }
  return(defs)
}


#' Prepare Input for Regression Models
#'
#' @param data Data frame with columns stimulus, rating, correct
#' @param formulas List of formulas for parameters
#' @param param_defs List of parameter definitions (default_formula, grid, link)
#' @param nRatings Number of ratings
#' @param nInits Number of initializations
#' @return List of prepared data structures
preprocess_regression_input <- function(data, formulas, param_defs = NULL, model = NULL, nInits = 10, nRatings = NULL) {
  
  if (is.null(param_defs)) {
    if (is.null(model)) stop("Either 'param_defs' or 'model' must be provided.")
    param_defs <- get_model_defs_reg(model)
  }

  if (is.null(nRatings)) {
    if(is.factor(data$rating)) {
       nRatings <- length(levels(data$rating))
    } else {
       nRatings <- max(as.integer(data$rating))
    }
  }

  if (nRatings < 2) {
    msg <- if (!is.null(model)) paste0(model, " model requires at least 2 ratings.") else "Model requires at least 2 ratings."
    stop(msg)
  }

  if(!all(c("stimulus", "rating", "correct") %in% colnames(data))) 
    stop("Data must contain 'stimulus', 'rating', and 'correct'.")
  
  nTrials_total <- nrow(data)
  
  formula_vars <- unique(unlist(lapply(formulas, all.vars)))
  group_vars <- unique(c("stimulus", "rating", "correct", formula_vars))
  group_vars <- group_vars[group_vars %in% colnames(data)]
  
  data$.counts_internal <- 1
  data_grouped <- aggregate(.counts_internal ~ ., data = data[, c(group_vars, ".counts_internal")], FUN = length)
  counts_vec <- data_grouped$.counts_internal
  
  data <- data_grouped
  nTrials_unique <- nrow(data)
  
  if(is.factor(data$stimulus)) {
    stim_vec <- as.integer(data$stimulus) - 1 
  } else {
    stim_vec <- as.integer(as.factor(data$stimulus)) - 1
  }
  
  rating_vec <- as.integer(data$rating)
  
  correct_vec <- as.integer(data$correct)
  if(max(correct_vec) > 1) correct_vec <- correct_vec - min(correct_vec)

  nTrials <- nTrials_unique
  
  X_list <- list()
  param_names <- names(param_defs)
  nParams <- 0
  
  grid_list <- list()
  
  for(p in param_names) {
    def <- param_defs[[p]]
    
    f <- formulas[[p]]
    if(is.null(f)) f <- def$default_formula
    
    X <- model.matrix(f, data = data)
    X_list[[p]] <- X
    
    nParams <- nParams + ncol(X)
    
    grid_list[[p]] <- def$grid
  }
  
  grid_list$tau <- seq(0.5, 2.0, length.out = max(3, min(nInits, 3)))
  
  full_grid <- do.call(expand.grid, grid_list)
  
  if(nrow(full_grid) > nInits) {
     idx <- unique(round(seq(1, nrow(full_grid), length.out = nInits)))
     full_grid <- full_grid[idx, , drop = FALSE]
  }
  
  nTheta_A <- nRatings - 1
  nTheta_B <- nRatings - 1
  nTheta <- nTheta_A + nTheta_B
  nTotalParams <- nParams + nTheta
  
  inits <- matrix(0, nrow = nrow(full_grid), ncol = nTotalParams)
  
  current_col <- 1
  for(p in param_names) {
    def <- param_defs[[p]]
    nBeta <- ncol(X_list[[p]])
    
    vals <- full_grid[[p]]
    
    if(!is.null(def$link_fun)) {
       vals <- def$link_fun(vals)
    }
    
    inits[, current_col] <- vals
    current_col <- current_col + nBeta
  }
  
  if(nTheta > 0) {
    tau_vals <- full_grid$tau
    step_val <- log(tau_vals / (nRatings - 1))
    
    inits[, current_col:nTotalParams] <- step_val
  }
  
  return(list(
    inits = inits,
    X_list = X_list,
    stim_vec = stim_vec,
    rating_vec = rating_vec,
    correct_vec = correct_vec,
    counts_vec = counts_vec,
    nRatings = nRatings,
    nTrials = nTrials_total,
    nUniqueTrials = nTrials_unique,
    nParams = nParams,
    nTotalParams = nTotalParams,
    param_defs = param_defs
  ))
}


#' Process Regression Results
#'
#' @param fit_res Result object from C++ optimization
#' @param prep Prepared input object
#' @param param_defs Parameter definitions
#' @return List of processed results
process_regression_results <- function(fit_res, prep, param_defs) {
  if(fit_res$error) stop("Optimization failed.")
  
  par <- fit_res$par
  negLogLik <- fit_res$value
  k <- length(par)
  nTrials <- prep$nTrials
  
  res <- data.frame(matrix(nrow=1, ncol=0))
  
  current_idx <- 1
  
  param_names <- names(param_defs)
  
  for(p in param_names) {
    def <- param_defs[[p]]
    X <- prep$X_list[[p]]
    nBeta <- ncol(X)
    
    raw_beta <- par[current_idx:(current_idx + nBeta - 1)]
    current_idx <- current_idx + nBeta
    
    if(!is.null(def$inv_link_fun)) {
      val_natural <- def$inv_link_fun(raw_beta)
    } else {
      val_natural <- raw_beta
    }
    
    clean_names <- colnames(X)
    clean_names <- gsub("\\(Intercept\\)", "Intercept", clean_names)
    
    col_names <- paste0(p, "_", clean_names)
    res[col_names] <- as.list(val_natural)
  }
  
  theta_raw <- par[current_idx:length(par)]
  
  nRatings <- prep$nRatings
  steps_A <- exp(theta_raw[1:(nRatings - 1)])
  cum_steps_A <- cumsum(steps_A)
  theta_minus <- sort(-cum_steps_A) 
  
  steps_B <- exp(theta_raw[(nRatings):(2*(nRatings - 1))])
  theta_plus  <-  cumsum(steps_B)
  
  res[,paste("delta_theta_minus.",(nRatings-1):1, sep="")] <- theta_minus
  res[,paste("delta_theta_plus.",1:(nRatings-1), sep="")] <- theta_plus
  
  res$negLogLik <- negLogLik
  res$N <- nTrials
  res$k <- k
  res$BIC <- 2 * negLogLik + k * log(nTrials)
  res$AICc <- 2 * negLogLik + k * 2 + 2*k*(k-1)/(nTrials-k-1)
  res$AIC <- 2 * negLogLik + 2 * k
  
  return(res)
}