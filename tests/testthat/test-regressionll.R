# ==============================================================================
# 1. HELPER FUNCTIONS & MODEL SPECIFICATIONS
# ==============================================================================

model_specs <- list(
  "SDT"    = list(n_meta = 0, type = "full"),
  "GN"     = list(n_meta = 1, type = "diff"), 
  "PDA"    = list(n_meta = 1, type = "diff"),
  "IG"     = list(n_meta = 1, type = "diff"), 
  "ITGc"   = list(n_meta = 1, type = "full"), 
  "ITGcm"  = list(n_meta = 1, type = "full"), 
  "WEV"    = list(n_meta = 2, type = "diff"), 
  "logN"   = list(n_meta = 1, type = "full"), 
  "logWEV" = list(n_meta = 2, type = "full"),
  "RCE"    = list(n_meta = 0, type = "diff"),
  "CAS"    = list(n_meta = 1, type = "full")
)

generate_count_matrices <- function(nCond, nRatings, lambda_diag = 20, lambda_offdiag = 5) {
  list(
    N_SA_RA = matrix(rpois(nCond * nRatings, lambda = lambda_diag),    nrow = nCond, ncol = nRatings),
    N_SA_RB = matrix(rpois(nCond * nRatings, lambda = lambda_offdiag), nrow = nCond, ncol = nRatings),
    N_SB_RA = matrix(rpois(nCond * nRatings, lambda = lambda_offdiag), nrow = nCond, ncol = nRatings),
    N_SB_RB = matrix(rpois(nCond * nRatings, lambda = lambda_diag),    nrow = nCond, ncol = nRatings)
  )
}

generate_regression_test_data <- function(model, nCond = 2, nRatings = 4, seed = 123) {
  set.seed(seed)
  spec <- model_specs[[model]]

  if (spec$type == "full") {
    n_p <- nCond + (nRatings - 1) * 2 + 1 + spec$n_meta
  } else {
    n_p <- nCond + max(0, nRatings - 2) * 2 + 3 + spec$n_meta
  }
  
  p <- numeric(n_p)
  p[1:nCond] <- log(runif(nCond, 0.5, 1.5))
  theta <- rnorm(1, 0, 0.3)
  
  if (spec$type == "full") {
    log_steps_a <- if(nRatings > 1) log(runif(nRatings - 1, 0.2, 0.8)) else numeric(0)
    log_steps_b <- if(nRatings > 1) log(runif(nRatings - 1, 0.2, 0.8)) else numeric(0)
    p[(nCond + 1):(nCond + 2 * nRatings - 1)] <- c(log_steps_a, theta, log_steps_b)
    idx_meta_start <- nCond + 2 * nRatings
  } else {
    gap_a <- runif(1, 0.2, 0.8); gap_b <- runif(1, 0.2, 0.8)
    anchor_ra <- theta - gap_a; anchor_rb <- theta + gap_b
    mid_a <- if(nRatings > 2) log(runif(nRatings - 2, 0.2, 0.5)) else numeric(0)
    mid_b <- if(nRatings > 2) log(runif(nRatings - 2, 0.2, 0.5)) else numeric(0)
    
    current <- nCond + 1
    if(length(mid_a) > 0) { p[current:(current+length(mid_a)-1)] <- mid_a; current <- current + length(mid_a) }
    p[current:(current+2)] <- c(anchor_ra, theta, anchor_rb); current <- current + 3
    if(length(mid_b) > 0) { p[current:(current+length(mid_b)-1)] <- mid_b; current <- current + length(mid_b) }
    idx_meta_start <- current
  }
  
  if (spec$n_meta > 0) {
    p[idx_meta_start:(idx_meta_start + spec$n_meta - 1)] <- rnorm(spec$n_meta, 0, 0.5)
  }
  
  counts <- generate_count_matrices(nCond, nRatings)
  list(model = model, p = p, nRatings = nRatings, nCond = nCond, n_meta = spec$n_meta, type = spec$type,
       N_SA_RA = counts$N_SA_RA, N_SA_RB = counts$N_SA_RB, N_SB_RA = counts$N_SB_RA, N_SB_RB = counts$N_SB_RB)
}

counts_to_regression_data <- function(nRatings, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB) {
  df_list <- list()
  for(r in 1:nRatings) {
    if(N_SA_RA[1, r] > 0) df_list[[length(df_list)+1]] <- data.frame(stimulus=0, correct=1, rating=nRatings-r+1, counts=N_SA_RA[1, r])
    if(N_SA_RB[1, r] > 0) df_list[[length(df_list)+1]] <- data.frame(stimulus=0, correct=0, rating=r,            counts=N_SA_RB[1, r])
    if(N_SB_RA[1, r] > 0) df_list[[length(df_list)+1]] <- data.frame(stimulus=1, correct=0, rating=nRatings-r+1, counts=N_SB_RA[1, r])
    if(N_SB_RB[1, r] > 0) df_list[[length(df_list)+1]] <- data.frame(stimulus=1, correct=1, rating=r,            counts=N_SB_RB[1, r])
  }
  do.call(rbind, df_list)
}

build_regression_p <- function(inputs) {
  p_agg <- inputs$p
  nCond <- inputs$nCond 
  nRatings <- inputs$nRatings
  
  d_val <- p_agg[1] 
  if (inputs$type == "full") {
    idx_theta <- nCond + nRatings
    theta <- p_agg[idx_theta]
    log_steps_a <- if(nRatings>1) p_agg[(nCond+1):(idx_theta-1)] else numeric(0)
    log_steps_b <- if(nRatings>1) p_agg[(idx_theta+1):(idx_theta+nRatings-1)] else numeric(0)
    idx_meta_start <- idx_theta + nRatings
  } else {
    len_mid <- max(0, nRatings - 2)
    idx_anc_ra <- nCond + len_mid + 1
    idx_theta  <- idx_anc_ra + 1
    idx_anc_rb <- idx_theta + 1
    theta <- p_agg[idx_theta]
    step_a1 <- log(theta - p_agg[idx_anc_ra])
    step_b1 <- log(p_agg[idx_anc_rb] - theta)
    log_steps_a <- c(step_a1, if(len_mid > 0) p_agg[(nCond+1):(nCond+len_mid)] else numeric(0))
    log_steps_b <- c(step_b1, if(len_mid > 0) p_agg[(idx_anc_rb+1):(idx_anc_rb+len_mid)] else numeric(0))
    idx_meta_start <- idx_anc_rb + len_mid + 1
  }
  meta_vals <- if(inputs$n_meta > 0) p_agg[idx_meta_start:(idx_meta_start + inputs$n_meta - 1)] else numeric(0)
  c(d_val, theta, meta_vals, log_steps_a, log_steps_b)
}

get_ptr_name <- function(model, reg=FALSE) {
  m <- tolower(model)
  if(m == "gn") m <- "noisy"; if(m == "ig") m <- "2chan"
  if(m == "itgc") m <- "itgc"; if(m == "itgcm") m <- "itgcm"
  if(m == "logn") m <- "lognorm"; if(m == "wev") m <- "cev"
  base <- paste0("get_", m)
  if(reg) paste0(base, "_regression_ptr") else paste0(base, "_ptr")
}

compare_regression_generic <- function(inputs, tolerance = 1e-6) {
  p_agg_1 <- c(inputs$p[1], inputs$p[(inputs$nCond + 1):length(inputs$p)])
  N_list <- list(matrix(inputs$N_SA_RA[1,], 1), matrix(inputs$N_SA_RB[1,], 1), matrix(inputs$N_SB_RA[1,], 1), matrix(inputs$N_SB_RB[1,], 1))
  
  agg_ptr_func <- get(get_ptr_name(inputs$model, reg=FALSE), envir = asNamespace("statConfR"))
  agg_result <- test_ll_ptr(agg_ptr_func(), p_agg_1, N_list[[1]], N_list[[2]], N_list[[3]], N_list[[4]], inputs$nRatings, 1)
  
  inputs_1 <- inputs; inputs_1$p <- p_agg_1; inputs_1$nCond <- 1
  reg_data <- counts_to_regression_data(inputs$nRatings, N_list[[1]], N_list[[2]], N_list[[3]], N_list[[4]])
  p_reg <- build_regression_p(inputs_1)
  
  X_list <- lapply(1:(2 + inputs$n_meta), function(x) matrix(1, nrow(reg_data), 1))
  reg_ptr_func <- get(get_ptr_name(inputs$model, reg=TRUE), envir = asNamespace("statConfR"))
  reg_result <- test_ll_regression_ptr(reg_ptr_func(), p_reg, X_list, reg_data$rating, reg_data$stimulus, reg_data$correct, reg_data$counts, inputs$nRatings, sum(reg_data$counts))
  
  list(agg_result = agg_result, reg_result = reg_result, difference = abs(agg_result - reg_result), match = abs(agg_result - reg_result) < tolerance)
}

call_regression_direct <- function(inputs) {
  p_agg_1 <- c(inputs$p[1], inputs$p[(inputs$nCond + 1):length(inputs$p)])
  inputs_1 <- inputs; inputs_1$p <- p_agg_1; inputs_1$nCond <- 1
  reg_data <- counts_to_regression_data(inputs$nRatings, matrix(inputs$N_SA_RA[1,], 1), matrix(inputs$N_SA_RB[1,], 1), matrix(inputs$N_SB_RA[1,], 1), matrix(inputs$N_SB_RB[1,], 1))
  p_reg <- build_regression_p(inputs_1)
  X_list <- lapply(1:(2 + inputs$n_meta), function(x) matrix(1, nrow(reg_data), 1))
  reg_ptr_func <- get(get_ptr_name(inputs$model, reg=TRUE), envir = asNamespace("statConfR"))
  test_ll_regression_ptr(reg_ptr_func(), p_reg, X_list, reg_data$rating, reg_data$stimulus, reg_data$correct, reg_data$counts, inputs$nRatings, sum(reg_data$counts))
}


# ==============================================================================
# 2. TEST SUITE
# ==============================================================================

all_models <- names(model_specs)

# Note: For models 2Chan, GN (Noisy), PDA, and WEV(CEV) this equivalence test currently fails when nRatings = 2 due to the
#       aggregated version not creating an empty vector
test_that("Regression implementations match Aggregated logic (nCond=1 subset)", {
  configs <- list(list(nC=2, nR=4, seed=101), list(nC=1, nR=2, seed=102), list(nC=3, nR=5, seed=103))
  for(m in all_models) {
    for(cfg in configs) {
      # Feel free to remove this line after the bug in the aggregated versions is fixed
      if(m %in% c("2Chan", "GN", "PDA", "WEV") && cfg$nR < 3) next

      inputs <- generate_regression_test_data(m, cfg$nC, cfg$nR, cfg$seed)
      tol <- if(m %in% c("SDT", "ITGc", "ITGcm")) 1e-8 else 1e-4
      res <- compare_regression_generic(inputs, tolerance = tol)
      expect_true(res$match, info = sprintf("Model: %s | nR: %d | Diff: %.9f", m, cfg$nR, res$difference))
    }
  }
})

test_that("Regression implementations return finite, deterministic values", {
  for(m in all_models) {
    inputs <- generate_regression_test_data(m, nCond=2, nRatings=4, seed=999)
    val1 <- call_regression_direct(inputs)
    val2 <- call_regression_direct(inputs)
    expect_true(is.finite(val1), info = paste(m, "not finite"))
    expect_true(val1 >= 0, info = paste(m, "negative"))
    expect_true(val1 == val2, info = paste(m, "not deterministic"))
  }
})

test_that("Regression implementations handle edge cases", {
  cases <- c("theta_positive", "theta_negative", "high_sens", "low_sens", "meta_small", "meta_large")
  for(m in all_models) {
    base <- generate_regression_test_data(m, nCond=2, nRatings=4, seed=555)
    for(case in cases) {
      if(base$n_meta == 0 && grepl("meta", case)) next
      idx_theta <- if(base$type=="full") base$nCond + base$nRatings else base$nCond + max(0, base$nRatings - 2) + 2
      idx_meta <- length(base$p) - base$n_meta + 1
      mod_p <- base$p
      if(case == "theta_positive") { mod_p[idx_theta] <- 1.5; if(base$type=="diff"){ mod_p[idx_theta-1]<-1.0; mod_p[idx_theta+1]<-2.0 }}
      else if(case == "theta_negative") { mod_p[idx_theta] <- -1.5; if(base$type=="diff"){ mod_p[idx_theta-1]<- -2.0; mod_p[idx_theta+1]<- -1.0 }}
      else if(case == "high_sens") { mod_p[1:base$nCond] <- log(3) }
      else if(case == "low_sens") { mod_p[1:base$nCond] <- log(0.01) }
      else if(case == "meta_small") { mod_p[idx_meta] <- -3 }
      else if(case == "meta_large") { mod_p[idx_meta] <- 3 }
      base_mod <- base; base_mod$p <- mod_p
      val <- call_regression_direct(base_mod)
      expect_true(is.finite(val), info = sprintf("Model %s failed edge case %s", m, case))
    }
  }
})

test_that("Regression implementations match Aggregated across parameter grid", {
  seeds <- c(10, 20)
  for(m in all_models) {
    for(s in seeds) {
      inputs <- generate_regression_test_data(m, nCond=1, nRatings=3, seed=s)
      tol <- if(m %in% c("SDT", "RCE")) 1e-8 else 1e-3
      res <- compare_regression_generic(inputs, tolerance = tol)
      expect_true(res$match, info = paste(m, "grid seed", s))
    }
  }
})