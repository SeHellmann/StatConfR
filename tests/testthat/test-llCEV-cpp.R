# Tests for ll_CEV_cpp vs. R reference (ll_CEV)
# CEV parameter vector: [log_sens x nCond | log_crit_A x nRatings-1 | theta | log_crit_B x nRatings-1 |
#                        log_sigma | logit_w]

generate_cev_inputs <- function(nCond = 2, nRatings = 4, seed = 123) {
  set.seed(seed)
  p <- numeric(nCond + 2 * nRatings + 1)
  p[1:nCond] <- log(runif(nCond, 0.5, 1.5))
  if (nRatings > 1) {
    p[(nCond + 1):(nCond + nRatings - 1)] <- log(runif(nRatings - 1, 0.2, 0.8))
  }
  p[nCond + nRatings] <- rnorm(1, 0, 0.3)
  if (nRatings > 1) {
    p[(nCond + nRatings + 1):(nCond + 2 * nRatings - 1)] <- log(runif(nRatings - 1, 0.2, 0.8))
  }
  p[nCond + 2 * nRatings]     <- log(runif(1, 0.3, 1.5))
  p[nCond + 2 * nRatings + 1] <- rnorm(1, 0, 1)
  c(list(p = p, nRatings = nRatings, nCond = nCond), generate_count_matrices(nCond, nRatings))
}

cev_edge_cases <- list(
  theta_positive = function(inputs) {
    inputs$p[inputs$nCond + inputs$nRatings] <- 1.5
    inputs
  },
  theta_negative = function(inputs) {
    inputs$p[inputs$nCond + inputs$nRatings] <- -1.5
    inputs
  },
  theta_zero = function(inputs) {
    inputs$p[inputs$nCond + inputs$nRatings] <- 0
    inputs
  },
  sigma_small = function(inputs) {
    inputs$p[inputs$nCond + 2 * inputs$nRatings] <- log(0.1)
    inputs
  },
  sigma_large = function(inputs) {
    inputs$p[inputs$nCond + 2 * inputs$nRatings] <- log(2.0)
    inputs
  },
  w_near_zero = function(inputs) {
    inputs$p[inputs$nCond + 2 * inputs$nRatings + 1] <- -3
    inputs
  },
  w_near_one = function(inputs) {
    inputs$p[inputs$nCond + 2 * inputs$nRatings + 1] <- 3
    inputs
  },
  w_half = function(inputs) {
    inputs$p[inputs$nCond + 2 * inputs$nRatings + 1] <- 0
    inputs
  },
  sparse_counts = function(inputs) {
    inputs$N_SA_RA[1, 1] <- 0
    inputs$N_SB_RB[inputs$nCond, inputs$nRatings] <- 0
    inputs
  },
  zero_matrix = function(inputs) {
    inputs$N_SA_RB <- matrix(0, nrow = inputs$nCond, ncol = inputs$nRatings)
    inputs
  },
  high_sensitivity = function(inputs) {
    inputs$p[1:inputs$nCond] <- log(rep(2, inputs$nCond))
    inputs
  },
  low_sensitivity = function(inputs) {
    inputs$p[1:inputs$nCond] <- log(rep(0.1, inputs$nCond))
    inputs
  }
)

test_that("ll_CEV_cpp matches R across configurations", {
  configs <- list(
    list(nCond = 2, nRatings = 4, seed = 123),
    list(nCond = 1, nRatings = 4, seed = 456),
    list(nCond = 2, nRatings = 2, seed = 789),
    list(nCond = 2, nRatings = 3, seed = 321),
    list(nCond = 5, nRatings = 6, seed = 101, tol = 1e-3),
    list(nCond = 3, nRatings = 4, seed = 42)
  )
  for (cfg in configs) {
    inputs <- generate_cev_inputs(cfg$nCond, cfg$nRatings, cfg$seed)
    tol <- if (!is.null(cfg$tol)) cfg$tol else 1e-4
    result <- compare_likelihood_generic("CEV", ll_CEV, inputs, tolerance = tol)
    expect_true(result$match,
      info = sprintf("nCond=%d nRatings=%d seed=%d: R=%f C++=%f diff=%e",
                     cfg$nCond, cfg$nRatings, cfg$seed,
                     result$r_result, result$cpp_result, result$difference))
  }
})

test_that("ll_CEV_cpp handles edge cases", {
  base <- generate_cev_inputs(nCond = 2, nRatings = 4, seed = 123)
  for (nm in names(cev_edge_cases)) {
    inputs <- cev_edge_cases[[nm]](base)
    result <- compare_likelihood_generic("CEV", ll_CEV, inputs, tolerance = 1e-4)
    expect_true(is.finite(result$cpp_result), info = sprintf("'%s': result not finite", nm))
    expect_true(result$match,
      info = sprintf("'%s': R=%f C++=%f diff=%e", nm, result$r_result, result$cpp_result, result$difference))
  }
})

test_that("ll_CEV_cpp output is finite, non-negative, and deterministic", {
  inputs <- generate_cev_inputs(nCond = 2, nRatings = 4, seed = 123)
  call_cpp <- function(inp) {
    ptr <- get_cev_ptr()
    test_ll_ptr(ptr, inp$p, inp$N_SA_RA, inp$N_SA_RB, inp$N_SB_RA, inp$N_SB_RB, inp$nRatings, inp$nCond)
  }
  r1 <- call_cpp(inputs)
  expect_true(is.finite(r1))
  expect_gte(r1, 0)
  expect_identical(r1, call_cpp(inputs))
})

test_that("ll_CEV_cpp matches R across parameter grid", {
  for (nCond in 1:3) {
    for (nRatings in 2:5) {
      for (seed in c(111, 222, 333)) {
        inputs <- generate_cev_inputs(nCond, nRatings, seed)
        result <- compare_likelihood_generic("CEV", ll_CEV, inputs, tolerance = 1e-3)
        expect_true(result$match,
          info = sprintf("nCond=%d, nRatings=%d, seed=%d: diff=%e",
                         nCond, nRatings, seed, result$difference))
      }
    }
  }
})

test_that("ll_CEV_cpp is faster than R version", {
  skip_on_cran()
  skip_on_ci()
  inputs <- generate_cev_inputs(nCond = 3, nRatings = 5, seed = 123)
  n_iter <- 50
  r_time   <- system.time(for (i in seq_len(n_iter)) ll_CEV(inputs$p, inputs$N_SA_RA, inputs$N_SA_RB, inputs$N_SB_RA, inputs$N_SB_RB, inputs$nRatings, inputs$nCond))["elapsed"]
  cpp_time <- system.time(for (i in seq_len(n_iter)) {
    ptr <- get_cev_ptr()
    test_ll_ptr(ptr, inputs$p, inputs$N_SA_RA, inputs$N_SA_RB, inputs$N_SB_RA, inputs$N_SB_RB, inputs$nRatings, inputs$nCond)
  })["elapsed"]
  message(sprintf("ll_CEV - R: %.3fs, C++: %.3fs, speedup: %.1fx", r_time, cpp_time, r_time / cpp_time))
  expect_true(cpp_time < r_time, info = sprintf("R: %.3fs, C++: %.3fs", r_time, cpp_time))
})
