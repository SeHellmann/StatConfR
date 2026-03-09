# Tests for ll_RCE_cpp vs. R reference (ll_RCE)

generate_rce_inputs <- function(nCond = 2, nRatings = 4, seed = 123) {
  set.seed(seed)
  p <- numeric(nCond + 2 * nRatings - 1)
  p[1:nCond] <- log(runif(nCond, 0.5, 1.5))

  theta <- rnorm(1, 0, 0.3)
  anchor_ra <- theta - abs(rnorm(1, 0.5, 0.2))  # ensure anchor_ra < theta
  anchor_rb <- theta + abs(rnorm(1, 0.5, 0.2))  # ensure anchor_rb > theta

  if (nRatings > 2) {
    mid_A <- log(runif(nRatings - 2, 0.2, 0.8))
    mid_B <- log(runif(nRatings - 2, 0.2, 0.8))
    p[(nCond + 1):(nCond + nRatings - 2)] <- mid_A
  }
  p[nCond + nRatings - 1] <- anchor_ra
  p[nCond + nRatings]     <- theta
  p[nCond + nRatings + 1] <- anchor_rb
  if (nRatings > 2) {
    p[(nCond + nRatings + 2):(nCond + 2 * nRatings - 1)] <- mid_B
  }
  c(list(p = p, nRatings = nRatings, nCond = nCond), generate_count_matrices(nCond, nRatings))
}


rce_edge_cases <- list(
  theta_positive = function(inputs) {
    inputs$p[inputs$nCond + inputs$nRatings] <- 1.5
    inputs$p[inputs$nCond + inputs$nRatings - 1] <- 1.0
    inputs$p[inputs$nCond + inputs$nRatings + 1] <- 2.0
    inputs
  },
  theta_negative = function(inputs) {
    inputs$p[inputs$nCond + inputs$nRatings] <- -1.5
    inputs$p[inputs$nCond + inputs$nRatings - 1] <- -2.0
    inputs$p[inputs$nCond + inputs$nRatings + 1] <- -1.0
    inputs
  },
  theta_zero = function(inputs) {
    inputs$p[inputs$nCond + inputs$nRatings] <- 0
    inputs$p[inputs$nCond + inputs$nRatings - 1] <- -0.5
    inputs$p[inputs$nCond + inputs$nRatings + 1] <- 0.5
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
  },
  wide_criteria = function(inputs) {
    nR <- inputs$nRatings
    nC <- inputs$nCond
    if (nR > 2) {
      inputs$p[(nC + 1):(nC + nR - 2)] <- log(rep(1.5, nR - 2))
      inputs$p[(nC + nR + 2):(nC + 2 * nR - 1)] <- log(rep(1.5, nR - 2))
    }
    inputs
  },
  narrow_criteria = function(inputs) {
    nR <- inputs$nRatings
    nC <- inputs$nCond
    if (nR > 2) {
      inputs$p[(nC + 1):(nC + nR - 2)] <- log(rep(0.1, nR - 2))
      inputs$p[(nC + nR + 2):(nC + 2 * nR - 1)] <- log(rep(0.1, nR - 2))
    }
    inputs
  }
)


test_that("ll_RCE_cpp matches R across configurations", {
  configs <- list(
    list(nCond = 2, nRatings = 4, seed = 123),
    list(nCond = 1, nRatings = 4, seed = 456),
    list(nCond = 2, nRatings = 3, seed = 321),
    list(nCond = 5, nRatings = 6, seed = 101),
    list(nCond = 3, nRatings = 4, seed = 42)
  )
  for (cfg in configs) {
    inputs <- generate_rce_inputs(cfg$nCond, cfg$nRatings, cfg$seed)
    result <- compare_likelihood_generic("RCE", ll_RCE, inputs, tolerance = 1e-4)
    expect_true(result$match,
      info = sprintf("nCond=%d nRatings=%d seed=%d: R=%f C++=%f diff=%e",
                     cfg$nCond, cfg$nRatings, cfg$seed,
                     result$r_result, result$cpp_result, result$difference))
  }
})

test_that("ll_RCE_cpp handles edge cases", {
  base <- generate_rce_inputs(nCond = 2, nRatings = 4, seed = 123)
  for (nm in names(rce_edge_cases)) {
    inputs <- rce_edge_cases[[nm]](base)
    result <- compare_likelihood_generic("RCE", ll_RCE, inputs, tolerance = 1e-4)
    expect_true(is.finite(result$cpp_result), info = sprintf("'%s': result not finite", nm))
    expect_true(result$match,
      info = sprintf("'%s': R=%f C++=%f diff=%e", nm, result$r_result, result$cpp_result, result$difference))
  }
})

test_that("ll_RCE_cpp output is finite, non-negative, and deterministic", {
  inputs <- generate_rce_inputs(nCond = 2, nRatings = 4, seed = 123)
  call_cpp <- function(inp) {
    ptr <- get_rce_ptr()
    test_ll_ptr(ptr, inp$p, inp$N_SA_RA, inp$N_SA_RB, inp$N_SB_RA, inp$N_SB_RB, inp$nRatings, inp$nCond)
  }
  r1 <- call_cpp(inputs)
  expect_true(is.finite(r1))
  expect_gte(r1, 0)
  expect_identical(r1, call_cpp(inputs))
})

test_that("ll_RCE_cpp matches R across parameter grid", {
  for (nCond in 1:3) {
    for (nRatings in 3:5) {
      for (seed in c(111, 222, 333)) {
        inputs <- generate_rce_inputs(nCond, nRatings, seed)
        result <- compare_likelihood_generic("RCE", ll_RCE, inputs, tolerance = 1e-3)
        expect_true(result$match,
          info = sprintf("nCond=%d, nRatings=%d, seed=%d: diff=%e",
                         nCond, nRatings, seed, result$difference))
      }
    }
  }
})

test_that("ll_RCE_cpp is faster than R version", {
  skip_on_cran()
  skip_on_ci()
  inputs <- generate_rce_inputs(nCond = 3, nRatings = 5, seed = 123)
  n_iter <- 50
  r_time   <- system.time(for (i in seq_len(n_iter)) ll_RCE(inputs$p, inputs$N_SA_RA, inputs$N_SA_RB, inputs$N_SB_RA, inputs$N_SB_RB, inputs$nRatings, inputs$nCond))["elapsed"]
  cpp_time <- system.time(for (i in seq_len(n_iter)) {
    ptr <- get_rce_ptr()
    test_ll_ptr(ptr, inputs$p, inputs$N_SA_RA, inputs$N_SA_RB, inputs$N_SB_RA, inputs$N_SB_RB, inputs$nRatings, inputs$nCond)
  })["elapsed"]
  message(sprintf("ll_RCE - R: %.3fs, C++: %.3fs, speedup: %.1fx", r_time, cpp_time, r_time / cpp_time))
  expect_true(cpp_time < r_time, info = sprintf("R: %.3fs, C++: %.3fs", r_time, cpp_time))
})
