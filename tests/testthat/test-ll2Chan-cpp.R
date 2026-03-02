# Tests for ll_2Chan_cpp vs. R reference (ll2Chan)
# 2Chan parameter vector: [log_sens x nCond | log_crit_A x nRatings-2 | theta_prev | theta | theta_next |
#                          log_crit_B x nRatings-2 | log_mratio]
# R indexing bug skip nRatings=2 for comparison

generate_2chan_inputs <- function(nCond = 2, nRatings = 4, seed = 123) {
  set.seed(seed)
  p <- numeric(nCond + 2 * nRatings)
  p[1:nCond] <- log(runif(nCond, 0.5, 2.0))
  if (nRatings > 2) {
    p[(nCond + 1):(nCond + nRatings - 2)] <- log(runif(nRatings - 2, 0.2, 0.8))
  }
  theta      <- rnorm(1, 0, 0.3)
  theta_prev <- theta - abs(rnorm(1, 0.5, 0.2))
  theta_next <- theta + abs(rnorm(1, 0.5, 0.2))
  p[nCond + nRatings - 1] <- theta_prev
  p[nCond + nRatings]     <- theta
  p[nCond + nRatings + 1] <- theta_next
  if (nRatings > 2) {
    p[(nCond + nRatings + 2):(nCond + 2 * nRatings - 1)] <- log(runif(nRatings - 2, 0.2, 0.8))
  }
  p[nCond + 2 * nRatings] <- log(runif(1, 0.5, 1.5))
  c(list(p = p, nRatings = nRatings, nCond = nCond), generate_count_matrices(nCond, nRatings))
}

chan2_edge_cases <- list(
  theta_positive = function(inputs) {
    inputs$p[inputs$nCond + inputs$nRatings]     <- 1.5
    inputs$p[inputs$nCond + inputs$nRatings - 1] <- 1.0
    inputs$p[inputs$nCond + inputs$nRatings + 1] <- 2.0
    inputs
  },
  theta_negative = function(inputs) {
    inputs$p[inputs$nCond + inputs$nRatings]     <- -1.5
    inputs$p[inputs$nCond + inputs$nRatings - 1] <- -2.0
    inputs$p[inputs$nCond + inputs$nRatings + 1] <- -1.0
    inputs
  },
  theta_zero = function(inputs) {
    inputs$p[inputs$nCond + inputs$nRatings]     <- 0
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
    inputs$p[1:inputs$nCond] <- log(rep(3, inputs$nCond))
    inputs
  },
  low_sensitivity = function(inputs) {
    inputs$p[1:inputs$nCond] <- log(rep(0.2, inputs$nCond))
    inputs
  },
  high_mratio = function(inputs) {
    inputs$p[inputs$nCond + 2 * inputs$nRatings] <- log(2.0)
    inputs
  },
  low_mratio = function(inputs) {
    inputs$p[inputs$nCond + 2 * inputs$nRatings] <- log(0.3)
    inputs
  },
  wide_criteria = function(inputs) {
    if (inputs$nRatings > 2) {
      inputs$p[(inputs$nCond + 1):(inputs$nCond + inputs$nRatings - 2)] <- log(rep(1.5, inputs$nRatings - 2))
      inputs$p[(inputs$nCond + inputs$nRatings + 2):(inputs$nCond + 2 * inputs$nRatings - 1)] <- log(rep(1.5, inputs$nRatings - 2))
    }
    inputs
  },
  narrow_criteria = function(inputs) {
    if (inputs$nRatings > 2) {
      inputs$p[(inputs$nCond + 1):(inputs$nCond + inputs$nRatings - 2)] <- log(rep(0.1, inputs$nRatings - 2))
      inputs$p[(inputs$nCond + inputs$nRatings + 2):(inputs$nCond + 2 * inputs$nRatings - 1)] <- log(rep(0.1, inputs$nRatings - 2))
    }
    inputs
  }
)

test_that("ll_2Chan_cpp matches R across configurations", {
  configs <- list(
    list(nCond = 2, nRatings = 4, seed = 123),
    list(nCond = 1, nRatings = 4, seed = 456),
    list(nCond = 2, nRatings = 2, seed = 789),
    list(nCond = 2, nRatings = 3, seed = 321),
    list(nCond = 5, nRatings = 6, seed = 101),
    list(nCond = 3, nRatings = 4, seed = 42)
  )
  for (cfg in configs) {
    if (cfg$nRatings == 2) next  # R indexing bug skip nRatings=2 for comparison
    inputs <- generate_2chan_inputs(cfg$nCond, cfg$nRatings, cfg$seed)
    result <- compare_likelihood_generic(ll2Chan, ll_2Chan_cpp, inputs)
    expect_true(result$match,
      info = sprintf("nCond=%d nRatings=%d seed=%d: R=%f C++=%f diff=%e",
                     cfg$nCond, cfg$nRatings, cfg$seed,
                     result$r_result, result$cpp_result, result$difference))
  }
})

test_that("ll_2Chan_cpp handles edge cases", {
  base <- generate_2chan_inputs(nCond = 2, nRatings = 4, seed = 123)
  for (nm in names(chan2_edge_cases)) {
    inputs <- chan2_edge_cases[[nm]](base)
    result <- compare_likelihood_generic(ll2Chan, ll_2Chan_cpp, inputs, tolerance = 1e-8)
    expect_true(is.finite(result$cpp_result), info = sprintf("'%s': result not finite", nm))
    expect_true(result$match,
      info = sprintf("'%s': R=%f C++=%f diff=%e", nm, result$r_result, result$cpp_result, result$difference))
  }
})

test_that("ll_2Chan_cpp output is finite, non-negative, and deterministic", {
  inputs <- generate_2chan_inputs(nCond = 2, nRatings = 4, seed = 123)
  call_cpp <- function(inp) {
    ll_2Chan_cpp(inp$p, inp$N_SA_RA, inp$N_SA_RB, inp$N_SB_RA, inp$N_SB_RB, inp$nRatings, inp$nCond)
  }
  r1 <- call_cpp(inputs)
  expect_true(is.finite(r1))
  expect_gte(r1, 0)
  expect_identical(r1, call_cpp(inputs))
})

test_that("ll_2Chan_cpp neglogL scales proportionally with data size", {
  inputs <- generate_2chan_inputs(nCond = 2, nRatings = 4, seed = 123)
  call_cpp <- function(inp) {
    ll_2Chan_cpp(inp$p, inp$N_SA_RA, inp$N_SA_RB, inp$N_SB_RA, inp$N_SB_RB, inp$nRatings, inp$nCond)
  }
  r1 <- call_cpp(inputs)
  inputs2 <- inputs
  inputs2[c("N_SA_RA", "N_SA_RB", "N_SB_RA", "N_SB_RB")] <- lapply(
    inputs[c("N_SA_RA", "N_SA_RB", "N_SB_RA", "N_SB_RB")], `*`, 2
  )
  ratio <- call_cpp(inputs2) / r1
  expect_true(ratio > 1.5 && ratio < 2.5,
    info = sprintf("negLogL ratio with doubled counts: %.2f (expected ~2)", ratio))
})

test_that("ll_2Chan_cpp responds to meta-d' ratio changes", {
  inputs <- generate_2chan_inputs(nCond = 2, nRatings = 4, seed = 123)
  call_cpp <- function(inp) {
    ll_2Chan_cpp(inp$p, inp$N_SA_RA, inp$N_SA_RB, inp$N_SB_RA, inp$N_SB_RB, inp$nRatings, inp$nCond)
  }
  r1 <- call_cpp(inputs)
  inputs2 <- inputs
  inputs2$p[inputs$nCond + 2 * inputs$nRatings] <- log(0.5)
  expect_false(r1 == call_cpp(inputs2),
    info = "Likelihood should change when meta-d' ratio changes")
})

test_that("ll_2Chan_cpp matches R across parameter grid", {
  for (nCond in 1:3) {
    for (nRatings in 2:5) {
      if (nRatings == 2) next  # R indexing bug skip nRatings=2 for comparison
      for (seed in c(111, 222, 333)) {
        inputs <- generate_2chan_inputs(nCond, nRatings, seed)
        result <- compare_likelihood_generic(ll2Chan, ll_2Chan_cpp, inputs)
        expect_true(result$match,
          info = sprintf("nCond=%d, nRatings=%d, seed=%d: diff=%e",
                         nCond, nRatings, seed, result$difference))
      }
    }
  }
})

test_that("ll_2Chan_cpp is faster than R version", {
  skip_on_cran()
  skip_on_ci()
  inputs <- generate_2chan_inputs(nCond = 3, nRatings = 5, seed = 123)
  n_iter <- 100
  r_time   <- system.time(for (i in seq_len(n_iter)) ll2Chan(inputs$p, inputs$N_SA_RA, inputs$N_SA_RB, inputs$N_SB_RA, inputs$N_SB_RB, inputs$nRatings, inputs$nCond))["elapsed"]
  cpp_time <- system.time(for (i in seq_len(n_iter)) ll_2Chan_cpp(inputs$p, inputs$N_SA_RA, inputs$N_SA_RB, inputs$N_SB_RA, inputs$N_SB_RB, inputs$nRatings, inputs$nCond))["elapsed"]
  message(sprintf("ll_2Chan - R: %.3fs, C++: %.3fs, speedup: %.1fx", r_time, cpp_time, r_time / cpp_time))
  expect_true(cpp_time < r_time, info = sprintf("R: %.3fs, C++: %.3fs", r_time, cpp_time))
})
