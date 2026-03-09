# Tests for ll_SDT_cpp vs. R reference (llSDT)
# SDT parameter vector: [log_sens x nCond | log_crit_A x nRatings-1 | theta | log_crit_B x nRatings-1]

generate_sdt_inputs <- function(nCond = 2, nRatings = 4, seed = 123) {
  set.seed(seed)
  p <- numeric(nCond + 2 * nRatings - 1)
  p[1:nCond] <- log(runif(nCond, 0.5, 1.5))
  if (nRatings > 1) {
    p[(nCond + 1):(nCond + nRatings - 1)] <- log(runif(nRatings - 1, 0.2, 0.8))
  }
  p[nCond + nRatings] <- rnorm(1, 0, 0.3)
  if (nRatings > 1) {
    p[(nCond + nRatings + 1):(nCond + 2 * nRatings - 1)] <- log(runif(nRatings - 1, 0.2, 0.8))
  }
  c(list(p = p, nRatings = nRatings, nCond = nCond), generate_count_matrices(nCond, nRatings))
}

sdt_edge_cases <- list(
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
    if (inputs$nRatings > 1) {
      inputs$p[(inputs$nCond + 1):(inputs$nCond + inputs$nRatings - 1)] <- log(rep(1.5, inputs$nRatings - 1))
      inputs$p[(inputs$nCond + inputs$nRatings + 1):(inputs$nCond + 2 * inputs$nRatings - 1)] <- log(rep(1.5, inputs$nRatings - 1))
    }
    inputs
  },
  narrow_criteria = function(inputs) {
    if (inputs$nRatings > 1) {
      inputs$p[(inputs$nCond + 1):(inputs$nCond + inputs$nRatings - 1)] <- log(rep(0.1, inputs$nRatings - 1))
      inputs$p[(inputs$nCond + inputs$nRatings + 1):(inputs$nCond + 2 * inputs$nRatings - 1)] <- log(rep(0.1, inputs$nRatings - 1))
    }
    inputs
  }
)

test_that("ll_SDT_cpp matches R across configurations", {
  configs <- list(
    list(nCond = 2, nRatings = 4, seed = 123),
    list(nCond = 1, nRatings = 4, seed = 456),
    list(nCond = 2, nRatings = 2, seed = 789),
    list(nCond = 2, nRatings = 3, seed = 321),
    list(nCond = 5, nRatings = 6, seed = 101),
    list(nCond = 3, nRatings = 4, seed = 42)
  )
  for (cfg in configs) {
    inputs <- generate_sdt_inputs(cfg$nCond, cfg$nRatings, cfg$seed)
    result <- compare_likelihood_generic("SDT", llSDT, inputs)
    expect_true(result$match,
      info = sprintf("nCond=%d nRatings=%d seed=%d: R=%f C++=%f diff=%e",
                     cfg$nCond, cfg$nRatings, cfg$seed,
                     result$r_result, result$cpp_result, result$difference))
  }
})

test_that("ll_SDT_cpp handles edge cases", {
  base <- generate_sdt_inputs(nCond = 2, nRatings = 4, seed = 123)
  for (nm in names(sdt_edge_cases)) {
    inputs <- sdt_edge_cases[[nm]](base)
    result <- compare_likelihood_generic("SDT", llSDT, inputs, tolerance = 1e-8)
    expect_true(is.finite(result$cpp_result), info = sprintf("'%s': result not finite", nm))
    expect_true(result$match,
      info = sprintf("'%s': R=%f C++=%f diff=%e", nm, result$r_result, result$cpp_result, result$difference))
  }
})

test_that("ll_SDT_cpp output is finite, non-negative, and deterministic", {
  inputs <- generate_sdt_inputs(nCond = 2, nRatings = 4, seed = 123)
  call_cpp <- function(inp) {
    ptr <- get_sdt_ptr()
    test_ll_ptr(ptr, inp$p, inp$N_SA_RA, inp$N_SA_RB, inp$N_SB_RA, inp$N_SB_RB, inp$nRatings, inp$nCond)
  }
  r1 <- call_cpp(inputs)
  expect_true(is.finite(r1))
  expect_gte(r1, 0)
  expect_identical(r1, call_cpp(inputs))
})

test_that("ll_SDT_cpp neglogL scales proportionally with data size", {
  inputs <- generate_sdt_inputs(nCond = 2, nRatings = 4, seed = 123)
  call_cpp <- function(inp) {
    ptr <- get_sdt_ptr()
    test_ll_ptr(ptr, inp$p, inp$N_SA_RA, inp$N_SA_RB, inp$N_SB_RA, inp$N_SB_RB, inp$nRatings, inp$nCond)
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

test_that("ll_SDT_cpp matches R across parameter grid", {
  for (nCond in 1:3) {
    for (nRatings in 2:5) {
      for (seed in c(111, 222, 333)) {
        inputs <- generate_sdt_inputs(nCond, nRatings, seed)
        result <- compare_likelihood_generic("SDT", llSDT, inputs)
        expect_true(result$match,
          info = sprintf("nCond=%d, nRatings=%d, seed=%d: diff=%e",
                         nCond, nRatings, seed, result$difference))
      }
    }
  }
})

test_that("ll_SDT_cpp is faster than R version", {
  skip_on_cran()
  skip_on_ci()
  inputs <- generate_sdt_inputs(nCond = 3, nRatings = 5, seed = 123)
  n_iter <- 100
  r_time   <- system.time(for (i in seq_len(n_iter)) llSDT(inputs$p, inputs$N_SA_RA, inputs$N_SA_RB, inputs$N_SB_RA, inputs$N_SB_RB, inputs$nRatings, inputs$nCond))["elapsed"]
  cpp_time <- system.time(for (i in seq_len(n_iter)) {
    ptr <- get_sdt_ptr()
    test_ll_ptr(ptr, inputs$p, inputs$N_SA_RA, inputs$N_SA_RB, inputs$N_SB_RA, inputs$N_SB_RB, inputs$nRatings, inputs$nCond)
  })["elapsed"]
  message(sprintf("ll_SDT - R: %.3fs, C++: %.3fs, speedup: %.1fx", r_time, cpp_time, r_time / cpp_time))
  expect_true(cpp_time < r_time, info = sprintf("R: %.3fs, C++: %.3fs", r_time, cpp_time))
})
