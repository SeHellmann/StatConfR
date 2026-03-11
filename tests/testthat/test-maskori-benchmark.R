# End-to-end benchmark on MaskOri data.
#
# Compares R vs C++ across:
#   - Per-person (fitConf): R serial vs C++ serial
#   - Group (fitConfModels serial): R vs C++
#   - Group (fitConfModels parallel): R vs C++
#
# Also validates output structure and correctness.
#
# Skipped on CRAN and CI

#skip_on_cran()
#skip_on_ci()

data(MaskOri, package = "statConfR")

all_models <- c("WEV", "SDT", "IG", "ITGc", "ITGcm",
                "GN", "PDA", "logN", "logWEV", "RCE", "CAS")

# Shared fitting settings
n_inits   <- 3
n_restart <- 2
n_cores   <- 2

test_participants <- unique(MaskOri$participant)[1:3]
group_data <- subset(MaskOri, participant %in% test_participants)

single_data <- subset(MaskOri, participant == test_participants[1])
nCond <- length(unique(single_data$diffCond))
d_cols <- paste0("d_", seq_len(nCond))
required_cols <- c("negLogLik", "N", "k", "BIC", "AIC", "AICc")

# =============================================================================
# 1. Per-person: R vs C++ serial (fitConf), all models
# =============================================================================

test_that("R and C++ fitConf agree on negLogLik and C++ is faster (all models)", {
  results <- data.frame(model = character(), nll_r = numeric(),
                        nll_cpp = numeric(), rel_diff = numeric(),
                        r_sec = numeric(), cpp_sec = numeric(),
                        speedup = numeric(), stringsAsFactors = FALSE)

  for (mod in all_models) {
    r_time <- system.time(
      r_fit <- fitConf(single_data, model = mod,
                       nInits = n_inits, nRestart = n_restart, fast = FALSE)
    )[["elapsed"]]

    cpp_time <- system.time(
      cpp_fit <- fitConf(single_data, model = mod,
                         nInits = n_inits, nRestart = n_restart, fast = TRUE)
    )[["elapsed"]]

    # --- output structure checks ---
    for (fit_i in list(list(fit = r_fit, label = "R"),
                       list(fit = cpp_fit, label = "C++"))) {
      fit <- fit_i$fit
      label <- fit_i$label

      expect_true(is.data.frame(fit),
        info = sprintf("%s %s: not a data.frame", mod, label))
      expect_equal(nrow(fit), 1L,
        info = sprintf("%s %s: expected 1 row", mod, label))

      for (col in c(required_cols, d_cols, "c")) {
        expect_true(col %in% names(fit),
          info = sprintf("%s %s: missing column '%s'", mod, label, col))
      }

      expect_true(is.finite(fit$negLogLik) && fit$negLogLik > 0,
        info = sprintf("%s %s: negLogLik=%.4f", mod, label, fit$negLogLik))

      ds <- unlist(fit[d_cols])
      expect_true(all(ds > 0),
        info = sprintf("%s %s: not all d > 0", mod, label))
      if (nCond > 1) {
        expect_true(all(diff(ds) >= 0),
          info = sprintf("%s %s: d values not ordered", mod, label))
      }

      expect_equal(fit$N, nrow(single_data),
        info = sprintf("%s %s: N mismatch", mod, label))
      expect_true(is.finite(fit$BIC),
        info = sprintf("%s %s: BIC not finite", mod, label))
      expect_true(is.finite(fit$AIC),
        info = sprintf("%s %s: AIC not finite", mod, label))
    }

    # --- agreement and speed checks ---
    nll_r   <- r_fit$negLogLik
    nll_cpp <- cpp_fit$negLogLik

    rel_diff <- abs(nll_r - nll_cpp) / max(abs(nll_r), 1)
    speedup  <- r_time / max(cpp_time, 1e-6)

    results <- rbind(results,
      data.frame(model = mod, nll_r = nll_r, nll_cpp = nll_cpp,
                 rel_diff = rel_diff, r_sec = r_time, cpp_sec = cpp_time,
                 speedup = speedup, stringsAsFactors = FALSE))

    expect_true(rel_diff < 0.05,
      info = sprintf("%s: R=%.4f C++=%.4f rel_diff=%.4e", mod, nll_r, nll_cpp, rel_diff))
    expect_true(cpp_time < r_time,
      info = sprintf("%s: R %.3fs  C++ %.3fs", mod, r_time, cpp_time))
  }

  message("\n=== Per-person: R serial vs C++ serial (participant 1) ===")
  for (i in seq_len(nrow(results))) {
    row <- results[i, ]
    message(sprintf("  %-8s  R %7.3fs  C++ %7.3fs  %6.1fx  nll_diff=%.2e",
                    row$model, row$r_sec, row$cpp_sec, row$speedup, row$rel_diff))
  }
  message("")
})

# =============================================================================
# 2. Group serial: R vs C++ (fitConfModels, .parallel=FALSE), all models
# =============================================================================

test_that("R and C++ fitConfModels agree and C++ is faster (serial, all models)", {
  r_time <- system.time(
    r_res <- fitConfModels(group_data, models = all_models,
                           nInits = n_inits, nRestart = n_restart,
                           .parallel = FALSE, fast = FALSE)
  )[["elapsed"]]

  cpp_time <- system.time(
    cpp_res <- fitConfModels(group_data, models = all_models,
                             nInits = n_inits, nRestart = n_restart,
                             .parallel = FALSE, fast = TRUE)
  )[["elapsed"]]

  r_res   <- r_res[order(r_res$model, r_res$participant), ]
  cpp_res <- cpp_res[order(cpp_res$model, cpp_res$participant), ]

  expect_equal(nrow(r_res), nrow(cpp_res))

  for (i in seq_len(nrow(r_res))) {
    mod <- r_res$model[i]
    sbj <- r_res$participant[i]
    nll_r   <- r_res$negLogLik[i]
    nll_cpp <- cpp_res$negLogLik[i]

    expect_true(is.finite(nll_r) && is.finite(nll_cpp),
      info = sprintf("%s sbj=%s: non-finite negLogLik", mod, sbj))

    rel_diff <- abs(nll_r - nll_cpp) / max(abs(nll_r), 1)
    expect_true(rel_diff < 0.05,
      info = sprintf("%s sbj=%s: R=%.4f C++=%.4f rel_diff=%.2e",
                     mod, sbj, nll_r, nll_cpp, rel_diff))
  }

  # --- output structure checks (both paths) ---
  for (res_i in list(list(res = r_res, label = "R"),
                     list(res = cpp_res, label = "C++"))) {
    res <- res_i$res
    label <- res_i$label

    expected_rows <- length(all_models) * length(test_participants)
    expect_equal(nrow(res), expected_rows,
      info = sprintf("%s: expected %d rows, got %d", label, expected_rows, nrow(res)))

    expect_true(all(all_models %in% res$model),
      info = sprintf("%s: not all models present", label))
    expect_true(all(test_participants %in% res$participant),
      info = sprintf("%s: not all participants present", label))

    expect_true(all(is.finite(res$negLogLik) & res$negLogLik > 0),
      info = sprintf("%s: some negLogLik non-finite or non-positive", label))
  }

  speedup <- r_time / max(cpp_time, 1e-6)
  message(sprintf(
    "\n=== Group serial (%d participants, %d models): R %.3fs  C++ %.3fs  (%.1fx) ===\n",
    length(test_participants), length(all_models), r_time, cpp_time, speedup))

  expect_true(cpp_time < r_time,
    info = sprintf("R %.3fs  C++ %.3fs", r_time, cpp_time))
})

# =============================================================================
# 3. Group parallel: R vs C++ (fitConfModels, .parallel=TRUE), all models
# =============================================================================

test_that("R and C++ fitConfModels agree and C++ is faster (parallel, all models)", {
  r_time <- system.time(
    r_res <- fitConfModels(group_data, models = all_models,
                           nInits = n_inits, nRestart = n_restart,
                           .parallel = TRUE, n.cores = n_cores, fast = FALSE)
  )[["elapsed"]]

  cpp_time <- system.time(
    cpp_res <- fitConfModels(group_data, models = all_models,
                             nInits = n_inits, nRestart = n_restart,
                             .parallel = TRUE, n.cores = n_cores, fast = TRUE)
  )[["elapsed"]]

  r_res   <- r_res[order(r_res$model, r_res$participant), ]
  cpp_res <- cpp_res[order(cpp_res$model, cpp_res$participant), ]

  expect_equal(nrow(r_res), nrow(cpp_res))

  for (i in seq_len(nrow(r_res))) {
    mod <- r_res$model[i]
    sbj <- r_res$participant[i]
    nll_r   <- r_res$negLogLik[i]
    nll_cpp <- cpp_res$negLogLik[i]

    expect_true(is.finite(nll_r) && is.finite(nll_cpp),
      info = sprintf("%s sbj=%s: non-finite negLogLik", mod, sbj))

    rel_diff <- abs(nll_r - nll_cpp) / max(abs(nll_r), 1)
    expect_true(rel_diff < 0.05,
      info = sprintf("%s sbj=%s: R=%.4f C++=%.4f rel_diff=%.2e",
                     mod, sbj, nll_r, nll_cpp, rel_diff))
  }

  speedup <- r_time / max(cpp_time, 1e-6)
  message(sprintf(
    "\n=== Group parallel (%d cores, %d participants, %d models): R %.3fs  C++ %.3fs  (%.1fx) ===\n",
    n_cores, length(test_participants), length(all_models), r_time, cpp_time, speedup))

  expect_true(cpp_time < r_time,
    info = sprintf("R %.3fs  C++ %.3fs", r_time, cpp_time))
})
