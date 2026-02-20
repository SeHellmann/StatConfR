# Cross-model sanity checks for C++ likelihood functions.
# Per-model R vs. C++ agreement tests live in the individual test-ll*-cpp.R files.

test_that("all C++ likelihood functions return finite, non-negative values", {
  set.seed(42)
  nCond <- 2; nRatings <- 4

  counts <- generate_count_matrices(nCond, nRatings)
  N_SA_RA <- counts$N_SA_RA; N_SA_RB <- counts$N_SA_RB
  N_SB_RA <- counts$N_SB_RA; N_SB_RB <- counts$N_SB_RB

  p_sdt <- rnorm(nCond + 2 * nRatings - 1, sd = 0.5)
  p_itg <- rnorm(nCond + 2 * nRatings,     sd = 0.5)
  p_ext <- rnorm(nCond + 2 * nRatings + 1, sd = 0.5)

  fns <- list(
    ll_SDT_cpp     = function() ll_SDT_cpp(    p_sdt, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
    ll_Mratio_cpp  = function() ll_Mratio_cpp( p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
    ll_MratioF_cpp = function() ll_MratioF_cpp(p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
    ll_2Chan_cpp   = function() ll_2Chan_cpp(  p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
    ll_CEV_cpp     = function() ll_CEV_cpp(    p_ext, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
    ll_LogNorm_cpp = function() ll_LogNorm_cpp(p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
    ll_LogWEV_cpp  = function() ll_LogWEV_cpp( p_ext, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
    ll_Noisy_cpp   = function() ll_Noisy_cpp(  p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
    ll_PDA_cpp     = function() ll_PDA_cpp(    p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
    ll_CAS_cpp     = function() ll_CAS_cpp(    p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
    ll_SDTvarS_cpp = function() ll_SDTvarS_cpp(p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)
  )
  for (nm in names(fns)) {
    val <- fns[[nm]]()
    expect_true(is.finite(val), info = sprintf("%s returned non-finite value", nm))
    expect_true(val >= 0,       info = sprintf("%s returned negative neglogL",  nm))
  }
})

test_that("C++ functions handle edge cases: single condition and minimum ratings", {
  for (cfg in list(list(nCond = 1, nRatings = 4), list(nCond = 2, nRatings = 2))) {
    set.seed(456)
    nCond <- cfg$nCond; nRatings <- cfg$nRatings
    counts <- generate_count_matrices(nCond, nRatings)
    p_sdt <- rnorm(nCond + 2 * nRatings - 1, sd = 0.5)
    p_itg <- rnorm(nCond + 2 * nRatings,     sd = 0.5)
    p_ext <- rnorm(nCond + 2 * nRatings + 1, sd = 0.5)

    expect_true(is.finite(ll_SDT_cpp(    p_sdt, counts$N_SA_RA, counts$N_SA_RB, counts$N_SB_RA, counts$N_SB_RB, nRatings, nCond)))
    expect_true(is.finite(ll_Mratio_cpp( p_itg, counts$N_SA_RA, counts$N_SA_RB, counts$N_SB_RA, counts$N_SB_RB, nRatings, nCond)))
    expect_true(is.finite(ll_MratioF_cpp(p_itg, counts$N_SA_RA, counts$N_SA_RB, counts$N_SB_RA, counts$N_SB_RB, nRatings, nCond)))
    # 2Chan: R reference p[3:2] at nRatings=2; skip that config
    if (nRatings >= 3)
      expect_true(is.finite(ll_2Chan_cpp(p_itg, counts$N_SA_RA, counts$N_SA_RB, counts$N_SB_RA, counts$N_SB_RB, nRatings, nCond)))
    expect_true(is.finite(ll_CEV_cpp(    p_ext, counts$N_SA_RA, counts$N_SA_RB, counts$N_SB_RA, counts$N_SB_RB, nRatings, nCond)))
    expect_true(is.finite(ll_LogNorm_cpp(p_itg, counts$N_SA_RA, counts$N_SA_RB, counts$N_SB_RA, counts$N_SB_RB, nRatings, nCond)))
    expect_true(is.finite(ll_LogWEV_cpp( p_ext, counts$N_SA_RA, counts$N_SA_RB, counts$N_SB_RA, counts$N_SB_RB, nRatings, nCond)))
    expect_true(is.finite(ll_Noisy_cpp(  p_itg, counts$N_SA_RA, counts$N_SA_RB, counts$N_SB_RA, counts$N_SB_RB, nRatings, nCond)))
    expect_true(is.finite(ll_PDA_cpp(    p_itg, counts$N_SA_RA, counts$N_SA_RB, counts$N_SB_RA, counts$N_SB_RB, nRatings, nCond)))
    expect_true(is.finite(ll_CAS_cpp(    p_itg, counts$N_SA_RA, counts$N_SA_RB, counts$N_SB_RA, counts$N_SB_RB, nRatings, nCond)))
    expect_true(is.finite(ll_SDTvarS_cpp(p_itg, counts$N_SA_RA, counts$N_SA_RB, counts$N_SB_RA, counts$N_SB_RB, nRatings, nCond)))
  }
})

test_that("C++ functions handle sparse count matrices", {
  set.seed(101)
  nCond <- 2; nRatings <- 4
  make_sparse <- function(n) matrix(sample(c(0, 0, 0, rpois(1, 8)), n, replace = TRUE), nrow = nCond)
  N_SA_RA <- make_sparse(nCond * nRatings); N_SA_RB <- make_sparse(nCond * nRatings)
  N_SB_RA <- make_sparse(nCond * nRatings); N_SB_RB <- make_sparse(nCond * nRatings)
  p_sdt <- rnorm(nCond + 2 * nRatings - 1, sd = 0.5)
  p_itg <- rnorm(nCond + 2 * nRatings,     sd = 0.5)
  p_ext <- rnorm(nCond + 2 * nRatings + 1, sd = 0.5)

  expect_true(is.finite(ll_SDT_cpp(    p_sdt, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)))
  expect_true(is.finite(ll_Mratio_cpp( p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)))
  expect_true(is.finite(ll_MratioF_cpp(p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)))
  expect_true(is.finite(ll_2Chan_cpp(  p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)))
  expect_true(is.finite(ll_CEV_cpp(    p_ext, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)))
  expect_true(is.finite(ll_LogNorm_cpp(p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)))
  expect_true(is.finite(ll_LogWEV_cpp( p_ext, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)))
  expect_true(is.finite(ll_Noisy_cpp(  p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)))
  expect_true(is.finite(ll_PDA_cpp(    p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)))
  expect_true(is.finite(ll_CAS_cpp(    p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)))
  expect_true(is.finite(ll_SDTvarS_cpp(p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)))
})

test_that("all C++ likelihoods match R implementations across standard configurations", {
  configs <- list(
    list(nCond = 1, nRatings = 2),
    list(nCond = 2, nRatings = 4),
    list(nCond = 3, nRatings = 5)
  )
  for (cfg in configs) {
    set.seed(42)
    nCond <- cfg$nCond; nRatings <- cfg$nRatings
    counts <- generate_count_matrices(nCond, nRatings)
    N_SA_RA <- counts$N_SA_RA; N_SA_RB <- counts$N_SA_RB
    N_SB_RA <- counts$N_SB_RA; N_SB_RB <- counts$N_SB_RB

    p_sdt <- rnorm(nCond + 2 * nRatings - 1, sd = 0.5)
    p_itg <- rnorm(nCond + 2 * nRatings,     sd = 0.5)
    p_ext <- rnorm(nCond + 2 * nRatings + 1, sd = 0.5)
    lbl   <- sprintf("nCond=%d, nRatings=%d", nCond, nRatings)

    # Closed-form models: exact agreement expected
    expect_equal(ll_SDT_cpp(    p_sdt, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 llSDT(         p_sdt, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 tolerance = 1e-10, info = paste("SDT", lbl))
    expect_equal(ll_Mratio_cpp( p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 ll_Mratio(     p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 tolerance = 1e-10, info = paste("Mratio", lbl))
    expect_equal(ll_MratioF_cpp(p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 ll_MratioF(    p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 tolerance = 1e-10, info = paste("MratioF", lbl))
    # 2Chan: R reference p[3:2] at nRatings=2; skip that config
    if (nRatings >= 3)
      expect_equal(ll_2Chan_cpp(p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                   ll2Chan(     p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                   tolerance = 1e-10, info = paste("2Chan", lbl))
    expect_equal(ll_SDTvarS_cpp(p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 llSDTvarS(     p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 tolerance = 1e-10, info = paste("SDTvarS", lbl))

    # Integration-based models: lower tolerance allow small numerical differences
    expect_equal(ll_CEV_cpp(    p_ext, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 ll_CEV(        p_ext, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 tolerance = 1e-4, info = paste("CEV", lbl))
    expect_equal(ll_LogNorm_cpp(p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 ll_lognorm(    p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 tolerance = 1e-4, info = paste("LogNorm", lbl))
    expect_equal(ll_LogWEV_cpp( p_ext, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 ll_LogWEV(     p_ext, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 tolerance = 1e-4, info = paste("LogWEV", lbl))
    expect_equal(ll_Noisy_cpp(  p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 ll_Noisy(      p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 tolerance = 1e-4, info = paste("Noisy", lbl))
    expect_equal(ll_PDA_cpp(    p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 ll_PDA(        p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 tolerance = 1e-4, info = paste("PDA", lbl))
    expect_equal(ll_CAS_cpp(    p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 ll_CAS(        p_itg, N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond),
                 tolerance = 1e-4, info = paste("CAS", lbl))
  }
})
