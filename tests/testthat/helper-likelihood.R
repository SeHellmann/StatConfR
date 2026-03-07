compare_likelihood_generic <- function(r_func, cpp_func, inputs, tolerance = 1e-10) {
  r_result <- r_func(
    inputs$p, inputs$N_SA_RA, inputs$N_SA_RB,
    inputs$N_SB_RA, inputs$N_SB_RB,
    inputs$nRatings, inputs$nCond
  )
  cpp_result <- cpp_func(
    inputs$p, inputs$N_SA_RA, inputs$N_SA_RB,
    inputs$N_SB_RA, inputs$N_SB_RB,
    inputs$nRatings, inputs$nCond
  )
  list(
    r_result   = r_result,
    cpp_result = cpp_result,
    difference = abs(r_result - cpp_result),
    rel_diff   = abs(r_result - cpp_result) / max(abs(r_result), 1),
    match      = abs(r_result - cpp_result) / max(abs(r_result), 1) < tolerance
  )
}

generate_count_matrices <- function(nCond, nRatings, lambda_diag = 20, lambda_offdiag = 10) {
  list(
    N_SA_RA = matrix(rpois(nCond * nRatings, lambda = lambda_diag),    nrow = nCond, ncol = nRatings),
    N_SA_RB = matrix(rpois(nCond * nRatings, lambda = lambda_offdiag), nrow = nCond, ncol = nRatings),
    N_SB_RA = matrix(rpois(nCond * nRatings, lambda = lambda_offdiag), nrow = nCond, ncol = nRatings),
    N_SB_RB = matrix(rpois(nCond * nRatings, lambda = lambda_diag),    nrow = nCond, ncol = nRatings)
  )
}
