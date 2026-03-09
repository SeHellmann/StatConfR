compare_likelihood_generic <- function(model_name, r_func, inputs, tolerance = 1e-10) {
  # Get the pointer for the model
  ptr_name <- if (tolower(model_name) == "noisy") {
    "get_noisy_ptr"
  } else if (tolower(model_name) == "itgc" || tolower(model_name) == "mratio") {
    "get_itgc_ptr"
  } else if (tolower(model_name) == "itgcm" || tolower(model_name) == "mratiof") {
    "get_itgcm_ptr"
  } else if (tolower(model_name) == "sdt") {
    "get_sdt_ptr"
  } else if (tolower(model_name) == "sdtvars") {
    "get_sdtvars_ptr"
  } else if (tolower(model_name) == "casandre" || tolower(model_name) == "cas") {
    "get_cas_ptr"
  } else {
    sprintf("get_%s_ptr", tolower(model_name))
  }
  
  ptr_func <- get(ptr_name, envir = asNamespace("statConfR"))
  ptr <- ptr_func()
  
  r_result <- r_func(
    inputs$p, inputs$N_SA_RA, inputs$N_SA_RB,
    inputs$N_SB_RA, inputs$N_SB_RB,
    inputs$nRatings, inputs$nCond
  )
  
  cpp_result <- test_ll_ptr(
    ptr, inputs$p, inputs$N_SA_RA, inputs$N_SA_RB,
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
