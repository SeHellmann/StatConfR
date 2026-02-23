#include <RcppArmadillo.h>
#include "likelihoods_func.h"
#include "shared_types.h"

using namespace Rcpp;

// =========================================================
// Aggregated Likelihood Pointers
// =========================================================

// [[Rcpp::export]]
SEXP get_sdt_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_SDT_cpp));
}

// [[Rcpp::export]]
SEXP get_sdtvars_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_SDTvarS_cpp));
}

// [[Rcpp::export]]
SEXP get_2chan_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_2Chan_cpp));
}

// [[Rcpp::export]]
SEXP get_itgc_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_Mratio_cpp));
}

// [[Rcpp::export]]
SEXP get_itgcm_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_MratioF_cpp));
}

// [[Rcpp::export]]
SEXP get_cas_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_CAS_cpp));
}

// [[Rcpp::export]]
SEXP get_cev_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_CEV_cpp));
}

// [[Rcpp::export]]
SEXP get_lognorm_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_LogNorm_cpp));
}

// [[Rcpp::export]]
SEXP get_logwev_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_LogWEV_cpp));
}

// [[Rcpp::export]]
SEXP get_noisy_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_Noisy_cpp));
}

// [[Rcpp::export]]
SEXP get_pda_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_PDA_cpp));
}

// [[Rcpp::export]]
SEXP get_rce_ptr() {
    return Rcpp::XPtr<ll_func_ptr>(new ll_func_ptr(&ll_RCE_cpp));
}

// =========================================================
// Regression Likelihood Pointers
// =========================================================

// [[Rcpp::export]]
SEXP get_sdt_regression_ptr() {
    return Rcpp::XPtr<ll_regression_func_ptr>(new ll_regression_func_ptr(&ll_SDT_regression));
}

// [[Rcpp::export]]
SEXP get_2chan_regression_ptr() {
    return Rcpp::XPtr<ll_regression_func_ptr>(new ll_regression_func_ptr(&ll_2Chan_regression));
}

// [[Rcpp::export]]
SEXP get_itgc_regression_ptr() {
    return Rcpp::XPtr<ll_regression_func_ptr>(new ll_regression_func_ptr(&ll_Mratio_regression));
}

// [[Rcpp::export]]
SEXP get_itgcm_regression_ptr() {
    return Rcpp::XPtr<ll_regression_func_ptr>(new ll_regression_func_ptr(&ll_MratioF_regression));
}

// [[Rcpp::export]]
SEXP get_cas_regression_ptr() {
    return Rcpp::XPtr<ll_regression_func_ptr>(new ll_regression_func_ptr(&ll_CAS_regression));
}

// [[Rcpp::export]]
SEXP get_cev_regression_ptr() {
    return Rcpp::XPtr<ll_regression_func_ptr>(new ll_regression_func_ptr(&ll_CEV_regression));
}

// [[Rcpp::export]]
SEXP get_lognorm_regression_ptr() {
    return Rcpp::XPtr<ll_regression_func_ptr>(new ll_regression_func_ptr(&ll_LogNorm_regression));
}

// [[Rcpp::export]]
SEXP get_logwev_regression_ptr() {
    return Rcpp::XPtr<ll_regression_func_ptr>(new ll_regression_func_ptr(&ll_LogWEV_regression));
}

// [[Rcpp::export]]
SEXP get_noisy_regression_ptr() {
    return Rcpp::XPtr<ll_regression_func_ptr>(new ll_regression_func_ptr(&ll_Noisy_regression));
}

// [[Rcpp::export]]
SEXP get_pda_regression_ptr() {
    return Rcpp::XPtr<ll_regression_func_ptr>(new ll_regression_func_ptr(&ll_PDA_regression));
}

// [[Rcpp::export]]
SEXP get_rce_regression_ptr() {
    return Rcpp::XPtr<ll_regression_func_ptr>(new ll_regression_func_ptr(&ll_RCE_regression));
}