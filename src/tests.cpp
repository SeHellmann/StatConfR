// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "shared_types.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
double test_ll_ptr(SEXP ptr, arma::vec p, 
                    arma::mat N_SA_RA, arma::mat N_SA_RB, 
                    arma::mat N_SB_RA, arma::mat N_SB_RB, 
                    int nRatings, int nCond) {
    Rcpp::XPtr<ll_func_ptr> xptr(ptr);
    ll_func_ptr model_fn = *xptr;
    ModelData dat;
    dat.N_SA_RA = N_SA_RA.eval(); dat.N_SA_RB = N_SA_RB.eval();
    dat.N_SB_RA = N_SB_RA.eval(); dat.N_SB_RB = N_SB_RB.eval();
    dat.nRatings = nRatings; dat.nCond = nCond;
    return model_fn(p, dat);
}

// [[Rcpp::export]]
double test_ll_regression_ptr(SEXP ptr, arma::vec p, 
                              Rcpp::List design_matrices_r,
                              arma::vec ratings, 
                              arma::vec stimulus, 
                              arma::vec correct, 
                              arma::vec counts, 
                              int nRatings, int nTrials) {
    
    Rcpp::XPtr<ll_regression_func_ptr> xptr(ptr);
    ll_regression_func_ptr model_fn = *xptr;
    
    RegressionData dat;
    int n_matrices = design_matrices_r.size();
    for (int i = 0; i < n_matrices; ++i) {
        dat.design_matrices.push_back(Rcpp::as<arma::mat>(design_matrices_r[i]));
    }
    dat.ratings = ratings;
    dat.stimulus = stimulus;
    dat.correct = correct;
    dat.counts = counts;
    dat.nRatings = nRatings;
    dat.nTrials = nTrials;
    
    return model_fn(p, dat);
}