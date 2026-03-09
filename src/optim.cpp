// [[Rcpp::depends(RcppArmadillo)]]
#include "shared_types.h"
#include "nelder_mead.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;


struct OptimContext {
    ModelData* data;
    ll_func_ptr func;
};

double generic_objective_wrapper(int n, double *par, void *ex) {
    OptimContext* ctx = (OptimContext*)ex;
    vec p(par, n, false, true); 
    return ctx->func(p, *(ctx->data));
}


// [[Rcpp::export]]
NumericVector compute_generic_grid_ll(SEXP ll_function_ptr, 
                                      arma::mat inits, 
                                      arma::mat N_SA_RA, arma::mat N_SA_RB, 
                                      arma::mat N_SB_RA, arma::mat N_SB_RB, 
                                      int nRatings, int nCond) {

    Rcpp::XPtr<ll_func_ptr> xptr(ll_function_ptr);
    ll_func_ptr model_fn = *xptr;

    ModelData dat;
    dat.N_SA_RA = N_SA_RA.eval(); dat.N_SA_RB = N_SA_RB.eval();
    dat.N_SB_RA = N_SB_RA.eval(); dat.N_SB_RB = N_SB_RB.eval();
    dat.nRatings = nRatings; dat.nCond = nCond;
    
    int n_rows = inits.n_rows;
    NumericVector results(n_rows);
    
    for(int i=0; i<n_rows; ++i) {
        results[i] = model_fn(inits.row(i).t(), dat);
    }
    return results;
}

// [[Rcpp::export]]
List optimize_generic_loop(SEXP ll_function_ptr, 
                           arma::mat selected_inits, 
                           arma::mat N_SA_RA, arma::mat N_SA_RB, 
                           arma::mat N_SB_RA, arma::mat N_SB_RB,
                           int nRestart, int nRatings, int nCond) {

    Rcpp::XPtr<ll_func_ptr> xptr(ll_function_ptr);
    ll_func_ptr model_fn = *xptr;
    
    ModelData dat;
    dat.N_SA_RA = N_SA_RA.eval(); dat.N_SA_RB = N_SA_RB.eval();
    dat.N_SB_RA = N_SB_RA.eval(); dat.N_SB_RB = N_SB_RB.eval();
    dat.nRatings = nRatings; dat.nCond = nCond;
    
    int nInits = selected_inits.n_rows;
    int nParams = selected_inits.n_cols;
    
    mat final_pars(nInits, nParams);
    vec final_vals(nInits);
    std::vector<int> fails(nInits, 0);
    
    for(int i = 0; i < nInits; ++i) {
        
        vec start_par = selected_inits.row(i).t();
        vec current_par = start_par;
        double val = model_fn(start_par, dat);
        int fail = 0; 
        int fncount = 0;

        OptimContext ctx;
        ctx.data = &dat;
        ctx.func = model_fn;
        
        nelder_mead_optimize(nParams, start_par.memptr(), current_par.memptr(), &val,
                             generic_objective_wrapper, &fail, 1e-4, &ctx,
                             &fncount, 10000);
        
        if (fail == 0 || fail == 1) {
            for(int j = 0; j < nRestart; ++j) {
                vec prev_par = current_par;
                nelder_mead_optimize(nParams, prev_par.memptr(), current_par.memptr(), &val,
                                     generic_objective_wrapper, &fail, 1e-8, &ctx,
                                     &fncount, 1000000);
            }
        }
        
        final_pars.row(i) = current_par.t();
        final_vals[i] = val;
        fails[i] = fail;
    }
    
    double best_value = datum::inf;
    vec best_par;
    int best_fail_code = -1;
    bool found_any = false;
    
    for(int i=0; i<nInits; ++i) {
        if(std::isfinite(final_vals[i]) && final_vals[i] < best_value) {
            best_value = final_vals[i];
            best_par = final_pars.row(i).t();
            best_fail_code = fails[i];
            found_any = true;
        }
    }
    
    if (!found_any) {
        return List::create(Named("error") = true,
                            Named("msg") = "All optimizations returned Inf/NaN");
    }
    
    return List::create(Named("par") = best_par, 
                        Named("value") = best_value,
                        Named("error") = false,
                        Named("fail_code") = best_fail_code);
}



struct RegressionOptimContext {
    RegressionData* data;
    ll_regression_func_ptr func;
};

double regression_objective_wrapper(int n, double *par, void *ex) {
    RegressionOptimContext* ctx = (RegressionOptimContext*)ex;
    vec p(par, n, false, true); 
    return ctx->func(p, *(ctx->data));
}

// [[Rcpp::export]]
List optimize_regression_loop(SEXP ll_function_ptr, 
                         arma::mat selected_inits, 
                         List design_matrices, 
                         arma::vec stimulus, arma::vec ratings,
                         arma::vec correct,
                         arma::vec counts,
                         int nRestart, int nRatings, int nTrials) {

    Rcpp::XPtr<ll_regression_func_ptr> xptr(ll_function_ptr);
    ll_regression_func_ptr model_fn = *xptr;

    RegressionData dat;
    for(int i = 0; i < design_matrices.size(); ++i) {
        dat.design_matrices.push_back(as<arma::mat>(design_matrices[i]));
    }
    
    dat.stimulus = stimulus.eval();
    dat.ratings = ratings.eval();
    dat.correct = correct.eval();
    dat.counts = counts.eval();
    dat.nRatings = nRatings;
    dat.nTrials = counts.n_elem;
    
    int nInits = selected_inits.n_rows;
    int nParams = selected_inits.n_cols;
    
    mat final_pars(nInits, nParams);
    vec final_vals(nInits);
    std::vector<int> fails(nInits, 0);     

    for(int i = 0; i < nInits; ++i) {
        vec start_par = selected_inits.row(i).t();
        vec current_par = start_par;
        double val = model_fn(start_par, dat);
        int fail = 0; int fncount = 0;
        RegressionOptimContext ctx;
        ctx.data = &dat;
        ctx.func = model_fn;
        nelder_mead_optimize(nParams, start_par.memptr(), current_par.memptr(), &val,
                             regression_objective_wrapper, &fail, 1e-4, &ctx,
                             &fncount, 10000);
        if (fail == 0 || fail == 1) {
            for(int j = 0; j < nRestart; ++j) {
                vec prev_par = current_par;
                nelder_mead_optimize(nParams, prev_par.memptr(), current_par.memptr(), &val,
                                     regression_objective_wrapper, &fail, 1e-8, &ctx,
                                     &fncount, 1000000);
            }
        }
        final_pars.row(i) = current_par.t();
        final_vals[i] = val;
        fails[i] = fail;
    }

    double best_value = datum::inf;
    vec best_par;
    bool found_any = false;
    for(int i=0; i<nInits; ++i) {
        if(std::isfinite(final_vals[i]) && final_vals[i] < best_value) {
            best_value = final_vals[i];
            best_par = final_pars.row(i).t();
            found_any = true;
        }
    }
    if (!found_any) return List::create(Named("error") = true);
    return List::create(Named("par") = best_par, Named("value") = best_value, Named("error") = false);
}