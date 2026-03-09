#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

#include <RcppArmadillo.h>
#include <cmath>

// Data Structures
struct ModelData {
    arma::mat N_SA_RA;
    arma::mat N_SA_RB;
    arma::mat N_SB_RA;
    arma::mat N_SB_RB;
    int nRatings;
    int nCond;
};

struct RegressionData {
    std::vector<arma::mat> design_matrices; // [0]=X_d, [1]=X_c, etc.
    arma::vec ratings;
    arma::vec stimulus;
    arma::vec correct;
    arma::vec counts;
    int nRatings;
    int nTrials;
};

// Typedef for the function pointer signature
typedef double (*ll_func_ptr)(const arma::vec&, const ModelData&);
typedef double (*ll_regression_func_ptr)(const arma::vec&, const RegressionData&);

#endif