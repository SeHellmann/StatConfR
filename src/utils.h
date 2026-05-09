#pragma once
#include <RcppArmadillo.h>
#include "shared_types.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifndef M_SQRT2
#define M_SQRT2    1.41421356237309504880
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2  0.70710678118654752440
#endif

namespace constants {
    constexpr double MIN_P = 1e-64;
    constexpr double INV_SQRT_2PI = 0.3989422804014327; // 1/sqrt(2*pi)
    constexpr double LOG_MIN_P = -147.4131591025766;    // log(1e-64)
    constexpr double M1_SQRTPI = 0.5641895835477563; // 1/sqrt(pi)
}

inline arma::vec compute_sensitivity(const arma::vec& p, int nCond) {
    return arma::cumsum(arma::exp(p.subvec(0, nCond - 1)));
}

inline double pnorm_cpp(double x, double mu, double sigma) {
    return 0.5 * std::erfc(-(x - mu) / (sigma * M_SQRT2));
}

inline double plnorm_cpp(double q, double meanlog, double sdlog) {
    if (q <= 0.0) return 0.0;
    return 0.5 * std::erfc(-(std::log(q) - meanlog) / (sdlog * M_SQRT2));
}

inline double normcdf_cpp(double z) {
    return 0.5 * std::erfc(-z * M_SQRT1_2);
}

inline double clamped_log(double p) {
    if (std::isnan(p) || p < constants::MIN_P) return constants::LOG_MIN_P;
    if (p > 1.0) return 0.0;
    return std::log(p);
}

template <typename ProbCalc>
inline double compute_regression_negLogL(const arma::vec& p, const RegressionData& dat,
                                         int n_meta, ProbCalc calc_prob) {
    int nRatings = dat.nRatings;
    int nUniqueTrials = dat.ratings.n_elem;
    
    // Unpack default design matrices for d' and c
    const arma::mat& X_d = dat.design_matrices[0];
    const arma::mat& X_c = dat.design_matrices[1];
    
    int nBeta_d = X_d.n_cols;
    int nBeta_c = X_c.n_cols;
    
    arma::vec beta_d = p.subvec(0, nBeta_d - 1);
    arma::vec beta_c = p.subvec(nBeta_d, nBeta_d + nBeta_c - 1);
    
    // Unpack meta-parameters
    int current_idx = nBeta_d + nBeta_c;
    std::vector<arma::vec> meta_params(n_meta);
    std::vector<arma::vec> meta_vecs(n_meta);
    
    for (int m = 0; m < n_meta; ++m) {
        int nBeta_m = dat.design_matrices[2 + m].n_cols;
        meta_params[m] = p.subvec(current_idx, current_idx + nBeta_m - 1);
        meta_vecs[m] = dat.design_matrices[2 + m] * meta_params[m];
        current_idx += nBeta_m;
    }
    
    // Unpack Confidence Steps
    arma::vec theta_raw;
    arma::vec p_1, p_2;
    if (nRatings > 1) {
        theta_raw = p.subvec(current_idx, p.n_elem - 1);
        p_1 = arma::reverse(arma::cumsum(arma::exp(theta_raw.subvec(0, nRatings - 2))));
        p_2 = arma::cumsum(arma::exp(theta_raw.subvec(nRatings - 1, 2 * nRatings - 3)));
    }
    
    // Calculate vectors for d' and c 
    arma::vec d_vec = arma::exp(X_d * beta_d);
    arma::vec c_vec = X_c * beta_c;
    
    double negLogL = 0.0;
    // Per trial contributions to the negative log-likelihood
    arma::vec contribs(nUniqueTrials, arma::fill::zeros);
    
    // Optimization Loop
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < nUniqueTrials; ++k) {
        double d = d_vec(k);
        double theta = c_vec(k);
        
        int stim = std::round(dat.stimulus(k));
        int correct = std::round(dat.correct(k));
        int rating = std::round(dat.ratings(k));
        double count = dat.counts(k);
        
        bool resp_A = (stim != correct);
        double loc = (stim == 0) ? -d / 2.0 : d / 2.0;
        
        double lower_bound, upper_bound;
        
        // Map response to integration bounds
        if (resp_A) {
            int idx_lower = nRatings - rating;
            int idx_upper = nRatings - rating + 1;
            lower_bound = (idx_lower == 0) ? -arma::datum::inf : 
                          (idx_lower == nRatings) ? theta : (theta - p_1(idx_lower - 1));
            upper_bound = (idx_upper == 0) ? -arma::datum::inf : 
                          (idx_upper == nRatings) ? theta : (theta - p_1(idx_upper - 1));
        } else {
            int idx_lower = rating - 1;
            int idx_upper = rating;
            lower_bound = (idx_lower == 0) ? theta : 
                          (idx_lower == nRatings) ? arma::datum::inf : (theta + p_2(idx_lower - 1));
            upper_bound = (idx_upper == 0) ? theta : 
                          (idx_upper == nRatings) ? arma::datum::inf : (theta + p_2(idx_upper - 1));
        }
        
        std::vector<double> m_vals(n_meta);
        for (int m = 0; m < n_meta; ++m) {
            m_vals[m] = meta_vecs[m](k);
        }
        
        // Invoke the provided model-specific probability calculator
        double p_obs = calc_prob(resp_A, loc, theta, lower_bound, upper_bound, d, m_vals);
        
        contribs(k) = -count * std::log(std::max(p_obs, constants::MIN_P));
    }
    
    negLogL = arma::accu(contribs);
    return negLogL;
}