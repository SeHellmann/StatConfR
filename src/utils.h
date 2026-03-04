#pragma once
#include <RcppArmadillo.h>
#define _USE_MATH_DEFINES
#include <cmath>

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

arma::vec compute_sensitivity(const arma::vec& p, int nCond);

double compute_negLogL(
    const arma::mat& p_SA_RA, const arma::mat& p_SA_RB,
    const arma::mat& p_SB_RA, const arma::mat& p_SB_RB,
    const arma::mat& N_SA_RA, const arma::mat& N_SA_RB,
    const arma::mat& N_SB_RA, const arma::mat& N_SB_RB
);

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
