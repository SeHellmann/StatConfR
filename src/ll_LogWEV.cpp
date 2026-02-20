#include "likelihoods_func.h"
#include "utils.h"
#include <RcppNumerical.h>

using namespace Numer;

// (plnorm(q = c_RB[i+1], (1 - w) * x + ds[j] * w, sigma) - plnorm(q = c_RB[i], (1 - w) * x + ds[j] * w,  sigma)
class LogWEVIntegrand_RB : public Func
{
private:
    double loc_x;    // mean of dnorm (locA or locB)
    double c_lo;     // c_RB[i]
    double c_hi;     // c_RB[i+1]
    double w;        // weight 
    double ds_j;     // sensitivity j
    double sigma;
public:
    LogWEVIntegrand_RB() = default;

    void set_params(double loc_x_, double c_lo_, double c_hi_,
                    double w_, double ds_j_, double sigma_) {
        loc_x = loc_x_; c_lo = c_lo_; c_hi = c_hi_;
        w = w_; ds_j = ds_j_; sigma = sigma_;
    }

    double operator()(const double& x) const override {
        const double z = x - loc_x;
        const double dnorm_val = constants::INV_SQRT_2PI * std::exp(-0.5 * z * z);
        // meanlog = (1 - w) * x + ds * w
        const double meanlog = (1.0 - w) * x + ds_j * w;
        const double plnorm_hi = plnorm_cpp(c_hi, meanlog, sigma);
        const double plnorm_lo = plnorm_cpp(c_lo, meanlog, sigma);
        return dnorm_val * (plnorm_hi - plnorm_lo);
    }
};

// (plnorm(q = -c_RA[i+1], -(1 - w) * x + ds[j] * w, sigma) - plnorm(q = -c_RA[i], -(1 - w) * x + ds[j] * w, sigma)
class LogWEVIntegrand_RA : public Func
{
private:
    double loc_x;    // mean of dnorm (locA or locB)
    double c_lo;     // c_RA[i]
    double c_hi;     // c_RA[i+1]
    double w;        // weight
    double ds_j;     // sensitivity j
    double sigma;
public:
    LogWEVIntegrand_RA() = default;

    void set_params(double loc_x_, double c_lo_, double c_hi_,
                    double w_, double ds_j_, double sigma_) {
        loc_x = loc_x_; c_lo = c_lo_; c_hi = c_hi_;
        w = w_; ds_j = ds_j_; sigma = sigma_;
    }

    double operator()(const double& x) const override {
        const double z = x - loc_x;
        const double dnorm_val = constants::INV_SQRT_2PI * std::exp(-0.5 * z * z);
        // meanlog = -(1 - w) * x + ds * w  (note the negative on x term)
        const double meanlog = -(1.0 - w) * x + ds_j * w;
        // criteria are negated: -c_RA[i+1] and -c_RA[i]
        const double plnorm_hi = plnorm_cpp(-c_hi, meanlog, sigma);
        const double plnorm_lo = plnorm_cpp(-c_lo, meanlog, sigma);
        return dnorm_val * (plnorm_hi - plnorm_lo);
    }
};

// [[Rcpp::export]]
double ll_LogWEV_cpp(const arma::vec& p, const arma::mat& N_SA_RA, const arma::mat& N_SA_RB,
                     const arma::mat& N_SB_RA, const arma::mat& N_SB_RB,
                     int nRatings, int nCond) {

    const arma::vec ds = compute_sensitivity(p, nCond);
    const arma::vec locA = -ds / 2.0;
    const arma::vec locB =  ds / 2.0;
    const double theta = p(nCond + nRatings - 1);

    arma::vec c_RA(nRatings + 1);
    arma::vec c_RB(nRatings + 1);

    c_RA(0) = 0.0;
    c_RA(nRatings) = -arma::datum::inf;

    c_RB(0) = 0.0;
    c_RB(nRatings) = arma::datum::inf;

    if (nRatings > 1) {
        arma::vec p_1 = arma::cumsum(arma::exp(p.subvec(nCond, nCond + nRatings - 2)));
        arma::vec p_2 = arma::cumsum(arma::exp(p.subvec(nCond + nRatings, nCond + nRatings * 2 - 2)));
        c_RA.subvec(1, nRatings - 1) = -p_1;
        c_RB.subvec(1, nRatings - 1) = p_2;
    }

    const double sigma = std::exp(p(nCond + nRatings * 2 - 1));
    const double w_raw = p(nCond + nRatings * 2);
    //const double w = std::exp(w_raw) / (1.0 + std::exp(w_raw));  // log transform to (0,1)
    const double w = 1.0 / (1.0 + std::exp(-w_raw));  // avoid overflow

    // Probability matrices
    arma::mat p_SA_RA(nCond, nRatings);
    arma::mat p_SA_RB(nCond, nRatings);
    arma::mat p_SB_RA(nCond, nRatings);
    arma::mat p_SB_RB(nCond, nRatings);

    double err_est;
    int err_code;
    const double tol = 1e-8;
    const int max_subdiv = 100;

    LogWEVIntegrand_RB f_rb;
    LogWEVIntegrand_RA f_ra;

    // Compute all integrals
    for (int j = 0; j < nCond; ++j) {
        for (int i = 0; i < nRatings; ++i) {
            int i_rev = nRatings - 1 - i;
            if(N_SB_RB(j, i) > 0) {
                // P_SB_RB: stimulus B, response B - integrate [theta, Inf)
                // outer(1:nCond, 1:nRatings, P_SBRB) - normal indexing
                f_rb.set_params(locB(j), c_RB(i), c_RB(i + 1), w, ds(j), sigma);
                p_SB_RB(j, i) = integrate(f_rb, theta, arma::datum::inf, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SB_RB(j, i) = constants::MIN_P;
            }
            if(N_SB_RA(j, i_rev) > 0) { 
                // P_SB_RA: stimulus B, response A - integrate (-Inf, theta]
                // R: outer(1:nCond, nRatings:1, P_SBRA) - REVERSED
                f_ra.set_params(locB(j), c_RA(i), c_RA(i + 1), w, ds(j), sigma);
                p_SB_RA(j, i_rev) = integrate(f_ra, -arma::datum::inf, theta, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SB_RA(j, i_rev) = constants::MIN_P;
            }
            if(N_SA_RB(j, i) > 0) { 
                // P_SA_RB: stimulus A, response B - integrate [theta, Inf)
                // outer(1:nCond, 1:nRatings, P_SARB) - normal indexing
                f_rb.set_params(locA(j), c_RB(i), c_RB(i + 1), w, ds(j), sigma);
                p_SA_RB(j, i) = integrate(f_rb, theta, arma::datum::inf, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SA_RB(j, i) = constants::MIN_P;
            }
            if(N_SA_RA(j, i_rev) > 0) {
                // P_SA_RA: stimulus A, response A - integrate (-Inf, theta]
                // R: outer(1:nCond, nRatings:1, P_SARA) - REVERSED
                f_ra.set_params(locA(j), c_RA(i), c_RA(i + 1), w, ds(j), sigma);
                p_SA_RA(j, i_rev) = integrate(f_ra, -arma::datum::inf, theta, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SA_RA(j, i_rev) = constants::MIN_P;
            }
        }
    }

    return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB,
                           N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB);
} 