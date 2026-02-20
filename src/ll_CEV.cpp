#include "likelihoods_func.h"
#include "utils.h"
#include <RcppNumerical.h>

using namespace Numer;

// RB integrand: mean = (1-w)*x + ds*w
class CEVIntegrand_RB : public Func
{
private:
    double loc_x;    // mean of dnorm (locA or locB)
    double c_lo;     // c_RB[i]
    double c_hi;     // c_RB[i+1]
    double w;        // weight 
    double ds_j;     // sensitivity j
    double sigma;
public:
    CEVIntegrand_RB() = default;

    void set_params(double loc_x_, double c_lo_, double c_hi_,
                    double w_, double ds_j_, double sigma_) {
        loc_x = loc_x_; c_lo = c_lo_; c_hi = c_hi_;
        w = w_; ds_j = ds_j_; sigma = sigma_;
    }

    double operator()(const double& x) const override {
        const double z = x - loc_x;
        const double dnorm_val = constants::INV_SQRT_2PI * std::exp(-0.5 * z * z);
        // mean = (1 - w) * x + ds * w  (positive for RB)
        const double mean_val = (1.0 - w) * x + ds_j * w;
        const double pnorm_hi = pnorm_cpp(c_hi, mean_val, sigma);
        const double pnorm_lo = pnorm_cpp(c_lo, mean_val, sigma);
        return dnorm_val * (pnorm_hi - pnorm_lo);
    }
};

// RA integrand: mean = (1-w)*x - ds*w
class CEVIntegrand_RA : public Func
{
private:
    double loc_x;
    double c_lo;
    double c_hi;
    double w;
    double ds_j;
    double sigma;
public:
    CEVIntegrand_RA() = default;

    void set_params(double loc_x_, double c_lo_, double c_hi_,
                    double w_, double ds_j_, double sigma_) {
        loc_x = loc_x_; c_lo = c_lo_; c_hi = c_hi_;
        w = w_; ds_j = ds_j_; sigma = sigma_;
    }

    double operator()(const double& x) const override {
        const double z = x - loc_x;
        const double dnorm_val = constants::INV_SQRT_2PI * std::exp(-0.5 * z * z);
        // mean = (1 - w) * x - ds * w  (negative for RA)
        const double mean_val = (1.0 - w) * x - ds_j * w;
        const double pnorm_hi = pnorm_cpp(c_hi, mean_val, sigma);
        const double pnorm_lo = pnorm_cpp(c_lo, mean_val, sigma);
        return dnorm_val * (pnorm_hi - pnorm_lo);
    }
};

// [[Rcpp::export]]
double ll_CEV_cpp(const arma::vec& p, const arma::mat& N_SA_RA, const arma::mat& N_SA_RB, const arma::mat& N_SB_RA,
        const arma::mat& N_SB_RB, int nRatings, int nCond) {
    
    
    const arma::vec ds = compute_sensitivity(p, nCond);
    const arma::vec locA = -ds / 2.0;
    const arma::vec locB =  ds / 2.0;
    const double theta = p(nCond + nRatings - 1);

    // R: c_RA <- c(-Inf, p[nCond+nRatings-1] - rev(cumsum(exp(p[(nCond+1):(nCond+nRatings-2)]))), p[nCond+nRatings-1], Inf)
    // R: c_RB <- c(-Inf, p[nCond+nRatings+1], p[nCond+nRatings+1] + cumsum(exp(p[(nCond+nRatings+2):(nCond+nRatings*2-1)])), Inf)

    // R indexing
    int ra_start_r = nCond + 1;
    int ra_end_r = nCond + nRatings - 2;
    int rb_start_r = nCond + nRatings + 2;
    int rb_end_r = nCond + nRatings * 2 - 1;

    // allow reversed sequence |end - start| + 1 elements (R start > end is reversed)
    int n_ra_mid = std::abs(ra_end_r - ra_start_r) + 1;
    int n_rb_mid = std::abs(rb_end_r - rb_start_r) + 1;

    const double anchor_ra = p(nCond + nRatings - 2);  // p[nCond+nRatings-1]
    const double anchor_rb = p(nCond + nRatings);      // p[nCond+nRatings+1]

    // c_RA: c(-Inf, anchor - rev(cumsum(exp(mid))), anchor, Inf) = n_ra_mid + 3 elements
    // c_RB: c(-Inf, anchor, anchor + cumsum(exp(mid)), Inf) = n_rb_mid + 3 elements
    arma::vec c_RA(n_ra_mid + 3);
    arma::vec c_RB(n_rb_mid + 3);

    c_RA(0) = -arma::datum::inf;
    c_RA(n_ra_mid + 1) = anchor_ra;
    c_RA(n_ra_mid + 2) = arma::datum::inf;

    c_RB(0) = -arma::datum::inf;
    c_RB(1) = anchor_rb;
    c_RB(n_rb_mid + 2) = arma::datum::inf;

    arma::vec p_ra_mid(n_ra_mid);
    arma::vec p_rb_mid(n_rb_mid);

    if (ra_start_r <= ra_end_r) {
        p_ra_mid = p.subvec(ra_start_r - 1, ra_end_r - 1);
    } else {
        // Reversed case: R returns elements in reverse order
        for (int k = 0; k < n_ra_mid; ++k) {
            p_ra_mid(k) = p(ra_start_r - 1 - k);
        }
    }

    if (rb_start_r <= rb_end_r) {
        p_rb_mid = p.subvec(rb_start_r - 1, rb_end_r - 1);
    } else {
        // Reversed case: R returns elements in reverse order
        for (int k = 0; k < n_rb_mid; ++k) {
            p_rb_mid(k) = p(rb_start_r - 1 - k);
        }
    }

    // c_RA intermediate: anchor - rev(cumsum(exp(p_ra_mid)))
    arma::vec cumsum_ra = arma::cumsum(arma::exp(p_ra_mid));
    c_RA.subvec(1, n_ra_mid) = anchor_ra - arma::reverse(cumsum_ra);

    // c_RB intermediate: anchor + cumsum(exp(p_rb_mid))
    arma::vec cumsum_rb = arma::cumsum(arma::exp(p_rb_mid));
    c_RB.subvec(2, n_rb_mid + 1) = anchor_rb + cumsum_rb;

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

    CEVIntegrand_RB f_rb;
    CEVIntegrand_RA f_ra;

    // Compute all integrals
    for (int j = 0; j < nCond; ++j) {
        for (int i = 0; i < nRatings; ++i) {
            if(N_SB_RB(j, i) > 0) {
                // P_SB_RB: stimulus B, response B - integrate [theta, Inf)
                f_rb.set_params(locB(j), c_RB(i), c_RB(i + 1), w, ds(j), sigma);
                p_SB_RB(j, i) = integrate(f_rb, theta, arma::datum::inf, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SB_RB(j, i) = constants::MIN_P; 
            }
            if(N_SB_RA(j, i) > 0) {
                // P_SB_RA: stimulus B, response A - integrate (-Inf, theta]
                f_ra.set_params(locB(j), c_RA(i), c_RA(i + 1), w, ds(j), sigma);
                p_SB_RA(j, i) = integrate(f_ra, -arma::datum::inf, theta, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SB_RA(j, i) = constants::MIN_P; 
            }
            if(N_SA_RA(j, i) > 0) {
                // P_SA_RA: stimulus A, response A - integrate (-Inf, theta]
                f_ra.set_params(locA(j), c_RA(i), c_RA(i + 1), w, ds(j), sigma);
                p_SA_RA(j, i) = integrate(f_ra, -arma::datum::inf, theta, err_est, err_code, max_subdiv , tol, tol);
            } else {
                p_SA_RA(j, i) = constants::MIN_P; 
            }
            if(N_SA_RB(j, i) > 0) {
                // P_SA_RB: stimulus A, response B - integrate [theta, Inf)
                f_rb.set_params(locA(j), c_RB(i), c_RB(i + 1), w, ds(j), sigma);
                p_SA_RB(j, i) = integrate(f_rb, theta, arma::datum::inf, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SA_RB(j, i) = constants::MIN_P; 
            }   
        }
    }

    return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB,
                           N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB);

}