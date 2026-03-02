#include "likelihoods_func.h"
#include "utils.h"
#include <RcppNumerical.h>

using namespace Numer;

class PDAIntegrand : public Func
{
private:
    double loc_x;    // mean of dnorm (locA or locB)
    double c_lo;     // c_RB[i]
    double c_hi;     // c_RB[i+1]
    double sigma;
    double a;

public:
    PDAIntegrand() = default;

    void set_params(double loc_x_, double c_lo_, double c_hi_, double sigma_, double a_) {
        loc_x = loc_x_; c_lo = c_lo_; c_hi = c_hi_; sigma = sigma_; a = a_;
    }

    double operator()(const double& x) const override {
        const double z = x - loc_x;
        const double dnorm_val = constants::INV_SQRT_2PI * std::exp(-0.5 * z * z);
        const double inter = x + loc_x * a;
        const double pnorm_hi = pnorm_cpp(c_hi, inter, sigma);
        const double pnorm_lo = pnorm_cpp(c_lo, inter, sigma);
        return dnorm_val * (pnorm_hi - pnorm_lo);
    }
};

// [[Rcpp::export]]
double ll_PDA_cpp(const arma::vec& p, const arma::mat& N_SA_RA, const arma::mat& N_SA_RB, const arma::mat& N_SB_RA,
        const arma::mat& N_SB_RB, int nRatings, int nCond) {
    const arma::vec ds = compute_sensitivity(p, nCond);
    const arma::vec locA = -ds / 2.0;
    const arma::vec locB =  ds / 2.0;
    const double theta = p(nCond + nRatings - 1);

    // R: c_RA <- c(-Inf, p[nCond+nRatings-1] - rev(cumsum(exp(p[(nCond+1):(nCond+nRatings-2)]))), p[nCond+nRatings-1], Inf)
    // R: c_RB <- c(-Inf, p[nCond+nRatings+1], p[nCond+nRatings+1] + cumsum(exp(p[(nCond+nRatings+2):(nCond+nRatings*2-1)])), Inf)

    const double anchor_ra = p(nCond + nRatings - 2);
    const double anchor_rb = p(nCond + nRatings);

    arma::vec c_RA(nRatings + 1);
    arma::vec c_RB(nRatings + 1);

    c_RA(0)            = -arma::datum::inf;
    c_RA(nRatings - 1) = anchor_ra;
    c_RA(nRatings)     = arma::datum::inf;

    c_RB(0)        = -arma::datum::inf;
    c_RB(1)        = anchor_rb;
    c_RB(nRatings) = arma::datum::inf;

    if (nRatings > 2) {
        const arma::vec p_ra = p.subvec(nCond, nCond + nRatings - 3);
        c_RA.subvec(1, nRatings - 2) = anchor_ra - arma::reverse(arma::cumsum(arma::exp(p_ra)));

        const arma::vec p_rb = p.subvec(nCond + nRatings + 1, nCond + nRatings * 2 - 2);
        c_RB.subvec(2, nRatings - 1) = anchor_rb + arma::cumsum(arma::exp(p_rb));
    }

    // R: a <- exp(p[nCond + nRatings*2]); sigma <- sqrt(a)
    const double a = std::exp(p(nCond + nRatings * 2 - 1));
    const double sigma = std::sqrt(a);

     // Probability matrices
    arma::mat p_SA_RA(nCond, nRatings);
    arma::mat p_SA_RB(nCond, nRatings);
    arma::mat p_SB_RA(nCond, nRatings);
    arma::mat p_SB_RB(nCond, nRatings);

    double err_est;
    int err_code;
    const double tol = 1e-8;
    const int max_subdiv = 100;

    PDAIntegrand f;

    // Compute all integrals
    for (int j = 0; j < nCond; j++) {
        for (int i = 0; i < nRatings; i++) {
            if(N_SB_RB(j, i) > 0) {
                // P_SB_RB: stimulus B, response B - integrate [theta, Inf)
                f.set_params(locB(j), c_RB(i), c_RB(i + 1), sigma, a);
                p_SB_RB(j, i) = integrate(f, theta, arma::datum::inf, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SB_RB(j, i) = constants::MIN_P;
            }
            if(N_SB_RA(j, i) > 0) {
            // P_SB_RA: stimulus B, response A - integrate (-Inf, theta]
                f.set_params(locB(j), c_RA(i), c_RA(i + 1), sigma, a);
                p_SB_RA(j, i) = integrate(f, -arma::datum::inf, theta, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SB_RA(j, i) = constants::MIN_P;
            }
            if(N_SA_RA(j, i) > 0) {
                // P_SA_RA: stimulus A, response A - integrate (-Inf, theta]
                f.set_params(locA(j), c_RA(i), c_RA(i + 1), sigma,  a);
                p_SA_RA(j, i) = integrate(f,-arma::datum::inf , theta , err_est , err_code , max_subdiv , tol , tol );
            } else {
                p_SA_RA(j, i) = constants::MIN_P;
            }
            if(N_SA_RB(j, i) > 0) {
                // P_SA_RB: stimulus A, response B - integrate [theta, Inf)
                f.set_params(locA(j), c_RB(i), c_RB(i + 1), sigma, a);
                p_SA_RB(j, i) = integrate(f, theta, arma::datum::inf, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SA_RB(j, i) = constants::MIN_P;
            }
        }
    }
    return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB,
        N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB);
}