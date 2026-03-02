// loglikelihood funtion of the CASANDRE model (Boundy-Singer et al., 2022, Nat Hum Behav)
#include "likelihoods_func.h"
#include "utils.h"
#include <RcppNumerical.h>

using namespace Numer;

//(plnorm((x-theta)/c_lo, meanlog, sdlog) - plnorm((x-theta)/c_hi, meanlog, sdlog))
class CASIntegrand_RB : public Func
{
private:
    double loc_x; // locA or locB
    double loc_lo; // lower bound
    double loc_hi; // upper bound;
    double theta;
    double meanlog;
    double sdlog;
public:
    CASIntegrand_RB() = default;

    void set_params(double loc_x_, double loc_lo_, double loc_hi_, double theta_, double meanlog_, double sdlog_) {
        loc_x = loc_x_;
        loc_lo = loc_lo_;
        loc_hi = loc_hi_;
        theta = theta_;
        meanlog = meanlog_;
        sdlog = sdlog_;

    }

    double operator()(const double& x) const override {
        const double z = x - loc_x;
        const double dnorm_val = constants::INV_SQRT_2PI * std::exp(-0.5 * z * z);
        const double diff = x - theta;
        const double plnom_lo = plnorm_cpp(diff / loc_lo, meanlog, sdlog);
        const double plnom_hi = plnorm_cpp(diff / loc_hi, meanlog, sdlog);
        return dnorm_val * (plnom_lo - plnom_hi);
    }
};

//(plnorm(abs((x-theta)/c_RA[i+1]), meanlog, sdlog) - plnorm((x-theta)/c_RA[i], meanlog, sdlog))
class CASIntegrand_RA : public Func
{
private:
    double loc_x; // locA or locB
    double loc_lo; // lower bound
    double loc_hi; // upper bound;
    double theta;
    double meanlog;
    double sdlog;
public:
    CASIntegrand_RA() = default;

    void set_params(double loc_x_, double loc_lo_, double loc_hi_, double theta_, double meanlog_, double sdlog_) {
        loc_x = loc_x_;
        loc_lo = loc_lo_;
        loc_hi = loc_hi_;
        theta = theta_;
        meanlog = meanlog_;
        sdlog = sdlog_;

    }

    double operator()(const double& x) const override {
        const double z = x - loc_x;
        const double dnorm_val = constants::INV_SQRT_2PI * std::exp(-0.5 * z * z);
        const double diff = x - theta;
        const double plnorm_lo = plnorm_cpp(std::abs(diff / loc_hi), meanlog, sdlog);
        const double plnom_hi = plnorm_cpp(diff / loc_lo, meanlog, sdlog);
        return dnorm_val * (plnorm_lo - plnom_hi);
    }
};

// [[Rcpp::export]]
double ll_CAS_cpp(const arma::vec& p, const arma::mat& N_SA_RA, const arma::mat& N_SA_RB, const arma::mat& N_SB_RA,
        const arma::mat& N_SB_RB, int nRatings, int nCond){

    const arma::vec ds = compute_sensitivity(p, nCond);
    const arma::vec locA = -ds / 2.0;
    const arma::vec locB =  ds / 2.0;
    const double theta = p(nCond + nRatings - 1);
    const double sigma = std::exp(p(nCond + nRatings * 2 - 1));

    const double sigma_sq = sigma * sigma;
    const double meanlog = std::log(1.0 / std::sqrt(1 + sigma_sq));
    const double sdlog = std::sqrt(std::log(1 + sigma_sq));

    // R: c_RA <- c(-Inf, -rev(cumsum(c(exp(p[(nCond+1):(nCond+nRatings-1)])))), 0)
    // R: c_RB <- c(0, cumsum(c(exp(p[(nCond+nRatings+1):(nCond + nRatings*2-1)]))), Inf)

    arma::vec c_RA(nRatings + 1);
    arma::vec c_RB(nRatings + 1);

    c_RA(0) = -arma::datum::inf;
    c_RA(nRatings) = 0;

    c_RB(0) = 0;
    c_RB(nRatings) = arma::datum::inf;

    if (nRatings > 1) {
        const arma::vec p_ra = p.subvec(nCond, nCond + nRatings - 2);
        c_RA.subvec(1, nRatings - 1) = -arma::reverse(arma::cumsum(arma::exp(p_ra)));

        const arma::vec p_rb = p.subvec(nCond + nRatings, nCond + nRatings * 2 - 2);
        c_RB.subvec(1, nRatings - 1) = arma::cumsum(arma::exp(p_rb));
    }

    // Probability matrices
    arma::mat p_SA_RA(nCond, nRatings);
    arma::mat p_SA_RB(nCond, nRatings);
    arma::mat p_SB_RA(nCond, nRatings);
    arma::mat p_SB_RB(nCond, nRatings);

    double err_est;
    int err_code;
    const double tol = 1e-8;
    const int max_subdiv = 100;

    CASIntegrand_RA f_ra;
    CASIntegrand_RB f_rb;

        // Compute all integrals
    for (int j = 0; j < nCond; ++j) {
        for (int i = 0; i < nRatings; ++i) {
            if(N_SB_RB(j, i) > 0) {
                // P_SB_RB: stimulus B, response B - integrate [theta, Inf)
                f_rb.set_params(locB(j), c_RB(i), c_RB(i + 1), theta, meanlog, sdlog);
                p_SB_RB(j, i) = integrate(f_rb, theta, arma::datum::inf, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SB_RB(j, i) = constants::MIN_P; 
            }
        
            if(N_SB_RA(j, i) > 0) {
                // P_SB_RA: stimulus B, response A - integrate (-Inf, theta]
                f_ra.set_params(locB(j), c_RA(i), c_RA(i + 1), theta, meanlog, sdlog);
                p_SB_RA(j, i) = integrate(f_ra, -arma::datum::inf, theta, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SB_RA(j, i) = constants::MIN_P; 
            }
            if(N_SA_RA(j, i) > 0) {
                // P_SA_RA: stimulus A, response A - integrate (-Inf, theta]
                f_ra.set_params(locA(j), c_RA(i), c_RA(i + 1), theta, meanlog, sdlog);
                p_SA_RA(j, i) = integrate(f_ra, -arma::datum::inf, theta, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SA_RA(j, i) = constants::MIN_P; 
            }
            if(N_SA_RB(j, i) > 0) {
                // P_SA_RB: stimulus A, response B - integrate [theta, Inf)
                f_rb.set_params(locA(j), c_RB(i), c_RB(i + 1), theta, meanlog, sdlog);
                p_SA_RB(j, i) = integrate(f_rb, theta, arma::datum::inf, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SA_RB(j, i) = constants::MIN_P; 
            }
        }
    }

    return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB,
                           N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB);


}