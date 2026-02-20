#include "likelihoods_func.h"
#include "utils.h"
#include <RcppNumerical.h>

using namespace Numer;

// Integrate theta to +Inf
class LogNormIntegrand_RB : public Func
{
private:
    double loc_x; // locA or locB
    double loc_lo; // lower bound
    double loc_hi; // upper bound
    double sigma;
    double theta;
public:
    LogNormIntegrand_RB() = default;

    void set_params(double loc_x_, double loc_lo_, double loc_hi_, double sigma_, double theta_) {
        loc_x = loc_x_;
        loc_lo = loc_lo_;
        loc_hi = loc_hi_;
        sigma = sigma_;
        theta = theta_;
    }

    double operator()(const double& x) const override {
        const double z = x - loc_x;
        const double dnorm_val = constants::INV_SQRT_2PI * std::exp(-0.5 * z * z);
        const double diff = x - theta;
        const double plnom_lo = plnorm_cpp(diff, loc_lo, sigma);
        const double plnom_hi = plnorm_cpp(diff, loc_hi, sigma);
        return dnorm_val * (plnom_lo - plnom_hi);
    }
};

class LogNormIntegrand_RA : public Func
{
private:
    double loc_x; // locA or locB
    double loc_lo; // lower bound
    double loc_hi; // upper bound
    double sigma;
    double theta;
public:
    LogNormIntegrand_RA() = default;

    void set_params(double loc_x_, double loc_lo_, double loc_hi_, double sigma_, double theta_) {
        loc_x = loc_x_;
        loc_lo = loc_lo_;
        loc_hi = loc_hi_;
        sigma = sigma_;
        theta = theta_;
    }

    double operator()(const double& x) const override {
        const double z = x - loc_x;
        const double dnorm_val = constants::INV_SQRT_2PI * std::exp(-0.5 * z * z);
        const double diff = theta - x;
        double plnom_lo = plnorm_cpp(diff, loc_lo, sigma);
        double plnom_hi = plnorm_cpp(diff, loc_hi, sigma);
        return dnorm_val * (plnom_hi - plnom_lo);
    }
};

 // [[Rcpp::export]]
double ll_LogNorm_cpp(const arma::vec& p, const arma::mat& N_SA_RA, const arma::mat& N_SA_RB, const arma::mat& N_SB_RA,
        const arma::mat& N_SB_RB, int nRatings, int nCond){
    
    const arma::vec ds = compute_sensitivity(p, nCond);
    const arma::vec locA = -ds / 2.0;
    const arma::vec locB =  ds / 2.0;
    const double theta = p(nCond + nRatings - 1);

    // sigma needs to be bounded between 0 and Inf
    // see whether the results converge.
    const double sigma = std::exp(p(nCond + nRatings * 2 - 1));
        
    //average placement of the confidence criteria
    const arma::vec mu_cA = theta + arma::reverse(arma::cumsum(arma::exp(p.subvec(nCond, nCond + nRatings - 2))));
    const arma::vec mu_cB = theta + arma::cumsum(arma::exp(p.subvec(nCond + nRatings, nCond + nRatings * 2 - 2)));

    // convert to the location parameter of the latent lognormal distribution
    arma::vec loc_RA(nRatings + 1);
    arma::vec loc_RB(nRatings + 1);

    //loc_RA <-  c(Inf,log(abs(mu_cA - theta)) - .5*sigma^2, -Inf) # the order here is REVERSED!!!
    //loc_RB <- c(-Inf, log(mu_cB - theta) - .5*sigma^2, Inf)
    const double sigma2_div2 = 0.5 * sigma * sigma;
    loc_RA(0) = arma::datum::inf;
    loc_RA(nRatings) = -arma::datum::inf;

    loc_RB(0) = -arma::datum::inf;
    loc_RB(nRatings) = arma::datum::inf;

    for (int i = 0; i < nRatings - 1; i++) {
        loc_RA(i + 1) = std::log(std::abs(mu_cA(i) - theta)) - sigma2_div2;
        loc_RB(i + 1) = std::log(mu_cB(i) - theta) - sigma2_div2;
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

    LogNormIntegrand_RA f_ra;
    LogNormIntegrand_RB f_rb;

    // Compute all integrals
    for (int j = 0; j < nCond; j++) {
        for (int i = 0; i < nRatings; i++) {
            if(N_SB_RB(j, i) > 0) { 
                // P_SB_RB: stimulus B, response B - integrate [theta, Inf)
                f_rb.set_params(locB(j), loc_RB(i), loc_RB(i + 1), sigma, theta);
                p_SB_RB(j, i) = integrate(f_rb, theta, arma::datum::inf, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SB_RB(j, i) = constants::MIN_P; 
            }
                if(N_SB_RA(j, i) > 0) { 
                // P_SB_RA: stimulus B, response A - integrate (-Inf, theta]
                f_ra.set_params(locB(j), loc_RA(i), loc_RA(i + 1), sigma, theta);
                p_SB_RA(j, i) = integrate(f_ra, -arma::datum::inf, theta, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SB_RA(j, i) = constants::MIN_P; 
            }
                if (N_SA_RA(j, i) > 0) {
                // P_SA_RA: stimulus A, response A - integrate (-Inf, theta]
                f_ra.set_params(locA(j), loc_RA(i), loc_RA(i + 1), sigma, theta);
                p_SA_RA(j, i) = integrate(f_ra, -arma::datum::inf, theta, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SA_RA(j, i) = constants::MIN_P; 
            }
            if(N_SA_RB(j, i) > 0) {
                // P_SA_RB: stimulus A, response B - integrate [theta, Inf)
                f_rb.set_params(locA(j), loc_RB(i), loc_RB(i + 1), sigma, theta);
                p_SA_RB(j, i) = integrate(f_rb, theta, arma::datum::inf, err_est, err_code, max_subdiv, tol, tol);
            } else {
                p_SA_RB(j, i) = constants::MIN_P; 
            }
        }
    }
    return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB,
        N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB);
}
