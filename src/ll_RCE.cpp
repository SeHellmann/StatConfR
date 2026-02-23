#include "likelihoods_func.h"
#include "utils.h"
#include <RcppNumerical.h>

using namespace arma;
using namespace Numer;
using namespace arma;

class RCEIntegrand : public Func
{
private:
    double mu1, mu2, sigma;
public:
    RCEIntegrand() = default;
    void set_params(double mu1_, double mu2_, double sigma_) {
        mu1 = mu1_; mu2 = mu2_; sigma = sigma_;
    }
    double operator()(const double& x) const override {
        const double z = (x - mu1) / sigma;
        const double dnorm_val = (constants::INV_SQRT_2PI / sigma) * std::exp(-0.5 * z * z);
        const double pnorm_val = pnorm_cpp(x, mu2, sigma);
        return dnorm_val * pnorm_val;
    }
};

double ll_RCE_cpp(const arma::vec& p, const ModelData& dat) {
    const arma::mat& N_SA_RA = dat.N_SA_RA;
    const arma::mat& N_SA_RB = dat.N_SA_RB;
    const arma::mat& N_SB_RA = dat.N_SB_RA;
    const arma::mat& N_SB_RB = dat.N_SB_RB;
    int nRatings = dat.nRatings;
    int nCond = dat.nCond;

    vec ds = compute_sensitivity(p, nCond);
    double theta = p(nCond + nRatings - 1);

    // RA Criteria
    vec c_RA(nRatings + 1);
    c_RA(0) = -datum::inf;
    double anchor_ra = p(nCond + nRatings - 2);
    c_RA(nRatings - 1) = anchor_ra;
    c_RA(nRatings) = datum::inf;
    if (nRatings > 2) {
        vec steps_ra = reverse(cumsum(exp(p.subvec(nCond, nCond + nRatings - 3))));
        c_RA.subvec(1, nRatings - 2) = anchor_ra - steps_ra;
    }

    // RB Criteria
    vec c_RB(nRatings + 1);
    c_RB(0) = -datum::inf;
    double anchor_rb = p(nCond + nRatings);
    c_RB(1) = anchor_rb;
    c_RB(nRatings) = datum::inf;
    if (nRatings > 2) {
        vec steps_rb = cumsum(exp(p.subvec(nCond + nRatings + 1, nCond + 2 * nRatings - 2)));
        c_RB.subvec(2, nRatings - 1) = anchor_rb + steps_rb;
    }

    const double sigma = std::sqrt(0.5);

    mat p_SA_RA(nCond, nRatings, fill::zeros);
    mat p_SA_RB(nCond, nRatings, fill::zeros);
    mat p_SB_RA(nCond, nRatings, fill::zeros);
    mat p_SB_RB(nCond, nRatings, fill::zeros);

    RCEIntegrand f;
    double err_est;
    int err_code;
    const double tol = 1e-8;
    const int max_subdiv = 100;

    for (int j = 0; j < nCond; j++) {
        for (int i = 0; i < nRatings; i++) {
            // P_SB_RB: stimulus B, response B
            if (N_SB_RB(j, i) > 0) {
                f.set_params(ds(j) / 2.0, theta, sigma);
                p_SB_RB(j, i) = integrate(f, c_RB[i], c_RB[i+1], err_est, err_code, max_subdiv, tol, tol);
            }

            // P_SA_RB: stimulus A, response B
            if (N_SA_RB(j, i) > 0) {
                f.set_params(0.0, theta + ds(j) / 2.0, sigma);
                p_SA_RB(j, i) = integrate(f, c_RB[i], c_RB[i+1], err_est, err_code, max_subdiv, tol, tol);
            }

            // P_SB_RA: stimulus B, response A
            if (N_SB_RA(j, i) > 0) {
                f.set_params(0.0, ds(j) / 2.0 - theta, sigma);
                p_SB_RA(j, i) = integrate(f, -c_RA[i+1], -c_RA[i], err_est, err_code, max_subdiv, tol, tol);
            }

            // P_SA_RA: stimulus A, response A
            if (N_SA_RA(j, i) > 0) {
                f.set_params(ds(j) / 2.0, -theta, sigma);
                p_SA_RA(j, i) = integrate(f, -c_RA[i+1], -c_RA[i], err_est, err_code, max_subdiv, tol, tol);
            }
        }
    }

    return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB,
                          N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB);
}


double ll_RCE_regression(const vec& p, const RegressionData& dat) {
    int n_meta = 0;
    
    auto calc_prob = [](bool resp_A, double loc, double theta, 
                        double lower_bound, double upper_bound, 
                        double d, const std::vector<double>& m) {
        
        double sigma = std::sqrt(0.5);
        bool is_stimA = (loc < 0.0);

        double adj_upper = (resp_A && upper_bound == theta) ? arma::datum::inf : upper_bound;
        double adj_lower = (!resp_A && lower_bound == theta) ? -arma::datum::inf : lower_bound;
        
        RCEIntegrand f;
        double err_est; 
        int err_code;
        const double tol = 1e-8;
        const int max_subdiv = 100;
        
        if (resp_A) {
            // RCE logic maps RA integration from -upper_bound to -lower_bound
            if (is_stimA) f.set_params(d / 2.0, -theta, sigma);      // P_SA_RA
            else          f.set_params(0.0, d / 2.0 - theta, sigma); // P_SB_RA
            
            return integrate(f, -adj_upper, -lower_bound, err_est, err_code, max_subdiv, tol, tol);
            
        } else {
            // RCE logic maps RB integration from lower_bound to upper_bound
            if (is_stimA) f.set_params(0.0, theta + d / 2.0, sigma); // P_SA_RB
            else          f.set_params(d / 2.0, theta, sigma);       // P_SB_RB
            
            return integrate(f, adj_lower, upper_bound, err_est, err_code, max_subdiv, tol, tol);
        }
    };
    
    return compute_regression_negLogL(p, dat, n_meta, calc_prob);
}