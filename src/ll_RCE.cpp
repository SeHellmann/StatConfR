#include "gauss_legendre.h"
#include "likelihoods_func.h"
#include "utils.h"

using namespace arma;
using namespace Rcpp;

inline double RCE_cell(double mu1, double mu2, double sigma, double c_lo,
                       double c_hi) {
    constexpr double TRUNC = 7.0;
    double lo = std::isinf(c_lo) ? -TRUNC : (c_lo - mu1) / sigma;
    double hi = std::isinf(c_hi) ? TRUNC : (c_hi - mu1) / sigma;
    auto integrand = [&](double u) {
        return pnorm_cpp(u * sigma + mu1, mu2, sigma);
    };
    return gl_integrate(lo, hi, 0.0, integrand);
}

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
        vec steps_ra =
            reverse(cumsum(exp(p.subvec(nCond, nCond + nRatings - 3))));
        c_RA.subvec(1, nRatings - 2) = anchor_ra - steps_ra;
    }

    // RB Criteria
    vec c_RB(nRatings + 1);
    c_RB(0) = -datum::inf;
    double anchor_rb = p(nCond + nRatings);
    c_RB(1) = anchor_rb;
    c_RB(nRatings) = datum::inf;
    if (nRatings > 2) {
        vec steps_rb = cumsum(
            exp(p.subvec(nCond + nRatings + 1, nCond + 2 * nRatings - 2)));
        c_RB.subvec(2, nRatings - 1) = anchor_rb + steps_rb;
    }

    const double sigma = std::sqrt(0.5);

    mat p_SA_RA(nCond, nRatings, fill::zeros);
    mat p_SA_RB(nCond, nRatings, fill::zeros);
    mat p_SB_RA(nCond, nRatings, fill::zeros);
    mat p_SB_RB(nCond, nRatings, fill::zeros);

    for (int j = 0; j < nCond; j++) {
        for (int i = 0; i < nRatings; i++) {
            p_SB_RB(j, i) =
                (N_SB_RB(j, i) > 0)
                    ? RCE_cell(ds(j) / 2.0, theta, sigma, c_RB(i), c_RB(i + 1))
                    : constants::MIN_P;
            p_SA_RB(j, i) = (N_SA_RB(j, i) > 0)
                                ? RCE_cell(0.0, theta + ds(j) / 2.0, sigma,
                                           c_RB(i), c_RB(i + 1))
                                : constants::MIN_P;
            p_SB_RA(j, i) = (N_SB_RA(j, i) > 0)
                                ? RCE_cell(0.0, ds(j) / 2.0 - theta, sigma,
                                           -c_RA(i + 1), -c_RA(i))
                                : constants::MIN_P;
            p_SA_RA(j, i) = (N_SA_RA(j, i) > 0)
                                ? RCE_cell(ds(j) / 2.0, -theta, sigma,
                                           -c_RA(i + 1), -c_RA(i))
                                : constants::MIN_P;
        }
    }

    return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB, N_SA_RA, N_SA_RB,
                           N_SB_RA, N_SB_RB);
}

double ll_RCE_regression(const vec& p, const RegressionData& dat) {
    int n_meta = 0;

    auto calc_prob = [](bool resp_A, double loc, double theta,
                        double lower_bound, double upper_bound, double d,
                        const std::vector<double>& m) {
        double sigma = std::sqrt(0.5);
        bool is_stimA = (loc < 0.0);

        if (resp_A) {
            // RCE logic maps RA integration from -upper_bound to -lower_bound
            double mu1 = is_stimA ? d / 2.0 : 0.0;
            double mu2 = is_stimA ? -theta : d / 2.0 - theta;
            return RCE_cell(mu1, mu2, sigma, -upper_bound, -lower_bound);
        } else {
            // RCE logic maps RB integration from lower_bound to upper_bound
            double mu1 = is_stimA ? 0.0 : d / 2.0;
            double mu2 = is_stimA ? theta + d / 2.0 : theta;
            return RCE_cell(mu1, mu2, sigma, lower_bound, upper_bound);
        }
    };

    return compute_regression_negLogL(p, dat, n_meta, calc_prob);
}