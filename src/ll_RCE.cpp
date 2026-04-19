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

    double negLogL = 0.0;

    for (int j = 0; j < nCond; j++) {
        const double half_ds = ds(j) / 2.0;
        for (int i = 0; i < nRatings; i++) {
            if (N_SB_RB(j, i) > 0)
                negLogL -= N_SB_RB(j, i) * clamped_log(
                    RCE_cell(half_ds, theta, sigma, c_RB(i), c_RB(i + 1)));
            if (N_SA_RB(j, i) > 0)
                negLogL -= N_SA_RB(j, i) * clamped_log(
                    RCE_cell(0.0, theta + half_ds, sigma, c_RB(i), c_RB(i + 1)));
            if (N_SB_RA(j, i) > 0)
                negLogL -= N_SB_RA(j, i) * clamped_log(
                    RCE_cell(0.0, half_ds - theta, sigma, -c_RA(i + 1), -c_RA(i)));
            if (N_SA_RA(j, i) > 0)
                negLogL -= N_SA_RA(j, i) * clamped_log(
                    RCE_cell(half_ds, -theta, sigma, -c_RA(i + 1), -c_RA(i)));
        }
    }
    return negLogL;
}

double ll_RCE_regression(const vec& p, const RegressionData& dat) {
    int n_meta = 0;

    auto calc_prob = [](bool resp_A, double loc, double theta,
                        double lower_bound, double upper_bound, double d,
                        const std::vector<double>& m) {
        double sigma = std::sqrt(0.5);
        bool is_stimA = (loc < 0.0);

        // RCE logic rating criteria might extend past theta(widen to +/- inf) due to accumulator comparison
        if(resp_A && upper_bound == theta) {
            upper_bound = arma::datum::inf;
        } else if (!resp_A && lower_bound == theta) {
            lower_bound = -arma::datum::inf;
        }

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