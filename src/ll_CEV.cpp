#include "gauss_legendre.h"
#include "likelihoods_func.h"
#include "utils.h"

using namespace arma;
using namespace Rcpp;

inline double CEV_cell(double loc, double theta, double c_lo, double c_hi,
                       double w, double ds_j, double sigma, bool resp_A,
                       double trunc = 7.0) {
    auto integrand = [&](double x) {
        double mean_val = (1.0 - w) * x + (resp_A ? -w : w) * ds_j;
        return pnorm_cpp(c_hi, mean_val, sigma) -
               pnorm_cpp(c_lo, mean_val, sigma);
    };
    return resp_A ? gl_integrate(loc - trunc, theta, loc, integrand)
                  : gl_integrate(theta, loc + trunc, loc, integrand);
}

double ll_CEV_cpp(const arma::vec& p, const ModelData& dat) {
    const arma::mat& N_SA_RA = dat.N_SA_RA;
    const arma::mat& N_SA_RB = dat.N_SA_RB;
    const arma::mat& N_SB_RA = dat.N_SB_RA;
    const arma::mat& N_SB_RB = dat.N_SB_RB;
    int nRatings = dat.nRatings;
    int nCond = dat.nCond;

    const arma::vec ds = compute_sensitivity(p, nCond);
    const arma::vec locA = -ds / 2.0;
    const arma::vec locB = ds / 2.0;
    const double theta = p(nCond + nRatings - 1);

    const double anchor_ra = p(nCond + nRatings - 2);
    const double anchor_rb = p(nCond + nRatings);

    arma::vec c_RA(nRatings + 1);
    arma::vec c_RB(nRatings + 1);

    c_RA(0) = -arma::datum::inf;
    c_RA(nRatings - 1) = anchor_ra;
    c_RA(nRatings) = arma::datum::inf;

    c_RB(0) = -arma::datum::inf;
    c_RB(1) = anchor_rb;
    c_RB(nRatings) = arma::datum::inf;

    if (nRatings > 2) {
        const arma::vec p_ra = p.subvec(nCond, nCond + nRatings - 3);
        c_RA.subvec(1, nRatings - 2) =
            anchor_ra - arma::reverse(arma::cumsum(arma::exp(p_ra)));

        const arma::vec p_rb =
            p.subvec(nCond + nRatings + 1, nCond + nRatings * 2 - 2);
        c_RB.subvec(2, nRatings - 1) =
            anchor_rb + arma::cumsum(arma::exp(p_rb));
    }

    const double sigma = std::exp(p(nCond + nRatings * 2 - 1));
    const double w_raw = p(nCond + nRatings * 2);
    const double w = 1.0 / (1.0 + std::exp(-w_raw));

    double negLogL = 0.0;

    for (int j = 0; j < nCond; j++) {
        const double ds_j = ds(j);
        const double lB = locB(j), lA = locA(j);

        for (int i = 0; i < nRatings; i++) {
            const double cRB_lo = c_RB(i), cRB_hi = c_RB(i + 1);
            const double cRA_lo = c_RA(i), cRA_hi = c_RA(i + 1);

            if (N_SB_RB(j, i) > 0)
                negLogL -= N_SB_RB(j, i) * clamped_log(
                    CEV_cell(lB, theta, cRB_lo, cRB_hi, w, ds_j, sigma, false));
            if (N_SB_RA(j, i) > 0)
                negLogL -= N_SB_RA(j, i) * clamped_log(
                    CEV_cell(lB, theta, cRA_lo, cRA_hi, w, ds_j, sigma, true));
            if (N_SA_RA(j, i) > 0)
                negLogL -= N_SA_RA(j, i) * clamped_log(
                    CEV_cell(lA, theta, cRA_lo, cRA_hi, w, ds_j, sigma, true));
            if (N_SA_RB(j, i) > 0)
                negLogL -= N_SA_RB(j, i) * clamped_log(
                    CEV_cell(lA, theta, cRB_lo, cRB_hi, w, ds_j, sigma, false));
        }
    }
    return negLogL;
}

double ll_CEV_regression(const vec& p, const RegressionData& dat) {
    int n_meta = 2;

    auto calc_prob = [](bool resp_A, double loc, double theta,
                        double lower_bound, double upper_bound, double d,
                        const std::vector<double>& m) {
        double sigma = std::exp(m[0]);
        double w = 1.0 / (1.0 + std::exp(-m[1]));

        double adj_upper =
            (resp_A && upper_bound == theta) ? arma::datum::inf : upper_bound;
        double adj_lower =
            (!resp_A && lower_bound == theta) ? -arma::datum::inf : lower_bound;

        return CEV_cell(loc, theta, adj_lower, adj_upper, w, d, sigma, resp_A);
    };

    return compute_regression_negLogL(p, dat, n_meta, calc_prob);
}
