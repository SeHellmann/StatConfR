#include "gauss_legendre.h"
#include "likelihoods_func.h"
#include "utils.h"

using namespace arma;
using namespace Rcpp;

double inline logNorm_cell(const double loc, const double theta,
                           const double loc_lo, const double loc_hi,
                           const double sigma, const bool resp_A,
                           double trunc = 7.0) {
    if (resp_A) {
        auto ra = [&](double x) {
            const double diff = theta - x;
            return plnorm_cpp(diff, loc_hi, sigma) -
                   plnorm_cpp(diff, loc_lo, sigma);
        };
        return gl_integrate(loc - trunc, theta, loc, ra);
    } else {
        auto rb = [&](double x) {
            const double diff = x - theta;
            return plnorm_cpp(diff, loc_lo, sigma) -
                   plnorm_cpp(diff, loc_hi, sigma);
        };
        return gl_integrate(theta, loc + trunc, loc, rb);
    }
}

double ll_LogNorm_cpp(const arma::vec& p, const ModelData& dat) {
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

    // sigma needs to be bounded between 0 and Inf
    // see whether the results converge.
    const double sigma = std::exp(p(nCond + nRatings * 2 - 1));

    arma::vec mu_cA, mu_cB;
    // average placement of the confidence criteria
    if (nRatings > 1) {
        mu_cA = theta + arma::reverse(arma::cumsum(
                            arma::exp(p.subvec(nCond, nCond + nRatings - 2))));
        mu_cB = theta + arma::cumsum(arma::exp(p.subvec(
                            nCond + nRatings, nCond + nRatings * 2 - 2)));
    }
    // convert to the location parameter of the latent lognormal distribution
    arma::vec loc_RA(nRatings + 1);
    arma::vec loc_RB(nRatings + 1);

    const double sigma2_div2 = 0.5 * sigma * sigma;
    loc_RA(0) = arma::datum::inf;
    loc_RA(nRatings) = -arma::datum::inf;

    loc_RB(0) = -arma::datum::inf;
    loc_RB(nRatings) = arma::datum::inf;

    for (int i = 0; i < nRatings - 1; i++) {
        loc_RA(i + 1) = std::log(std::abs(mu_cA(i) - theta)) - sigma2_div2;
        loc_RB(i + 1) = std::log(mu_cB(i) - theta) - sigma2_div2;
    }

    double negLogL = 0.0;

    for (int j = 0; j < nCond; j++) {
        const double lB = locB(j), lA = locA(j);
        for (int i = 0; i < nRatings; i++) {
            const double locRB_lo = loc_RB(i), locRB_hi = loc_RB(i + 1);
            const double locRA_lo = loc_RA(i), locRA_hi = loc_RA(i + 1);

            if (N_SB_RB(j, i) > 0)
                negLogL -= N_SB_RB(j, i) * clamped_log(
                    logNorm_cell(lB, theta, locRB_lo, locRB_hi, sigma, false));
            if (N_SB_RA(j, i) > 0)
                negLogL -= N_SB_RA(j, i) * clamped_log(
                    logNorm_cell(lB, theta, locRA_lo, locRA_hi, sigma, true));
            if (N_SA_RA(j, i) > 0)
                negLogL -= N_SA_RA(j, i) * clamped_log(
                    logNorm_cell(lA, theta, locRA_lo, locRA_hi, sigma, true));
            if (N_SA_RB(j, i) > 0)
                negLogL -= N_SA_RB(j, i) * clamped_log(
                    logNorm_cell(lA, theta, locRB_lo, locRB_hi, sigma, false));
        }
    }
    return negLogL;
}

double ll_LogNorm_regression(const vec& p, const RegressionData& dat) {
    int n_meta = 1;

    auto calc_prob = [](bool resp_A, double loc, double theta,
                        double lower_bound, double upper_bound, double d,
                        const std::vector<double>& m) {
        double sigma = std::exp(m[0]);
        double sigma2_div2 = 0.5 * sigma * sigma;

        double loc_lo, loc_hi;

        if (resp_A) {
            // Distance is (theta - bound).
            loc_lo = std::isinf(lower_bound)
                         ? arma::datum::inf
                         : std::log(theta - lower_bound) - sigma2_div2;
            loc_hi = (upper_bound == theta)
                         ? -arma::datum::inf
                         : std::log(theta - upper_bound) - sigma2_div2;
        } else {
            // Distance is (bound - theta).
            loc_lo = (lower_bound == theta)
                         ? -arma::datum::inf
                         : std::log(lower_bound - theta) - sigma2_div2;
            loc_hi = std::isinf(upper_bound)
                         ? arma::datum::inf
                         : std::log(upper_bound - theta) - sigma2_div2;
        }

        return logNorm_cell(loc, theta, loc_lo, loc_hi, sigma, resp_A);
    };

    return compute_regression_negLogL(p, dat, n_meta, calc_prob);
}
