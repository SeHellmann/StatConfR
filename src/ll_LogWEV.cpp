#include "gauss_legendre.h"
#include "likelihoods_func.h"
#include "utils.h"

using namespace arma;
using namespace Rcpp;

inline double LogWEV_cell(double loc, double theta, double c_lo, double c_hi,
                          double w, double ds_j, double sigma, bool resp_A,
                          double trunc = 7.0) {
    if (resp_A) {
        auto ra = [&](double x) {
            double meanlog = -(1.0 - w) * x + ds_j * w;
            return plnorm_cpp(-c_hi, meanlog, sigma) -
                   plnorm_cpp(-c_lo, meanlog, sigma);
        };
        return gl_integrate(loc - trunc, theta, loc, ra);
    } else {
        auto rb = [&](double x) {
            double meanlog = (1.0 - w) * x + ds_j * w;
            return plnorm_cpp(c_hi, meanlog, sigma) -
                   plnorm_cpp(c_lo, meanlog, sigma);
        };
        return gl_integrate(theta, loc + trunc, loc, rb);
    }
}

double ll_LogWEV_cpp(const arma::vec& p, const ModelData& dat) {
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

    arma::vec c_RA(nRatings + 1);
    arma::vec c_RB(nRatings + 1);

    c_RA(0) = 0.0;
    c_RA(nRatings) = -arma::datum::inf;

    c_RB(0) = 0.0;
    c_RB(nRatings) = arma::datum::inf;

    if (nRatings > 1) {
        arma::vec p_1 =
            arma::cumsum(arma::exp(p.subvec(nCond, nCond + nRatings - 2)));
        arma::vec p_2 = arma::cumsum(
            arma::exp(p.subvec(nCond + nRatings, nCond + nRatings * 2 - 2)));
        c_RA.subvec(1, nRatings - 1) = -p_1;
        c_RB.subvec(1, nRatings - 1) = p_2;
    }

    const double sigma = std::exp(p(nCond + nRatings * 2 - 1));
    const double w_raw = p(nCond + nRatings * 2);
    const double w = 1.0 / (1.0 + std::exp(-w_raw));

    double negLogL = 0.0;
    const double trunc = 7.0;

    for (int j = 0; j < nCond; ++j) {
        const double ds_j = ds(j);
        const double lB = locB(j), lA = locA(j);
        for (int i = 0; i < nRatings; ++i) {
            const int i_rev = nRatings - 1 - i;
            const double cRA_lo = c_RA(i), cRA_hi = c_RA(i + 1);
            const double cRB_lo = c_RB(i), cRB_hi = c_RB(i + 1);
            // P_SB_RB: stimulus B, response B - integrate [theta, Inf)
            if (N_SB_RB(j, i) > 0) {
                negLogL -= N_SB_RB(j, i) * clamped_log(
                    LogWEV_cell(lB, theta, cRB_lo, cRB_hi, w, ds_j, sigma, false));
                }
            // P_SB_RA: stimulus B, response A - integrate (-Inf, theta] - REVERSED indexing
            if (N_SB_RA(j, i_rev) > 0) {
                negLogL -= N_SB_RA(j, i_rev) * clamped_log(
                    LogWEV_cell(lB, theta, cRA_lo, cRA_hi, w, ds_j, sigma, true));
                }
            // P_SA_RB: stimulus A, response B - integrate [theta, Inf)
            if (N_SA_RB(j, i) > 0){
                negLogL -= N_SA_RB(j, i) * clamped_log(
                    LogWEV_cell(lA, theta, cRB_lo, cRB_hi, w, ds_j, sigma, false));
                }
            // P_SA_RA: stimulus A, response A - integrate (-Inf, theta] - REVERSED indexing
            if (N_SA_RA(j, i_rev) > 0) {
                negLogL -= N_SA_RA(j, i_rev) * clamped_log(
                    LogWEV_cell(lA, theta, cRA_lo, cRA_hi, w, ds_j, sigma, true));
                }
        }
    }
    return negLogL;
}

double ll_LogWEV_regression(const vec& p, const RegressionData& dat) {
    int n_meta = 2;

    auto calc_prob = [](bool resp_A, double loc, double theta,
                        double lower_bound, double upper_bound, double d,
                        const std::vector<double>& m) {
        // Link functions: sigma is log-linked, w is qlogis-linked
        double sigma = std::exp(m[0]);
        double w = 1.0 / (1.0 + std::exp(-m[1]));

        double c_lo, c_hi;
        if (resp_A) {
            c_lo = upper_bound - theta;
            c_hi = lower_bound - theta;
            // Integrate decision variable from -Inf to theta

        } else {
            c_lo = lower_bound - theta;
            c_hi = upper_bound - theta;
            // Integrate decision variable from theta to Inf
        }
        return LogWEV_cell(loc, theta, c_lo, c_hi, w, d, sigma, resp_A);
    };

    return compute_regression_negLogL(p, dat, n_meta, calc_prob);
}