// [[Rcpp::depends(RcppArmadillo)]]
#include "likelihoods_func.h"
#include "utils.h"

using namespace arma;
using namespace Rcpp;

// Independent Gaussian - 2 channel

double ll_2Chan_cpp(const arma::vec& p, const ModelData& dat) {
    const arma::mat& N_SA_RA = dat.N_SA_RA;
    const arma::mat& N_SA_RB = dat.N_SA_RB;
    const arma::mat& N_SB_RA = dat.N_SB_RA;
    const arma::mat& N_SB_RB = dat.N_SB_RB;
    int nRatings = dat.nRatings;
    int nCond = dat.nCond;

    const arma::vec ds = arma::exp(p.subvec(0, nCond - 1));
    const arma::vec locA1 = -ds / 2.0;
    const arma::vec locB1 = ds / 2.0;

    const arma::vec metads = ds * std::exp(p(nCond + nRatings * 2 - 1));
    const arma::vec locA2 = - metads / 2.0;
    const arma::vec locB2 =   metads / 2.0;

    const double theta = p(nCond + nRatings - 1);

    // response A criterion
    const double theta_prev = p(nCond + nRatings - 2);

    arma::vec c_RA(nRatings + 1);
    c_RA(0) = -arma::datum::inf;
    c_RA(nRatings - 1) = theta_prev;
    c_RA(nRatings) = arma::datum::inf;

    // response B criterion
    double theta_next = p(nCond + nRatings);

    arma::vec c_RB(nRatings + 1);
    c_RB(0) = -arma::datum::inf;
    c_RB(1) = theta_next;
    c_RB(nRatings) = arma::datum::inf;

    if (nRatings > 2) {
        arma::vec p_1 = arma::reverse(arma::cumsum(arma::exp(p.subvec(nCond, nCond + nRatings - 3))));
        arma::vec p_2 = arma::cumsum(arma::exp(p.subvec(nCond + nRatings + 1, nCond + nRatings * 2 - 2)));

        c_RA.subvec(1, nRatings - 2) = theta_prev - p_1;
        c_RB.subvec(2, nRatings - 1) = theta_next + p_2;
    }

    double negLogL = 0.0;

    // prev CDF values and channel-1 probabilities
    // naming: {criterion}_{location} — RA_A2 means c_RA with locA2
    std::vector<double> prev_RA_A2(nCond), prev_RA_B2(nCond),
                        prev_RB_A2(nCond), prev_RB_B2(nCond);
    std::vector<double> P_sara(nCond), P_sarb(nCond),
                        P_sbra(nCond), P_sbrb(nCond);

    for (int j = 0; j < nCond; ++j) {
        prev_RA_A2[j] = normcdf_cpp(c_RA(0) - locA2(j));
        prev_RA_B2[j] = normcdf_cpp(c_RA(0) - locB2(j));
        prev_RB_A2[j] = normcdf_cpp(c_RB(0) - locA2(j));
        prev_RB_B2[j] = normcdf_cpp(c_RB(0) - locB2(j));

        P_sara[j] = normcdf_cpp(theta - locA1(j));
        P_sarb[j] = 1.0 - P_sara[j];
        P_sbra[j] = normcdf_cpp(theta - locB1(j));
        P_sbrb[j] = 1.0 - P_sbra[j];
    }

    for (int i = 0; i < nRatings; ++i) {
        for (int j = 0; j < nCond; ++j) {
            double curr_RA_A2 = normcdf_cpp(c_RA(i + 1) - locA2(j));
            double curr_RA_B2 = normcdf_cpp(c_RA(i + 1) - locB2(j));
            double curr_RB_A2 = normcdf_cpp(c_RB(i + 1) - locA2(j));
            double curr_RB_B2 = normcdf_cpp(c_RB(i + 1) - locB2(j));

            double p_sa_ra = std::max(P_sara[j] * (curr_RA_A2 - prev_RA_A2[j]), constants::MIN_P);
            double p_sa_rb = std::max(P_sarb[j] * (curr_RB_A2 - prev_RB_A2[j]), constants::MIN_P);
            double p_sb_ra = std::max(P_sbra[j] * (curr_RA_B2 - prev_RA_B2[j]), constants::MIN_P);
            double p_sb_rb = std::max(P_sbrb[j] * (curr_RB_B2 - prev_RB_B2[j]), constants::MIN_P);

            negLogL -= N_SA_RA(j, i) * std::log(p_sa_ra);
            negLogL -= N_SA_RB(j, i) * std::log(p_sa_rb);
            negLogL -= N_SB_RA(j, i) * std::log(p_sb_ra);
            negLogL -= N_SB_RB(j, i) * std::log(p_sb_rb);

            prev_RA_A2[j] = curr_RA_A2;
            prev_RA_B2[j] = curr_RA_B2;
            prev_RB_A2[j] = curr_RB_A2;
            prev_RB_B2[j] = curr_RB_B2;
        }
    }

    return negLogL;
}


double ll_2Chan_regression(const arma::vec& p, const RegressionData& dat) {
    int n_meta = 1;
    
    auto calc_prob = [](bool resp_A, double loc, double theta, 
                        double lower_bound, double upper_bound, 
                        double d, const std::vector<double>& m) {
        
        // Reverse log-link for meta parameter
        double m_ratio = std::exp(m[0]);
        double loc2 = loc * m_ratio; 
        
        double adj_upper = (resp_A && upper_bound == theta) ? arma::datum::inf : upper_bound;
        double adj_lower = (!resp_A && lower_bound == theta) ? -arma::datum::inf : lower_bound;
        
        double p_chan1 = resp_A ? normcdf_cpp(theta - loc) : (1.0 - normcdf_cpp(theta - loc));
        
        double p_chan2_upper = normcdf_cpp(adj_upper - loc2);
        double p_chan2_lower = normcdf_cpp(adj_lower - loc2);
        double p_chan2 = std::max(p_chan2_upper - p_chan2_lower, 0.0);
        
        return p_chan1 * p_chan2;
    };
    
    return compute_regression_negLogL(p, dat, n_meta, calc_prob);
}