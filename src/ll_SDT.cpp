// [[Rcpp::depends(RcppArmadillo)]]
#include "likelihoods_func.h"
#include "utils.h"

// [[Rcpp::export]]
double ll_SDT_cpp(const arma::vec& p, const arma::mat& N_SA_RA, const arma::mat& N_SA_RB,
     const arma::mat& N_SB_RA,const arma::mat& N_SB_RB, int nRatings, int nCond) {

    const arma::vec ds = compute_sensitivity(p, nCond);
    const arma::vec locA = -ds / 2.0;
    const arma::vec locB = ds / 2.0;
    // Index of theta (decision criterion) in p
    // R: theta <- p[nCond+nRatings]
    const double theta = p(nCond + nRatings - 1);

    // build criterion vectors
    arma::vec c_RA(nRatings + 1);
    arma::vec c_RB(nRatings + 1);

    c_RA(0) = -arma::datum::inf;
    c_RA(nRatings) = theta;
        
    c_RB(0) = theta;
    c_RB(nRatings) = arma::datum::inf;

    if (nRatings > 1) {
        arma::vec p_1 = arma::reverse(arma::cumsum(arma::exp(p.subvec(nCond, nCond + nRatings - 2))));
        arma::vec p_2 = arma::cumsum(arma::exp(p.subvec(nCond + nRatings, p.n_elem - 1)));
        
        c_RA.subvec(1, nRatings - 1) = theta - p_1;
        c_RB.subvec(1, nRatings - 1) = theta + p_2;
    }

    // P(rating i | stimulus, response) = pnorm(c[i], loc)

    double negLogL = 0.0;
    std::vector<double> prev_RA_A(nCond), prev_RA_B(nCond), prev_RB_A(nCond), prev_RB_B(nCond);
    
    for (int j = 0; j < nCond; j++) {
        prev_RA_A[j] = normcdf_cpp(c_RA(0) - locA(j));
        prev_RA_B[j] = normcdf_cpp(c_RA(0) - locB(j));
        prev_RB_A[j] = normcdf_cpp(c_RB(0) - locA(j));
        prev_RB_B[j] = normcdf_cpp(c_RB(0) - locB(j));
    }

    for (int i = 0; i < nRatings; i++) {
        for (int j = 0; j < nCond; j++) {
            double curr_RA_A = normcdf_cpp(c_RA(i + 1) - locA(j));
            double curr_RA_B = normcdf_cpp(c_RA(i + 1) - locB(j));
            double curr_RB_A = normcdf_cpp(c_RB(i + 1) - locA(j));
            double curr_RB_B = normcdf_cpp(c_RB(i + 1) - locB(j));

            double p_sa_ra = std::max(curr_RA_A - prev_RA_A[j], constants::MIN_P);
            double p_sa_rb = std::max(curr_RB_A - prev_RB_A[j], constants::MIN_P);
            double p_sb_ra = std::max(curr_RA_B - prev_RA_B[j], constants::MIN_P);
            double p_sb_rb = std::max(curr_RB_B - prev_RB_B[j], constants::MIN_P);

            negLogL -= N_SA_RA(j, i) * std::log(p_sa_ra);
            negLogL -= N_SA_RB(j, i) * std::log(p_sa_rb);
            negLogL -= N_SB_RA(j, i) * std::log(p_sb_ra);
            negLogL -= N_SB_RB(j, i) * std::log(p_sb_rb);

            prev_RA_A[j] = curr_RA_A;
            prev_RA_B[j] = curr_RA_B;
            prev_RB_A[j] = curr_RB_A;
            prev_RB_B[j] = curr_RB_B;
        }
    }
    return negLogL;
}
