#include "utils.h"

arma::vec compute_sensitivity(const arma::vec& p, int nCond) {
    return arma::cumsum(arma::exp(p.subvec(0, nCond - 1)));
}

double compute_negLogL(
    const arma::mat& p_SA_RA, const arma::mat& p_SA_RB,
    const arma::mat& p_SB_RA, const arma::mat& p_SB_RB,
    const arma::mat& N_SA_RA, const arma::mat& N_SA_RB,
    const arma::mat& N_SB_RA, const arma::mat& N_SB_RB
) {
    auto clamp_log = [](double x) {
      return (std::isnan(x) || x < constants::MIN_P) ? constants::LOG_MIN_P : (x > 1.0) ? 0.0 : std::log(x);
    };

    arma::mat lp_SA_RA = p_SA_RA; lp_SA_RA.transform(clamp_log);
    arma::mat lp_SA_RB = p_SA_RB; lp_SA_RB.transform(clamp_log);
    arma::mat lp_SB_RA = p_SB_RA; lp_SB_RA.transform(clamp_log);
    arma::mat lp_SB_RB = p_SB_RB; lp_SB_RB.transform(clamp_log);

    return -arma::accu(N_SA_RA % lp_SA_RA +
                       N_SA_RB % lp_SA_RB +
                       N_SB_RA % lp_SB_RA +
                       N_SB_RB % lp_SB_RB);
}
