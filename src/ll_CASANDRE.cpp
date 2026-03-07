// loglikelihood funtion of the CASANDRE model (Boundy-Singer et al., 2022, Nat
// Hum Behav)
#include "gauss_legendre.h"
#include "likelihoods_func.h"
#include "utils.h"

// [[Rcpp::export]]
double ll_CAS_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                  const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                  const arma::mat &N_SB_RB, int nRatings, int nCond) {

  const arma::vec ds = compute_sensitivity(p, nCond);
  const arma::vec locA = -ds / 2.0;
  const arma::vec locB = ds / 2.0;
  const double theta = p(nCond + nRatings - 1);
  const double sigma = std::exp(p(nCond + nRatings * 2 - 1));

  const double sigma_sq = sigma * sigma;
  const double meanlog = std::log(1.0 / std::sqrt(1 + sigma_sq));
  const double sdlog = std::sqrt(std::log(1 + sigma_sq));

  arma::vec c_RA(nRatings + 1);
  arma::vec c_RB(nRatings + 1);

  c_RA(0) = -arma::datum::inf;
  c_RA(nRatings) = 0;

  c_RB(0) = 0;
  c_RB(nRatings) = arma::datum::inf;

  if (nRatings > 1) {
    const arma::vec p_ra = p.subvec(nCond, nCond + nRatings - 2);
    c_RA.subvec(1, nRatings - 1) =
        -arma::reverse(arma::cumsum(arma::exp(p_ra)));

    const arma::vec p_rb = p.subvec(nCond + nRatings, nCond + nRatings * 2 - 2);
    c_RB.subvec(1, nRatings - 1) = arma::cumsum(arma::exp(p_rb));
  }

  // Probability matrices
  arma::mat p_SA_RA(nCond, nRatings);
  arma::mat p_SA_RB(nCond, nRatings);
  arma::mat p_SB_RA(nCond, nRatings);
  arma::mat p_SB_RB(nCond, nRatings);

  const double trunc = 7.0;

  for (int j = 0; j < nCond; j++) {
    const double lB = locB(j), lA = locA(j);
    for (int i = 0; i < nRatings; i++) {
      const double cRB_lo = c_RB(i), cRB_hi = c_RB(i + 1);
      const double cRA_lo = c_RA(i), cRA_hi = c_RA(i + 1);

      // RB (x > theta): plnorm((x-θ)/c, meanlog, sdlog) where c > 0                                                                                                                             
      // RA (x < theta): plnorm(|x-θ|/|c|, meanlog, sdlog) where c < 0 

      auto rb = [&](double x) {
        return plnorm_cpp((x - theta) / cRB_lo, meanlog, sdlog) -
               plnorm_cpp((x - theta) / cRB_hi, meanlog, sdlog);
      };

      auto ra = [&](double x) {
        return plnorm_cpp(std::abs((x - theta) / cRA_hi), meanlog, sdlog) -
               plnorm_cpp((x - theta) / cRA_lo, meanlog, sdlog);
      };

      p_SB_RB(j, i) =
          (N_SB_RB(j, i) > 0)
              ? gl_integrate(theta, lB + trunc, lB, rb)      
              : constants::MIN_P;
      p_SB_RA(j, i) =
          (N_SB_RA(j, i) > 0)
              ? gl_integrate(lB - trunc, theta, lB, ra)
              : constants::MIN_P;
      p_SA_RA(j, i) =
          (N_SA_RA(j, i) > 0)
              ? gl_integrate(lA - trunc, theta, lA, ra)
              : constants::MIN_P;
      p_SA_RB(j, i) =
          (N_SA_RB(j, i) > 0)
              ? gl_integrate(theta, lA + trunc, lA, rb)
              : constants::MIN_P;
    }
  }

  return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB, N_SA_RA, N_SA_RB,
                         N_SB_RA, N_SB_RB);
}