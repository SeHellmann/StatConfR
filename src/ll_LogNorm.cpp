#include "gauss_legendre.h"
#include "likelihoods_func.h"
#include "utils.h"

// [[Rcpp::export]]
double ll_LogNorm_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                      const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                      const arma::mat &N_SB_RB, int nRatings, int nCond) {

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
    mu_cB = theta + arma::cumsum(arma::exp(
                        p.subvec(nCond + nRatings, nCond + nRatings * 2 - 2)));
  }
  // convert to the location parameter of the latent lognormal distribution
  arma::vec loc_RA(nRatings + 1);
  arma::vec loc_RB(nRatings + 1);

  // loc_RA <-  c(Inf,log(abs(mu_cA - theta)) - .5*sigma^2, -Inf) # the order
  // here is REVERSED!!! loc_RB <- c(-Inf, log(mu_cB - theta) - .5*sigma^2, Inf)
  const double sigma2_div2 = 0.5 * sigma * sigma;
  loc_RA(0) = arma::datum::inf;
  loc_RA(nRatings) = -arma::datum::inf;

  loc_RB(0) = -arma::datum::inf;
  loc_RB(nRatings) = arma::datum::inf;

  for (int i = 0; i < nRatings - 1; i++) {
    loc_RA(i + 1) = std::log(std::abs(mu_cA(i) - theta)) - sigma2_div2;
    loc_RB(i + 1) = std::log(mu_cB(i) - theta) - sigma2_div2;
  }

  // Probability matrices
  arma::mat p_SA_RA(nCond, nRatings);
  arma::mat p_SA_RB(nCond, nRatings);
  arma::mat p_SB_RA(nCond, nRatings);
  arma::mat p_SB_RB(nCond, nRatings);

  for (int j = 0; j < nCond; j++) {
    for (int i = 0; i < nRatings; i++) {
      double diff;
      p_SB_RB(j, i) =
          (N_SB_RB(j, i) > 0)
              ? gl_integrate(locB(j), theta, true,
                             [&](double x) {
                               diff = (x - theta);
                               return plnorm_cpp(diff, loc_RB(i), sigma) -
                                      plnorm_cpp(diff, loc_RB(i + 1), sigma);
                             })
              : constants::MIN_P;
      p_SB_RA(j, i) =
          (N_SB_RA(j, i) > 0)
              ? gl_integrate(locB(j), theta, false,
                             [&](double x) {
                               diff = theta - x;
                               return plnorm_cpp(diff, loc_RA(i + 1), sigma) -
                                      plnorm_cpp(diff, loc_RA(i), sigma);
                             })
              : constants::MIN_P;
      p_SA_RA(j, i) =
          (N_SA_RA(j, i) > 0)
              ? gl_integrate(locA(j), theta, false,
                             [&](double x) {
                               diff = theta - x;
                               return plnorm_cpp(diff, loc_RA(i + 1), sigma) -
                                      plnorm_cpp(diff, loc_RA(i), sigma);
                             })
              : constants::MIN_P;
      p_SA_RB(j, i) =
          (N_SA_RB(j, i) > 0)
              ? gl_integrate(locA(j), theta, true,
                             [&](double x) {
                               diff = (x - theta);
                               return plnorm_cpp(diff, loc_RB(i), sigma) -
                                      plnorm_cpp(diff, loc_RB(i + 1), sigma);
                             })
              : constants::MIN_P;
    }
  }
  return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB, N_SA_RA, N_SA_RB,
                         N_SB_RA, N_SB_RB);
}
