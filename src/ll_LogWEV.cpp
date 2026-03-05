#include "gauss_legendre.h"
#include "likelihoods_func.h"
#include "utils.h"

// [[Rcpp::export]]
double ll_LogWEV_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                     const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                     const arma::mat &N_SB_RB, int nRatings, int nCond) {

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
  // const double w = std::exp(w_raw) / (1.0 + std::exp(w_raw));  // log
  // transform to (0,1)
  const double w = 1.0 / (1.0 + std::exp(-w_raw)); // avoid overflow

  // Probability matrices
  arma::mat p_SA_RA(nCond, nRatings);
  arma::mat p_SA_RB(nCond, nRatings);
  arma::mat p_SB_RA(nCond, nRatings);
  arma::mat p_SB_RB(nCond, nRatings);

  // Compute all integrals
  for (int j = 0; j < nCond; ++j) {
    double ds_j = ds(j);
    for (int i = 0; i < nRatings; ++i) {
      double meanlog;
      int i_rev = nRatings - 1 - i;
      p_SB_RB(j, i) =
          N_SB_RB(j, i) > 0
              ?
              // P_SB_RB: stimulus B, response B - integrate [theta, Inf)
              // outer(1:nCond, 1:nRatings, P_SBRB) - normal indexing
              gl_integrate(locB(j), theta, true,
                           [&](double x) {
                             meanlog = (1.0 - w) * x + ds_j * w;
                             return plnorm_cpp(c_RB(i + 1), meanlog, sigma) -
                                    plnorm_cpp(c_RB(i), meanlog, sigma);
                           })
              : p_SB_RB(j, i) = constants::MIN_P;
      p_SB_RA(j, i_rev) =
          (N_SB_RA(j, i_rev) > 0)
              ?
              // P_SB_RA: stimulus B, response A - integrate (-Inf, theta]
              // R: outer(1:nCond, nRatings:1, P_SBRA) - REVERSED
              gl_integrate(locB(j), theta, false,
                           [&](double x) {
                             meanlog = -(1.0 - w) * x + ds_j * w;
                             return plnorm_cpp(-c_RA(i + 1), meanlog, sigma) -
                                    plnorm_cpp(-c_RA(i), meanlog, sigma);
                           })
              : constants::MIN_P;
      p_SA_RB(j, i) =
          (N_SA_RB(j, i) > 0)
              ?
              // P_SA_RB: stimulus A, response B - integrate [theta, Inf)
              // outer(1:nCond, 1:nRatings, P_SARB) - normal indexing
              gl_integrate(locA(j), theta, true,
                           [&](double x) {
                             meanlog = (1.0 - w) * x + ds_j * w;
                             return plnorm_cpp(c_RB(i + 1), meanlog, sigma) -
                                    plnorm_cpp(c_RB(i), meanlog, sigma);
                           })
              : constants::MIN_P;
      p_SA_RA(j, i_rev) =
          (N_SA_RA(j, i_rev) > 0)
              ?
              // P_SA_RA: stimulus A, response A - integrate (-Inf, theta]
              // R: outer(1:nCond, nRatings:1, P_SARA) - REVERSED
              gl_integrate(locA(j), theta, false,
                           [&](double x) {
                             meanlog = -(1.0 - w) * x + ds_j * w;
                             return plnorm_cpp(-c_RA(i + 1), meanlog, sigma) -
                                    plnorm_cpp(-c_RA(i), meanlog, sigma);
                           })
              : constants::MIN_P;
    }
  }

  return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB, N_SA_RA, N_SA_RB,
                         N_SB_RA, N_SB_RB);
}