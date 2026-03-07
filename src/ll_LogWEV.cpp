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
  const double w = 1.0 / (1.0 + std::exp(-w_raw));

  // Probability matrices
  arma::mat p_SA_RA(nCond, nRatings);
  arma::mat p_SA_RB(nCond, nRatings);
  arma::mat p_SB_RA(nCond, nRatings);
  arma::mat p_SB_RB(nCond, nRatings);

  const double trunc = 7.0;

  for (int j = 0; j < nCond; ++j) {
    const double ds_j = ds(j);
    const double lB = locB(j), lA = locA(j);
    for (int i = 0; i < nRatings; ++i) {
      double meanlog;
      const int i_rev = nRatings - 1 - i;
      const double cRA_lo = c_RA(i), cRA_hi = c_RA(i + 1);
      const double cRB_lo = c_RB(i), cRB_hi = c_RB(i + 1);

      auto rb = [&](double x) {
        meanlog = (1.0 - w) * x + ds_j * w;
        return plnorm_cpp(cRB_hi, meanlog, sigma) - plnorm_cpp(cRB_lo, meanlog, sigma);
      };

      auto ra = [&](double x) {
        meanlog = -(1.0 - w) * x + ds_j * w;
        return plnorm_cpp(-cRA_hi, meanlog, sigma) - plnorm_cpp(-cRA_lo, meanlog, sigma);
      };

      p_SB_RB(j, i) =
          N_SB_RB(j, i) > 0
              ?
              // P_SB_RB: stimulus B, response B - integrate [theta, Inf)
              // outer(1:nCond, 1:nRatings, P_SBRB) - normal indexing
              gl_integrate(theta, lB + trunc, lB, rb)
              : constants::MIN_P;
      p_SB_RA(j, i_rev) =
          (N_SB_RA(j, i_rev) > 0)
              ?
              // P_SB_RA: stimulus B, response A - integrate (-Inf, theta]
              // R: outer(1:nCond, nRatings:1, P_SBRA) - REVERSED
              gl_integrate(lB - trunc, theta, lB, ra)
              : constants::MIN_P;
      p_SA_RB(j, i) =
          (N_SA_RB(j, i) > 0)
              ?
              // P_SA_RB: stimulus A, response B - integrate [theta, Inf)
              // outer(1:nCond, 1:nRatings, P_SARB) - normal indexing
              gl_integrate(theta, lA + trunc, lA, rb)
              : constants::MIN_P;
      p_SA_RA(j, i_rev) =
          (N_SA_RA(j, i_rev) > 0)
              ?
              // P_SA_RA: stimulus A, response A - integrate (-Inf, theta]
              // R: outer(1:nCond, nRatings:1, P_SARA) - REVERSED
              gl_integrate(lA - trunc, theta, lA, ra)
              : constants::MIN_P;
    }
  }

  return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB, N_SA_RA, N_SA_RB,
                         N_SB_RA, N_SB_RB);
}