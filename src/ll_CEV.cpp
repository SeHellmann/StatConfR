#include "gauss_legendre.h"
#include "likelihoods_func.h"
#include "utils.h"

inline double gl_cev(double loc, double c_lo, double c_hi, double w,
                     double ds_j, double sigma, double theta, bool response_B) {
  // beyond SD = 7 contribution from integration negligible
  double a_lim = response_B ? theta : loc - 7.0;
  double b_lim = response_B ? loc + 7.0 : theta;
  double mid = (a_lim + b_lim) / 2.0;
  double half = (b_lim - a_lim) / 2.0;
  double sum = 0.0;
  for (int k = 0; k < GL_ORDER; k++) {
    double x = mid + half * GL_NODES[k];
    double dnorm_val =
        std::exp(-0.5 * (x - loc) * (x - loc)) * constants::INV_SQRT_2PI;
    double mean_val =
        response_B ? (1.0 - w) * x + ds_j * w : (1.0 - w) * x - ds_j * w;
    sum +=
        GL_WEIGHTS[k] * dnorm_val *
        (pnorm_cpp(c_hi, mean_val, sigma) - pnorm_cpp(c_lo, mean_val, sigma));
  }
  return sum * half;
}

// [[Rcpp::export]]
double ll_CEV_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                  const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                  const arma::mat &N_SB_RB, int nRatings, int nCond) {

  const arma::vec ds = compute_sensitivity(p, nCond);
  const arma::vec locA = -ds / 2.0;
  const arma::vec locB = ds / 2.0;
  const double theta = p(nCond + nRatings - 1);

  // R: c_RA <- c(-Inf, p[nCond+nRatings-1] -
  // rev(cumsum(exp(p[(nCond+1):(nCond+nRatings-2)]))), p[nCond+nRatings-1],
  // Inf) R: c_RB <- c(-Inf, p[nCond+nRatings+1], p[nCond+nRatings+1] +
  // cumsum(exp(p[(nCond+nRatings+2):(nCond+nRatings*2-1)])), Inf)

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
    c_RB.subvec(2, nRatings - 1) = anchor_rb + arma::cumsum(arma::exp(p_rb));
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

  for (int j = 0; j < nCond; j++) {
    for (int i = 0; i < nRatings; i++) {
      p_SB_RB(j, i) = (N_SB_RB(j, i) > 0)
                          ? gl_cev(locB(j), c_RB(i), c_RB(i + 1), w, ds(j),
                                   sigma, theta, true)
                          : constants::MIN_P;
      p_SB_RA(j, i) = (N_SB_RA(j, i) > 0)
                          ? gl_cev(locB(j), c_RA(i), c_RA(i + 1), w, ds(j),
                                   sigma, theta, false)
                          : constants::MIN_P;
      p_SA_RA(j, i) = (N_SA_RA(j, i) > 0)
                          ? gl_cev(locA(j), c_RA(i), c_RA(i + 1), w, ds(j),
                                   sigma, theta, false)
                          : constants::MIN_P;
      p_SA_RB(j, i) = (N_SA_RB(j, i) > 0)
                          ? gl_cev(locA(j), c_RB(i), c_RB(i + 1), w, ds(j),
                                   sigma, theta, true)
                          : constants::MIN_P;
    }
  }
  return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB, N_SA_RA, N_SA_RB,
                         N_SB_RA, N_SB_RB);
}