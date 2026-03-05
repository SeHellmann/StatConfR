#include "gauss_legendre.h"
#include "likelihoods_func.h"
#include "utils.h"

// Integrates φ(x; loc, 1) · [Φ(c_hi; x, σ) - Φ(c_lo; x, σ)]
inline double gl_noisy(double loc, double c_lo, double c_hi, double sigma,
                       double theta, bool response_B) {
  // beyond SD = 7 contribution from integration negligible
  double a = response_B ? theta : loc - 7.0;
  double b = response_B ? loc + 7.0 : theta;
  double mid = (a + b) / 2.0;
  double half = (b - a) / 2.0;
  double sum = 0.0;
  for (int k = 0; k < GL_ORDER; k++) {
    double x = mid + half * GL_NODES[k];
    double phi_x =
        std::exp(-0.5 * (x - loc) * (x - loc)) * constants::INV_SQRT_2PI;
    sum += GL_WEIGHTS[k] * phi_x *
           (pnorm_cpp(c_hi, x, sigma) - pnorm_cpp(c_lo, x, sigma));
  }
  return sum * half;
}

// [[Rcpp::export]]
double ll_Noisy_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
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

  // Probability matrices
  arma::mat p_SA_RA(nCond, nRatings);
  arma::mat p_SA_RB(nCond, nRatings);
  arma::mat p_SB_RA(nCond, nRatings);
  arma::mat p_SB_RB(nCond, nRatings);

  for (int j = 0; j < nCond; j++) {
    for (int i = 0; i < nRatings; i++) {
      p_SB_RB(j, i) =
          gl_noisy(locB(j), c_RB(i), c_RB(i + 1), sigma, theta, true);
      p_SB_RA(j, i) =
          gl_noisy(locB(j), c_RA(i), c_RA(i + 1), sigma, theta, false);
      p_SA_RA(j, i) =
          gl_noisy(locA(j), c_RA(i), c_RA(i + 1), sigma, theta, false);
      p_SA_RB(j, i) =
          gl_noisy(locA(j), c_RB(i), c_RB(i + 1), sigma, theta, true);
    }
  }
  return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB, N_SA_RA, N_SA_RB,
                         N_SB_RA, N_SB_RB);
}