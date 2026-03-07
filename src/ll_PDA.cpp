#include "gauss_legendre.h"
#include "likelihoods_func.h"
#include "utils.h"

// [[Rcpp::export]]
double ll_PDA_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                  const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                  const arma::mat &N_SB_RB, int nRatings, int nCond) {
  const arma::vec ds = compute_sensitivity(p, nCond);
  const arma::vec locA = -ds / 2.0;
  const arma::vec locB = ds / 2.0;
  const double theta = p(nCond + nRatings - 1);

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

  const double a = std::exp(p(nCond + nRatings * 2 - 1));
  const double sigma = std::sqrt(a);
  const double trunc = 7.0;
  const double K = trunc * sigma;

  // Probability matrices
  arma::mat p_SA_RA(nCond, nRatings);
  arma::mat p_SA_RB(nCond, nRatings);
  arma::mat p_SB_RA(nCond, nRatings);
  arma::mat p_SB_RB(nCond, nRatings);

  for (int j = 0; j < nCond; j++) {
    const double lB = locB(j), lA = locA(j);
    for (int i = 0; i < nRatings; i++) {

      const double cRB_lo = c_RB(i), cRB_hi = c_RB(i + 1);
      const double cRA_lo = c_RA(i), cRA_hi = c_RA(i + 1);
      
      // For SB_RB (loc = lB, using c_RB):
      const double rb_lo_B = std::max(theta, cRB_lo - lB*a - K);
      const double rb_hi_B = std::min(lB + trunc, cRB_hi - lB*a + K);
      // For SB_RA (loc = lB, using c_RA):
      const double ra_lo_B = std::max(lB - trunc, cRA_lo - lB*a - K);
      const double ra_hi_B = std::min(theta, cRA_hi - lB*a + K);
      // For SA_RA (loc = lA, using c_RB):
      const double rb_lo_A = std::max(theta, cRB_lo - lA*a - K);
      const double rb_hi_A = std::min(lA + trunc, cRB_hi - lA*a + K);
      // For SA_RB (loc = lA, using c_RA):
      const double ra_lo_A = std::max(lA - trunc, cRA_lo - lA*a - K);
      const double ra_hi_A = std::min(theta, cRA_hi - lA*a + K);

  auto rb_B = [&](double x) {
      double m = x + lB * a;
      return pnorm_cpp(cRB_hi, m, sigma) - pnorm_cpp(cRB_lo, m, sigma);
  };
  auto rb_A = [&](double x) {
      double m = x + lA * a;
      return pnorm_cpp(cRB_hi, m, sigma) - pnorm_cpp(cRB_lo, m, sigma);
  };
  auto ra_B = [&](double x) {
      double m = x + lB * a;
      return pnorm_cpp(cRA_hi, m, sigma) - pnorm_cpp(cRA_lo, m, sigma);
  };
  auto ra_A = [&](double x) {
      double m = x + lA * a;
      return pnorm_cpp(cRA_hi, m, sigma) - pnorm_cpp(cRA_lo, m, sigma);
  };


      p_SB_RB(j, i) =
          (N_SB_RB(j, i) > 0)
              ? gl_integrate(rb_lo_B, rb_hi_B, lB, rb_B)
              : constants::MIN_P;
      p_SB_RA(j, i) =
          (N_SB_RA(j, i) > 0)
              ? gl_integrate(ra_lo_B, ra_hi_B, lB, ra_B)
              : constants::MIN_P;
      p_SA_RA(j, i) =
          (N_SA_RA(j, i) > 0)
              ? gl_integrate(ra_lo_A, ra_hi_A, lA, ra_A)
              : constants::MIN_P;
      p_SA_RB(j, i) =
          (N_SA_RB(j, i) > 0)
              ? gl_integrate(rb_lo_A, rb_hi_A, lA, rb_A)
              : constants::MIN_P;
    }
  }

  return compute_negLogL(p_SA_RA, p_SA_RB, p_SB_RA, p_SB_RB, N_SA_RA, N_SA_RB,
                         N_SB_RA, N_SB_RB);
}