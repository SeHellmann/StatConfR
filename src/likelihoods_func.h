#pragma once
#include <RcppArmadillo.h>

double ll_SDT_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                  const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                  const arma::mat &N_SB_RB, int nRatings, int nCond);

double ll_SDTvarS_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                      const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                      const arma::mat &N_SB_RB, int nRatings, int nCond);

double ll_2Chan_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                    const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                    const arma::mat &N_SB_RB, int nRatings, int nCond);

double ll_Mratio_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                     const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                     const arma::mat &N_SB_RB, int nRatings, int nCond);

double ll_MratioF_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                      const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                      const arma::mat &N_SB_RB, int nRatings, int nCond);

double ll_CAS_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                  const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                  const arma::mat &N_SB_RB, int nRatings, int nCond);

double ll_CEV_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                  const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                  const arma::mat &N_SB_RB, int nRatings, int nCond);

double ll_LogNorm_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                      const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                      const arma::mat &N_SB_RB, int nRatings, int nCond);

double ll_LogWEV_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                     const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                     const arma::mat &N_SB_RB, int nRatings, int nCond);

double ll_Noisy_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                    const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                    const arma::mat &N_SB_RB, int nRatings, int nCond);

double ll_PDA_cpp(const arma::vec &p, const arma::mat &N_SA_RA,
                  const arma::mat &N_SA_RB, const arma::mat &N_SB_RA,
                  const arma::mat &N_SB_RB, int nRatings, int nCond);