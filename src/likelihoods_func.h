#pragma once
#include <RcppArmadillo.h>
#include "shared_types.h"


// =========================================================
// Aggregated Likelihoods
// ======================================================

double ll_SDT_cpp(const arma::vec& p, const ModelData& dat);

double ll_SDTvarS_cpp(const arma::vec& p, const ModelData& dat);

double ll_2Chan_cpp(const arma::vec& p, const ModelData& dat);        

double ll_Mratio_cpp(const arma::vec& p, const ModelData& dat);

double ll_MratioF_cpp(const arma::vec& p, const ModelData& dat);
        
double ll_CAS_cpp(const arma::vec& p, const ModelData& dat);

double ll_CEV_cpp(const arma::vec& p, const ModelData& dat);

double ll_LogNorm_cpp(const arma::vec& p, const ModelData& dat);

double ll_LogWEV_cpp(const arma::vec& p, const ModelData& dat);

double ll_Noisy_cpp(const arma::vec& p, const ModelData& dat);

double ll_PDA_cpp(const arma::vec& p, const ModelData& dat);

double ll_RCE_cpp(const arma::vec& p, const ModelData& dat);

// =========================================================
// Regression Likelihoods
// =========================================================

double ll_SDT_regression(const arma::vec& p, const RegressionData& dat);

double ll_2Chan_regression(const arma::vec& p, const RegressionData& dat);

double ll_Mratio_regression(const arma::vec& p, const RegressionData& dat);

double ll_MratioF_regression(const arma::vec& p, const RegressionData& dat);

double ll_CAS_regression(const arma::vec& p, const RegressionData& dat);

double ll_CEV_regression(const arma::vec& p, const RegressionData& dat);

double ll_LogNorm_regression(const arma::vec& p, const RegressionData& dat);

double ll_LogWEV_regression(const arma::vec& p, const RegressionData& dat);

double ll_Noisy_regression(const arma::vec& p, const RegressionData& dat);

double ll_PDA_regression(const arma::vec& p, const RegressionData& dat);

double ll_RCE_regression(const arma::vec& p, const RegressionData& dat);
