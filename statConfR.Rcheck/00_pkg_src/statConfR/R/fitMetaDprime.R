#' @title title Compute measures of metacognitive sensitivity (meta-d') and metacognitive efficiency(meta-d'/d') for data from one or several subjects

#' @description This function computes the measures for metacognitive sensitivity, meta-d',
#' and metacognitive efficiency, meta-d'/d' (Maniscalco and Lau, 2012, 2014;
#' Fleming, 2017) to data from binary choice tasks with discrete confidence
#' judgments. Meta-d' and meta-d'/d' are computed using a maximum likelihood
#' method for each subset of the `data` argument indicated by different values
#' in the column `participant`, which can represent different subjects as well
#' as experimental conditions.

#' @param data  a `data.frame` where each row is one trial, containing following
#' variables:
#' * \code{rating} (discrete confidence judgments, should be given as factor;
#'    otherwise will be transformed to factor with a warning),
#' * \code{stimulus} (stimulus category in a binary choice task,
#'    should be a factor with two levels, otherwise it will be transformed to
#'    a factor with a warning),
#' * \code{correct} (encoding whether the response was correct; should  be 0 for incorrect responses and 1 for correct responses)
#' * \code{participant} (giving the subject ID; the models given in the second argument are fitted for each
#'   subject individually.
#' @param model `character` of length 1. Either "ML" to use the original model
#' specification by Maniscalco and Lau (2012,  2014) or "F" to use the model
#' specification by Fleming (2017)'s HmetaD method.  Defaults to "ML"
#' @param nInits `integer`. Number of initial values used for maximum likelihood optimization.
#' Defaults to 5.
#' @param nRestart `integer`. Number of times the optimization is restarted.
#' Defaults to 3.
#' @param .parallel `logical`. Whether to parallelize the fitting over models and participant
#' (default: FALSE)
#' @param n.cores `integer`. Number of cores used for parallelization. If NULL (default), the available
#' number of cores -1 will be used.

#' @return Gives data frame with one row for each participant and following columns:
#' - `model` gives the model used for the computation of meta-d' (see `model` argument)
#' - `participant` is the participant ID for the respecitve row
#' - `dprime` is the discrimination sensitivity index d, calculated using a standard SDT formula
#' - `c` is the discrimination bias c, calculated using a standard SDT formula
#' - `metaD` is meta-d', discrimination sensitivity estimated from confidence judgments conditioned on the response
#' - `Ratio` is meta-d'/d', a quantity usually referred to as metacognitive efficiency.

#' @details
#' The function computes meta-d' and meta-d'/d' either using the
#' hypothetical signal detection model assumed by Maniscalco and Lau (2012, 2014)
#' or the one assumed by Fleming (2014).
#'
#' The conceptual idea of meta-d' is to quantify metacognition in terms of sensitivity
#' in a hypothetical signal detection rating model describing the primary task,
#' under the assumption that participants had perfect access to the sensory evidence
#' and were perfectly consistent in placing their confidence criteria (Maniscalco & Lau, 2012, 2014).
#' Using a signal detection model describing the primary task to quantify metacognition allows
#'  a direct comparison between metacognitive accuracy and discrimination performance
#'  because both are measured on the same scale. Meta-d' can be compared against
#'   the estimate of the distance between the two stimulus distributions
#'    estimated from discrimination responses, which is referred to as d':
#'    If meta-d' equals d', it means that metacognitive accuracy is exactly
#'     as good as expected from discrimination performance.
#'     Ifmeta-d' is lower than d', it means that metacognitive accuracy is suboptimal.
#'     It can be shown that the implicit model of confidence underlying the meta-d'/d'
#'      method is identical to the independent truncated Gaussian model.
#'
#' The provided `data` argument is split into subsets according to the values of
#' the `participant` column. Then for each subset, the parameters of the
#' hypothetical signal detection model determined by the `model` argument
#' are fitted to the data subset.
#'
#' The fitting routine first performs a coarse grid search to find promising
#' starting values for the maximum likelihood optimization procedure. Then the best \code{nInits}
#' parameter sets found by the grid search are used as the initial values for separate
#' runs of the Nelder-Mead algorithm implemented in \code{\link[stats]{optim}}.
#' Each run is restarted \code{nRestart} times. Warning: meta-d'/d'
#' is only guaranteed to be unbiased from discrimination sensitivity, discrimination
#' bias, and confidence criteria if the data is generated according to the
#' independent truncated Gaussian model (see Rausch et al., 2023).

#' @author Manuel Rausch, \email{manuel.rausch@hochschule-rhein-waal.de}

#' @references Fleming, S. M. (2017). HMeta-d: Hierarchical Bayesian estimation of metacognitive efficiency from confidence ratings. Neuroscience of Consciousness, 1, 1–14. doi: 10.1093/nc/nix007
#' @references Maniscalco, B., & Lau, H. (2012). A signal detection theoretic method for estimating metacognitive sensitivity from confidence ratings. Consciousness and Cognition, 21(1), 422–430.
#' @references Maniscalco, B., & Lau, H. C. (2014). Signal Detection Theory Analysis of Type 1 and Type 2 Data: Meta-d’, Response- Specific Meta-d’, and the Unequal Variance SDT Model. In S. M. Fleming & C. D. Frith (Eds.), The Cognitive Neuroscience of Metacognition (pp. 25–66). Springer. doi: 10.1007/978-3-642-45190-4_3
#' @references Rausch, M., Hellmann, S., & Zehetleitner, M. (2023). Measures of metacognitive efficiency across cognitive models of decision confidence. Psychological Methods. doi: 10.31234/osf.io/kdz34
#'
#' @examples
#' # 1. Select two subject from the masked orientation discrimination experiment
#' data <- subset(MaskOri, participant %in% c(1:2))
#' head(data)
#'
#' # 2. Fit meta-d/d for each subject in data
#' MetaDs <- fitMetaDprime(data, model="F", .parallel = FALSE)

#' @import parallel
#' @importFrom stats dnorm pnorm pnorm optim integrate

#' @export
fitMetaDprime <- function(data, model="ML",  nInits = 5, nRestart = 3,
                          .parallel=FALSE, n.cores=NULL) {
  if (! model %in% c("ML", "F")) {
    stop("model must be either 'ML' or 'F'")
  }
  if(length(unique(data$stimulus)) != 2) {
      stop("There must be exacltly two different possible values of stimulus")
    }
  if (!is.factor(data$stimulus)) {
    data$stimulus <- factor(data$stimulus)
    warning("stimulus is transformed to a factor!")

  }
  if (!is.factor(data$rating)) {
    data$rating <- factor(data$rating)
    warning("rating is transformed to a factor!")
  }

  if(!all(data$correct %in% c(0,1))) stop("correct should be 1 or 0")
  if(!any(data$correct == 0)) stop("There should be at least one erroneous response")
  if(!any(data$correct == 1)) stop("There should be at least one correct response")
  if(nrow(data) < 400) warning("Warning! At least 400 trials per subject are recommended for measuring metacognitive performance")

  nRatings <-  length(levels(data$rating))
  abj_f <- 1 /(nRatings*2) # adjustment for low frequencies used by Maniscalco and Lau (2012)

  ## Define common names for the output to rbind all parameter fits together
  ## ToDo: Namen anpassen
  outnames <- c("model", "participant", "dprime", "c", "metaD", "Ratio")
  # This function will be called for every combination of participant and model
  call_fitfct <- function(X) {
    cur_model <- model[X[1]]
    cur_sbj <- X[2]
    participant <- NULL # to omit a note in R checks because of an unbound variable
    data_part <- subset(data, participant==cur_sbj)
    res <- int_fitMetaDprime(ratings=data_part$rating,
                             stimulus=data_part$stimulus, correct = data_part$correct,
                             ModelVersion = cur_model,
                             nInits = nInits, nRestart = nRestart,
                             nRatings = nRatings, abj_f = abj_f)
    res$model <- cur_model
    res$participant <- cur_sbj
    res[outnames[!(outnames %in% names(res))]] <- NA
    res <- res[,outnames]
    return(res)
  }

  # generate a list of fitting jobs to do and setup parallelization
  subjects <- unique(data$participant)
  nJobs <- length(model)*length(subjects)
  jobs <- expand.grid(model=1:length(model), sbj=subjects)
  if (.parallel) {
    listjobs <- list()
    for (i in 1:nrow(jobs)) {
      listjobs[[i]] <- c(model = jobs[["model"]][i], sbj = jobs[["sbj"]][i])
    }
    if (is.null(n.cores)) n.cores <- min(nJobs, detectCores() - 1)

    cl <- makeCluster(type="SOCK", n.cores)
    clusterExport(cl, c("data",  "model","outnames", "call_fitfct", "nInits", "nRestart"),
                  envir = environment())
    # Following line ensures that the cluster is stopped even in cases of user
    # interrupt or errors
    on.exit(try(stopCluster(cl), silent = TRUE))
    res <- clusterApplyLB(cl, listjobs, fun=call_fitfct)
    stopCluster(cl)
  } else {
    res <- apply(X=jobs, 1, FUN=call_fitfct)
  }
  # bind list-outout together into data.frame
  res <- do.call(rbind, res)

  # finally, drop columns with unnecessary parameters
  res <- res[,apply(res, 2, function(X) any(!is.na(X)))]
  return(res)
}
