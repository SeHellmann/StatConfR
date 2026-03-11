#' @title Compute model weights
#'
#' @param fit a \code{data.frame} returned by \code{\link{fitConfModels}}.
#' @param ic \code{character}. One of \code{"AIC"}, \code{"BIC"}, or \code{"AICc"}.
#'
#' @return A \code{data.frame} with columns \code{participant}, \code{model},
#' the IC value, delta IC, and \code{weight} (sums to 1 per participant).
#'
#' @export
modelWeights <- function(fit, ic = "AIC") {
  stopifnot(ic %in% c("AIC", "BIC", "AICc"))
  stopifnot(ic %in% names(fit))
  stopifnot(all(c("model", "participant") %in% names(fit)))

  participants <- unique(fit$participant)
  out <- vector("list", length(participants))

  for (idx in seq_along(participants)) {
    sub <- fit[fit$participant == participants[idx], ]
    vals <- sub[[ic]]
    delta <- vals - min(vals, na.rm = TRUE)
    w <- exp(-0.5 * delta)
    w <- w / sum(w, na.rm = TRUE)

    out[[idx]] <- data.frame(
      participant = sub$participant,
      model = sub$model,
      IC = vals,
      deltaIC = delta,
      weight = w,
      stringsAsFactors = FALSE
    )
  }

  res <- do.call(rbind, out)
  names(res)[names(res) == "IC"] <- ic
  names(res)[names(res) == "deltaIC"] <- paste0("delta", ic)
  rownames(res) <- NULL
  res
}


# ---------------------------------------------------------------------------
# Model-to-prediction function mapping (reuses internals from plotConfModelFit.R)
# ---------------------------------------------------------------------------
get_predict_fun <- function(model) {
  switch(model,
         'WEV'   = predictDataWEV,
         'SDT'   = predictDataSDT,
         'GN'    = predictDataNoisy,
         'PDA'   = predictDataISDT,
         'IG'    = predictData2Chan,
         'ITGc'  = predictDataIndTruncF,
         'ITGcm' = predictDataIndTruncML,
         'logN'  = predictDataLognorm,
         'logWEV'= predictDataLogWEV,
         'CAS'   = predictDataCAS,
         'RCE'   = predictDataRCE,
         stop(paste0("Unknown model: ", model)))
}


#' @title Get predicted response probabilities from fitted parameters
#'
#' @param fit a \code{data.frame} returned by \code{\link{fitConfModels}} or
#'   \code{\link{fitConf}}.
#' @param model \code{character}. Which model to use. Required if \code{fit}
#'   contains multiple models.
#'
#' @return A \code{data.frame} with columns \code{participant}, \code{stimulus},
#'   \code{response}, \code{diffCond}, \code{rating}, and \code{p} (predicted
#'   probability).
#'
#' @importFrom plyr ddply
#' @export
predictConf <- function(fit, model = NULL) {
  if (is.null(model)) {
    if ("model" %in% names(fit) && length(unique(fit$model)) == 1) {
      model <- unique(fit$model)
    } else {
      stop("Specify 'model' when fit contains multiple models.")
    }
  }

  pred_fun <- get_predict_fun(model)

  if ("model" %in% names(fit) && length(unique(fit$model)) > 1) {
    fit <- fit[fit$model == model, ]
  }

  if ("participant" %in% names(fit)) {
    res <- plyr::ddply(fit, ~participant, pred_fun)
  } else {
    res <- pred_fun(fit)
  }

  res$correct <- NULL
  res
}


#' @title Print a clean summary of fitted model results
#'
#' @param fit a \code{data.frame} returned by \code{\link{fitConfModels}} or
#'   \code{\link{fitConf}}.
#' @param model \code{character}. Which model to summarize. Required if
#'   \code{fit} contains multiple models.
#' @param participant optional participant ID to summarize a single participant.
#'
#' @return Invisibly returns the input \code{fit}. Called for its side effect
#'   of printing a summary.
#'
#' @export
summarizeFit <- function(fit, model = NULL, participant = NULL) {
  if (!is.null(model) && "model" %in% names(fit)) {
    fit <- fit[fit$model == model, ]
  }
  if (!is.null(participant) && "participant" %in% names(fit)) {
    fit <- fit[fit$participant == participant, ]
  }
  if (nrow(fit) == 0) {
    message("No matching rows found.")
    return(invisible(fit))
  }

  # columns that are fit metrics or identifiers, not parameters
  meta_cols <- c("model", "participant", "negLogLik", "N", "k", "BIC", "AICc", "AIC")

  for (i in seq_len(nrow(fit))) {
    row <- fit[i, ]
    mod_name <- if ("model" %in% names(row)) row$model else "unknown"
    pid <- if ("participant" %in% names(row)) row$participant else ""

    cat(sprintf("Model: %s", mod_name))
    if (pid != "") cat(sprintf("  |  Participant: %s", pid))
    cat("\n")

    cat(sprintf("  Log-likelihood: %.2f\n", -row$negLogLik))
    cat(sprintf("  AIC:  %.2f\n", row$AIC))
    cat(sprintf("  BIC:  %.2f\n", row$BIC))
    cat(sprintf("  AICc: %.2f\n", row$AICc))
    cat(sprintf("  N: %d  |  k: %d\n", row$N, row$k))

    # --- structured parameter output ---
    par_cols <- setdiff(names(row), meta_cols)
    par_cols <- par_cols[!is.na(row[par_cols])]

    # sensitivity
    d_cols <- grep("^d_\\d+$", par_cols, value = TRUE)
    if (length(d_cols) > 0) {
      cat("\n  Sensitivity (d'):\n")
      for (j in seq_along(d_cols)) {
        cat(sprintf("    Condition %d:  %8.4f\n", j, as.numeric(row[[d_cols[j]]])))
      }
    }

    # decision bias
    if ("c" %in% par_cols) {
      cat(sprintf("\n  Decision bias (c):  %.4f\n", as.numeric(row$c)))
    }

    # confidence criteria
    tm_cols <- grep("^theta_minus\\.", par_cols, value = TRUE)
    tp_cols <- grep("^theta_plus\\.", par_cols, value = TRUE)
    if (length(tm_cols) > 0) {
      vals <- vapply(tm_cols, function(x) as.numeric(row[[x]]), numeric(1))
      cat(sprintf("\n  Confidence criteria (response A):  %s\n",
                  paste(sprintf("%.3f", vals), collapse = ", ")))
    }
    if (length(tp_cols) > 0) {
      vals <- vapply(tp_cols, function(x) as.numeric(row[[x]]), numeric(1))
      cat(sprintf("  Confidence criteria (response B):  %s\n",
                  paste(sprintf("%.3f", vals), collapse = ", ")))
    }

    # mean confidence criteria (logN)
    mtm_cols <- grep("^M_theta_minus\\.", par_cols, value = TRUE)
    mtp_cols <- grep("^M_theta_plus\\.", par_cols, value = TRUE)
    if (length(mtm_cols) > 0) {
      vals <- vapply(mtm_cols, function(x) as.numeric(row[[x]]), numeric(1))
      cat(sprintf("\n  Mean confidence criteria (response A):  %s\n",
                  paste(sprintf("%.3f", vals), collapse = ", ")))
    }
    if (length(mtp_cols) > 0) {
      vals <- vapply(mtp_cols, function(x) as.numeric(row[[x]]), numeric(1))
      cat(sprintf("  Mean confidence criteria (response B):  %s\n",
                  paste(sprintf("%.3f", vals), collapse = ", ")))
    }

    # model specific parameters
    special <- c("sigma", "w", "b", "m")
    special_labels <- c(sigma = "Confidence noise (sigma)",
                        w = "Visibility weight (w)",
                        b = "Post-decisional accumulation (b)",
                        m = "Metacognitive efficiency (m)")
    found_special <- intersect(special, par_cols)
    if (length(found_special) > 0) {
      cat("\n  Model-specific:\n")
      for (sp in found_special) {
        cat(sprintf("    %-38s %8.4f\n", special_labels[sp], as.numeric(row[[sp]])))
      }
    }
    cat("\n")
  }

  invisible(fit)
}
