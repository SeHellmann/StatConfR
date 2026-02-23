#' @importFrom stats optim aggregate
NULL

prepare_accum_inits <- function(model, nRatings, nCond, defs) {
  temp <- defs$grid
  
  # Parameter count
  nTheta <- if (defs$steps_mode == "full") (nRatings - 1) * 2 else (nRatings - 2) * 2 + 2
  nMeta <- length(defs$meta_params)
  nParams <- nCond + 1 + nTheta + nMeta
  
  inits <- matrix(NA, nrow = nrow(temp), ncol = nParams)
  
  # 1. Sensitivity
  if (nCond == 1) {
    inits[, 1] <- log(temp$maxD)
  } else {
    inits[, 1:nCond] <- log(t(mapply(function(maxD) diff(seq(0, maxD, length.out = nCond + 1)), temp$maxD)))
  }
  
  # 2. Theta
  inits[, nCond + nRatings] <- temp$theta
  
  # 3. Confidence Criteria
  if (defs$steps_mode == "full") {
    # SDT, ITG, logN, logWEV, CAS
    if (isTRUE(defs$cas_special)) {
      if (nRatings > 3) {
        inits[, (nCond + 1):(nCond + nRatings - 1)] <-
          log(t(mapply(function(tauMin, tauRange) diff(seq(-tauRange - tauMin, -tauMin, length.out = nRatings)),
                       temp$tauMin, temp$tauRange)))
        inits[, (nCond + nRatings + 1):(nCond + nRatings * 2 - 1)] <-
          log(t(mapply(function(tauMin, tauRange) diff(seq(tauMin, tauMin + tauRange, length.out = nRatings)),
                       temp$tauMin, temp$tauRange)))
      } else if (nRatings == 3) {
        inits[, (nCond + 1):(nCond + nRatings - 1)] <-
          log(mapply(function(tauMin, tauRange) diff(seq(-tauRange - tauMin, -tauMin, length.out = nRatings)),
                     temp$tauMin, temp$tauRange))
        inits[, (nCond + nRatings + 1):(nCond + nRatings * 2 - 1)] <-
          log(mapply(function(tauMin, tauRange) diff(seq(tauMin, tauMin + tauRange, length.out = nRatings)),
                     temp$tauMin, temp$tauRange))
      }
    } else {
      # Standard full mode (SDT, ITG, logN, logWEV)
      if (nRatings > 3) {
        step_vals <- t(mapply(function(tauRange) rep(tauRange / (nRatings - 1), nRatings - 2), temp$tauRange))
        inits[, (nCond + 1):(nCond + nRatings - 2)] <- log(step_vals)
        inits[, (nCond + nRatings + 2):(nCond + 2 * nRatings - 1)] <- log(step_vals)
      } else if (nRatings == 3) {
        step_vals <- mapply(function(tauRange) rep(tauRange / (nRatings - 1), nRatings - 2), temp$tauRange)
        inits[, nCond + 1] <- log(step_vals)
        inits[, nCond + 4] <- log(step_vals)
      }
      inits[, nCond + (nRatings - 1)] <- log(temp$tauMin)
      inits[, nCond + (nRatings + 1)] <- log(temp$tauMin)
    }
  } else {
    # Noisy, PDA, WEV, IG, RCE
    if (nRatings > 3) {
      inits[, (nCond + 1):(nCond + nRatings - 2)] <-
        log(t(mapply(function(tauMin, tauRange) diff(seq(-tauRange - tauMin, -tauMin, length.out = nRatings - 1)),
                     temp$tauMin, temp$tauRange)))
      inits[, (nCond + nRatings + 2):(nCond + nRatings * 2 - 1)] <-
        log(t(mapply(function(tauMin, tauRange) diff(seq(tauMin, tauMin + tauRange, length.out = nRatings - 1)),
                     temp$tauMin, temp$tauRange)))
    } else if (nRatings == 3) {
      inits[, (nCond + 1)] <-
        log(mapply(function(tauMin, tauRange) diff(seq(-tauRange - tauMin, -tauMin, length.out = nRatings - 1)),
                   temp$tauMin, temp$tauRange))
      inits[, (nCond + 4)] <-
        log(mapply(function(tauMin, tauRange) diff(seq(tauMin, tauMin + tauRange, length.out = nRatings - 1)),
                   temp$tauMin, temp$tauRange))
    }
    inits[, nCond + (nRatings - 1)] <- if (model %in% c("GN", "PDA", "WEV")) -temp$tauMin else temp$tauMin
    inits[, nCond + (nRatings + 1)] <- temp$tauMin
  }
  
  # 4. Meta Parameters
  if (nMeta > 0) {
    param_names <- names(defs$meta_params)
    for (i in seq_along(param_names)) {
      p_name <- param_names[i]
      pos <- nCond + 1 + nTheta + i
      inits[, pos] <- defs$meta_params[[p_name]]$link(temp[[p_name]])
    }
  }
  
  return(inits)
}

process_accum_results <- function(fit, model, nRatings, nCond, nTrials, defs) {
  res <- data.frame(matrix(nrow = 1, ncol = 0))
  if (fit$error) return(res)
  
  best_par <- as.vector(fit$par)
  best_val <- as.numeric(fit$value)
  k <- length(best_par)
  
  # 1. Sensitivity
  res[paste0("d_", 1:nCond)] <- as.vector(cumsum(exp(best_par[1:nCond])))
  
  # 2. Criterion
  c_val <- best_par[nCond + nRatings]
  res$c <- c_val
  
  # 3. Threshold Anchor logic
  anchor_val_type <- defs$anchor
  anchor_val <- if (anchor_val_type == "mc") {
    m <- exp(best_par[nCond + nRatings * 2])
    m * c_val
  } else if (anchor_val_type == "tauMin") {
    NA
  } else if (anchor_val_type == "zero") {
    0
  } else {
    c_val
  }
  
  # 4. Thresholds
  prefix <- if (!is.null(defs$prefix)) defs$prefix else ""
  
  if (defs$steps_mode == "full") {
    # SDT, ITG, logN, logWEV, CAS
    if (anchor_val_type == "zero") {
      # logWEV, CAS
      res[, paste0(prefix, "theta_minus.", (nRatings - 1):1)] <-
        rev(c(-as.vector(cumsum(c(exp(best_par[(nCond + 1):(nCond + nRatings - 1)]))))))
      res[, paste0(prefix, "theta_plus.", 1:(nRatings - 1))] <-
        as.vector(cumsum(c(exp(best_par[(nCond + nRatings + 1):(nCond + nRatings * 2 - 1)]))))
    } else {
      # SDT, ITG, logN
      res[, paste0(prefix, "theta_minus.", (nRatings - 1):1)] <-
        as.vector(anchor_val - rev(cumsum(c(exp(best_par[(nCond + 1):(nCond + nRatings - 1)])))))
      res[, paste0(prefix, "theta_plus.", 1:(nRatings - 1))] <-
        as.vector(anchor_val + cumsum(c(exp(best_par[(nCond + nRatings + 1):(nCond + nRatings * 2 - 1)]))))
    }
  } else {
    # Noisy, PDA, WEV, IG, RCE
    theta_minus_max <- best_par[nCond + nRatings - 1]
    res[, paste0(prefix, "theta_minus.", (nRatings - 1):1)] <-
      c(as.vector(theta_minus_max - rev(cumsum(c(exp(best_par[(nCond + 1):(nCond + nRatings - 2)]))))),
        theta_minus_max)
        
    theta_plus_min <- best_par[nCond + nRatings + 1]
    res[, paste0(prefix, "theta_plus.", 1:(nRatings - 1))] <-
      c(theta_plus_min,
        as.vector(theta_plus_min + cumsum(c(exp(best_par[(nCond + nRatings + 2):(nCond + nRatings * 2 - 1)])))))
  }
  
  # 5. Meta Parameters
  if (defs$has_meta) {
    param_names <- names(defs$meta_params)
    nTheta <- if (defs$steps_mode == "full") (nRatings - 1) * 2 else (nRatings - 2) * 2 + 2
    for (i in seq_along(param_names)) {
      p_name <- param_names[i]
      pos <- nCond + 1 + nTheta + i
      res[[p_name]] <- defs$meta_params[[p_name]]$inv_link(best_par[pos])
    }
    if ("m" %in% names(defs$meta_params) && model == "IG") {
       # Note: already handled by loop
    }
  }
  
  # 6. Model Fit Measures
  res$negLogLik <- best_val
  res$N <- nTrials
  res$k <- k
  res$BIC <- 2 * best_val + k * log(nTrials)
  res$AIC <- 2 * best_val + 2 * k
  denom <- nTrials - k - 1
  res$AICc <- if (denom > 0) res$AIC + (2 * k * (k + 1)) / denom else NA
  
  return(res)
}
