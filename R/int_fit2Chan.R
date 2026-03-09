###  Fit the independent Gaussian model

fit2Chan <-
  function(N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB,
           nInits, nRestart, nRatings, nCond, nTrials){



    # search for initial values using a coarse search grind
    temp <- expand.grid(maxD =  seq(1, 5, 1),
                        theta = seq(-1/2,1/2, 1/2),
                        tauMin =  c(.1, .3, 1),  # position of the most conservative confidence criteria with respect to theta
                        tauRange = seq(1, 5, 1),  #  position of the most liberal confidence criterion with respect to theta
                        a = c(.1, .3, 1, 3)) # metacognitive efficiency

    # number of parameters:
    # nCond -1 sensitivity parameters
    # 1 type 1 bias parameter
    # 2 * (nRatings - 1) confidence criteria

    inits <- data.frame(matrix(data=NA, nrow= nrow(temp),
                               ncol = nCond + nRatings*2 ))
    if(nCond==1)  {
      inits[,1] <-  log(temp$maxD)  }
    else{
      inits[,1:(nCond)] <-  log(t(mapply(function(maxD) diff(seq(0, maxD, length.out = nCond+1)), temp$maxD)))
    }
    if (nRatings > 3){
      inits[,(nCond+1):(nCond+nRatings-2)] <-
        log(t(mapply(function(tauMin, tauRange) diff(seq(-tauRange-tauMin, -tauMin, length.out=nRatings-1)),
                     temp$tauMin, temp$tauRange)))
      inits[,(nCond+nRatings+2):(nCond + nRatings*2-1)] <-
        log(t(mapply(function(tauMin, tauRange) diff(seq(tauMin, tauMin+tauRange, length.out=nRatings-1)),
                     temp$tauMin, temp$tauRange)))
    }
    if (nRatings == 3){
      inits[,(nCond+1):(nCond+nRatings-2)] <-
        log(mapply(function(tauMin, tauRange) diff(seq(-tauRange-tauMin, -tauMin, length.out=nRatings-1)),
                   temp$tauMin, temp$tauRange))
      inits[,(nCond+nRatings+2):(nCond + nRatings*2-1)] <-
        log(mapply(function(tauMin, tauRange) diff(seq(tauMin, tauMin+tauRange, length.out=nRatings-1)),
                   temp$tauMin, temp$tauRange))
    }
    inits[,nCond+(nRatings-1)] <- temp$tauMin
    inits[,nCond+nRatings] <- temp$theta
    inits[,nCond+(nRatings+1)] <- temp$tauMin
    inits[,(nCond + nRatings*2)] <- log(temp$a)

    logL <- apply(inits, MARGIN = 1,
                  function(p) try(ll2Chan(p, N_SA_RA, N_SA_RB, N_SB_RA,N_SB_RB, nRatings, nCond), silent = TRUE))
    logL <- as.numeric(logL)
    inits <- inits[order(logL),]
    inits <- inits[1:nInits,]

    noFitYet <- TRUE
    for (i in 1:nInits){

      m <- try(optim(par =  inits[i,],
                     fn = ll2Chan, gr = NULL,
                     N_SA_RA = N_SA_RA,N_SA_RB = N_SA_RB,
                     N_SB_RA = N_SB_RA,N_SB_RB = N_SB_RB, nRatings = nRatings, nCond = nCond,
                     control = list(maxit = 10^4, reltol = 10^-4)))

      if (!inherits(m, "try-error")){
        for(j in 2:nRestart){
          try(m <- optim(par = m$par,
                         fn = ll2Chan, gr = NULL,
                         N_SA_RA = N_SA_RA,N_SA_RB = N_SA_RB,
                         N_SB_RA = N_SB_RA,N_SB_RB = N_SB_RB, nRatings = nRatings, nCond = nCond,
                         control = list(maxit = 10^6, reltol = 10^-8)))

        }
        if (noFitYet) {
          fit <- m
          noFitYet <- FALSE
        } else {
          if (m$value < fit$value) fit <- m
        }
      }
    }

    res <-  data.frame(matrix(nrow=1, ncol=0))
    if(!inherits(fit, "try-error")){

      k <- length(fit$par)
      res[paste("d_",1:nCond, sep="")] <-  as.vector(cumsum(exp(fit$par[1:(nCond)])))
      res$c <-  as.vector(fit$par[nCond+nRatings])

      res[,paste("theta_minus.",(nRatings-1):1, sep="")] <-
        c(as.vector(fit$par[nCond+nRatings-1] - rev(cumsum(c(exp(fit$par[(nCond+1):(nCond+nRatings-2)]))))),
          as.vector(fit$par[nCond+nRatings-1]))
      res[,paste("theta_plus.",1:(nRatings-1), sep="")] <-
        c(as.vector(fit$par[nCond+nRatings+1]),
          as.vector(fit$par[nCond+nRatings+1]) +
            as.vector(cumsum(c(exp(fit$par[(nCond+nRatings+2):(nCond + nRatings*2-1)])))))

      res$m <- exp(fit$par[nCond + nRatings*2])

      res$negLogLik <- fit$value
      res$N <- nTrials
      res$k <- k
      res$BIC <-  2 * fit$value + k * log(nTrials)
      res$AIC <- 2 * fit$value + 2 * k
      denom <- nTrials - k - 1
      res$AICc <- if (denom > 0) res$AIC + (2 * k * (k + 1)) / denom else NA
    }
    res
  }


fit2Chan_fast <-
  function(N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB,
           nInits, nRestart, nRatings, nCond, nTrials){

    # 1. Define Model Parameters
    model_defs <- list(
      grid = expand.grid(maxD = seq(1, 5, 1),
                         theta = seq(-1/2, 1/2, 1/2),
                         tauMin = c(0.1, 0.3, 1),
                         tauRange = seq(1, 5, 1),
                         m = c(0.1, 0.3, 1, 3)),
      has_meta = TRUE,
      meta_params = list(m = list(link = log, inv_link = exp)),
      anchor = "tauMin",
      steps_mode = "partial"
    )
    inits <- prepare_accum_inits("IG", nRatings, nCond, model_defs)
    
    # 2. Compute Likelihoods on Grid (Using C++ for speed)
    fn_ptr <- get_2chan_ptr()
    logL <- compute_generic_grid_ll(fn_ptr, as.matrix(inits), N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond)
    
    # 3. Select Best Initial Values
    inits_top <- inits[order(logL)[1:nInits], , drop=FALSE]

    # 4. Optimization Loop (Using C++ for speed)
    fit_res <- optimize_generic_loop(fn_ptr, as.matrix(inits_top), 
                                 N_SA_RA, N_SA_RB, N_SB_RA, N_SB_RB, 
                                 nRestart, nRatings, nCond)

    # 5. Post Processing
    res <- process_accum_results(fit_res, "IG", nRatings, nCond, nTrials, model_defs)
    return(res)
  }


fit2Chan_regression <- function(data, formulas, nInits, nRestart) {
  
  # 1. Prepare Input
  prep <- preprocess_regression_input(data, formulas, model = "IG", nInits = nInits)
  
  # 2. Optimization
  fn_ptr <- get_2chan_regression_ptr()
  
  fit_res <- optimize_regression_loop(fn_ptr, 
                                 as.matrix(prep$inits),
                                 prep$X_list,
                                 prep$stim_vec, 
                                 prep$rating_vec, 
                                 prep$correct_vec,
                                 prep$counts_vec,
                                 nRestart, prep$nRatings, prep$nTrials)
  
  # 3. Postprocess Results
  res <- process_regression_results(fit_res, prep, prep$param_defs)
  class(res) <- "2chan_fit"
  return(res)
}