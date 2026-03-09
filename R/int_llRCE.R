ll_RCE <-
  function(p, N_SA_RA,N_SA_RB, N_SB_RA, N_SB_RB, nRatings, nCond){
    p <- c(t(p))
    ds <- cumsum(exp(p[1:(nCond)])) # enforce that sensitivity is ordered

    theta <- p[nCond+nRatings]
    c_RA <- c(-Inf, p[nCond+nRatings-1] -
                rev(cumsum(c(exp(p[(nCond+1):(nCond+nRatings-2)])))),
              p[nCond+nRatings-1], Inf)
    c_RB <- c(-Inf, p[nCond+nRatings+1], p[nCond+nRatings+1] +
                cumsum(c(exp(p[(nCond+nRatings+2):(nCond + nRatings*2-1)]))), Inf)
    sigma <- sqrt(1/2)

    p_SA_RA <- matrix(NA, nrow=nCond, ncol = nRatings)
    p_SA_RB <- matrix(NA, nrow=nCond, ncol = nRatings)
    p_SB_RA <- matrix(NA, nrow=nCond, ncol = nRatings)
    p_SB_RB <- matrix(NA, nrow=nCond, ncol = nRatings)

    P_SBRB <-  Vectorize(function(j, i){
      integrate(function(x) dnorm(x, mean=ds[j]/2, sigma) * pnorm(x-theta, mean=0, sigma),
                lower = c_RB[i],
                upper = c_RB[i+1],
                rel.tol = 10^-8)$value
    })
    P_SARB <- Vectorize(function(j,i){
      integrate(function(x) dnorm(x, mean = 0, sigma) * pnorm(x-theta, mean=ds[j]/2, sigma), # dnorm(x, mean=) * (1-pnorm(x, mean=b))
                lower = c_RB[i],
                upper = c_RB[i+1],
                rel.tol = 10^-8)$value
    })

    P_SBRA <-  Vectorize(function(j,i){
      integrate(function(x) dnorm(x, mean= 0, sigma) * pnorm(x+theta, mean=ds[j]/2, sigma) , # dnorm(x, mean=ds[j]+b) * (1 - pnorm(x, mean=-b))
                lower = -c_RA[i+1], # braucht es hier einen Vorzeichenwechsel?
                upper = -c_RA[i],
                rel.tol = 10^-8)$value
    })
    P_SARA <- Vectorize(function(j,i){
      integrate(function(x) dnorm(x, mean = ds[j]/2, sigma) * pnorm(x+theta, mean=0, sigma),
                lower = -c_RA[i+1],
                upper = -c_RA[i],
                rel.tol = 10^-8)$value
    })

    p_SB_RB <- outer(1:nCond, 1:nRatings, P_SBRB)
    p_SB_RA <- outer(1:nCond, 1:nRatings, P_SBRA)
    p_SA_RA <- outer(1:nCond, 1:nRatings, P_SARA)  # rowSums(p_SB_RA) + rowSums(p_SB_RB)
    p_SA_RB <- outer(1:nCond, 1:nRatings, P_SARB) # rowSums(p_SA_RA) + rowSums(p_SA_RB)


    p_SB_RB[(is.na(p_SB_RB))| is.nan(p_SB_RB)| p_SB_RB < 10^-64] <- 10^-64
    p_SB_RA[(is.na(p_SB_RA))| is.nan(p_SB_RA)| p_SB_RA < 10^-64] <- 10^-64
    p_SA_RB[(is.na(p_SA_RB))| is.nan(p_SA_RB)| p_SA_RB < 10^-64] <- 10^-64
    p_SA_RA[(is.na(p_SA_RA))| is.nan(p_SA_RA)| p_SA_RA < 10^-64] <- 10^-64

    negLogL <- - sum (c(log(p_SB_RB) * N_SB_RB, log(p_SB_RA) * N_SB_RA,
                        log(p_SA_RB) * N_SA_RB, log(p_SA_RA) * N_SA_RA))

    negLogL
  }