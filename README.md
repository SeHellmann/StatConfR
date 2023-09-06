# statConfR: R package for models of decision confidence

This package includes functions to fit the parameters for several static
models of decision making and confidence judgments. More precisely,
following models are included:

- signal-detection theory (with constant and increasing variances)

- noisy signal-detection theory

- weighted evidence and visibility model (WEV; with constant and
  increasing variances)

- post-decisional accumulation model

- two-channel model

See Rausch et al. (2018) for more detail about the models.

## Installation

For the current development version, the easiest way of installation is
using `devtools` and install from GitHub:

    devtools::install_github("ManuelRausch/StatConfR")

The latest released version of the package will soon be available on
CRAN via

``` r
install.packages("statConfR")
```

## Usage

### Data structure

The package includes a demo data set from a masked orientation
discrimination task with confidence judgments.

``` r
library(statConfR)
data("MaskOri")
head(MaskOri)
```

    ##   participant stimulus correct rating diffCond gender age trialNo
    ## 1           1        0       0      1     33.3      f  23       1
    ## 2           1        0       0      1     16.7      f  23       2
    ## 3           1       90       1      1    133.3      f  23       3
    ## 4           1        0       1      4    133.3      f  23       4
    ## 5           1       90       1      2     33.3      f  23       5
    ## 6           1       90       0      2     16.7      f  23       6

Data should be in the form of a data.frame object columns for following
variables:

- stimulus (factor with 2 levels): In a binary decision task the
  stimulus identity gives the correct response
- condition (factor): The experimental manipulation that is expected to
  affect sensitivity should be present
- correct (0-1): Indicating whether the choice was correct (1) or
  incorrect(0).
- rating (factor): A discrete variable encoding the decision confidence
  (high: very confident; low: less confident)

``` r
MaskOri$condition <- MaskOri$diffCond
head(MaskOri)
```

    ##   participant stimulus correct rating diffCond gender age trialNo condition
    ## 1           1        0       0      1     33.3      f  23       1      33.3
    ## 2           1        0       0      1     16.7      f  23       2      16.7
    ## 3           1       90       1      1    133.3      f  23       3     133.3
    ## 4           1        0       1      4    133.3      f  23       4     133.3
    ## 5           1       90       1      2     33.3      f  23       5      33.3
    ## 6           1       90       0      2     16.7      f  23       6      16.7

### Fitting

If there are several participants, for which several models should be
fitted independently, then fitting the models is done using the
`fitConfModels` function. In the following example, we fit the SDT and
WEV model. With the argument `.parallel=TRUE` we use parallelization
over all but one available core.

``` r
 fitted_pars <- fitConfModels(MaskOri, models=c("SDT", "WEV"), .parallel = TRUE) 
```

This parallelizes the fitting process over participant-model
combinations. The output is then a data frame with one row for each
participant-model combination and columns for parameters and measures
for model performance (negative log-likelihood, BIC, AIC and AICc).
These may be used for quantitative model comparison.

``` r
head(fitted_pars)
```

    ##   model participant negLogLik   N  k      BIC     AICc      AIC         d1
    ## 1   SDT           1  989.8068 960 11 2055.150 2001.846 2001.614 0.15071384
    ## 2   WEV           1  913.0819 960 13 1915.434 1852.494 1852.164 0.13854199
    ## 3   SDT           2 1076.5072 960 11 2228.551 2175.246 2175.014 0.24277316
    ## 4   WEV           2 1053.6981 960 13 2196.666 2133.726 2133.396 0.37470202
    ## 5   SDT           3  890.3000 960 11 1856.136 1802.832 1802.600 0.01609433
    ## 6   WEV           3  767.7225 960 13 1624.715 1561.775 1561.445 0.00244278
    ##          d2       d3       d4       theta         w     sigma        cA1
    ## 1 1.1622992 5.866728 8.625396  0.38575591        NA        NA -1.7634060
    ## 2 1.4499134 4.233249 5.506721  0.39877878 0.4147619 0.5123999 -0.8207672
    ## 3 1.4138645 4.925860 7.691836  0.02784248        NA        NA -2.1061454
    ## 4 1.4442048 4.073919 5.971996  0.02595525 0.2332637 0.4086341 -1.4745243
    ## 5 0.1404938 2.947725 7.086975 -1.26647186        NA        NA -2.2868067
    ## 6 0.4075982 3.206503 5.926012 -1.33589282 0.4360688 0.6875500 -1.9056750
    ##          cA2         cA3        cB1       cB2      cB3
    ## 1 -1.1169610 -0.12542605  1.0221640 1.7691779 2.183439
    ## 2 -0.2423225  0.82897295 -0.1272569 0.6705370 1.086434
    ## 3 -1.5485703 -0.90972112  0.7944319 1.5473073 1.995515
    ## 4 -0.9671368 -0.38593372  0.2404484 0.9270251 1.339125
    ## 5 -1.7878067 -1.35670379  0.8875427 1.7188347 2.410818
    ## 6 -1.2341607 -0.07619421  0.2861275 1.2793275 2.018620

## Future development

The package is still new and under development. We plan to include more
models from the literature, a function to compute measures of
metacognitive sensitivity (e.g. meta-d’) and function to compute the
predicted distribution of responses for given parameter sets. For any
suggestions and contributions, please contact us!

## Contact

For comments, remarks, and questions please contact either
<manuel.rausch@hochschule-rhein-waal.de> or <sebastian.hellmann@ku.de>
or [submit an issue](https://github.com/SeHellmann/StatConfR/issues).

## References

Rausch, M., Hellmann, S. & Zehetleitner, M. (2018). Confidence in masked
orientation judgments is informed by both evidence and visibility.
*Atten Percept Psychophys*, 80, 134–154. [doi:
10.3758/s13414-017-1431-5](https://doi.org/10.3758/s13414-017-1431-5)
