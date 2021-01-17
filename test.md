@def title = "Bitcoin Price EDA"
@def hascode = true

# Bitcoin volatility

```r
library(data.table)
library(ggplot2)
```


```r
pth <- "../input/bitstampUSD_1-min_data/"
dt <- fread(paste0(pth,"bitstampUSD_1-min_data_2012-01-01_to_2020-12-31.csv"))
```


```r
names(dt) <- gsub(pattern = "\\(",replacement = "",x = names(dt))
names(dt) <- gsub(pattern = "\\)",replacement = "",x = names(dt))
```



```r
dt
```

```
##           Timestamp     Open     High      Low    Close Volume_BTC Volume_Currency
##       1: 1325317920     4.39     4.39     4.39     4.39  0.4555809            2.00
##       2: 1325317980      NaN      NaN      NaN      NaN        NaN             NaN
##       3: 1325318040      NaN      NaN      NaN      NaN        NaN             NaN
##       4: 1325318100      NaN      NaN      NaN      NaN        NaN             NaN
##       5: 1325318160      NaN      NaN      NaN      NaN        NaN             NaN
##      ---                                                                          
## 4727773: 1609372560 28801.47 28829.42 28785.64 28829.42  0.9652210        27804.57
## 4727774: 1609372620 28829.42 28863.90 28829.42 28857.06  2.3688312        68332.35
## 4727775: 1609372680 28850.49 28900.52 28850.49 28882.82  2.4665898        71232.78
## 4727776: 1609372740 28910.54 28911.52 28867.60 28881.30  7.3327730       211870.91
## 4727777: 1609372800 28893.21 28928.49 28893.21 28928.49  5.7576794       166449.71
##          Weighted_Price
##       1:           4.39
##       2:            NaN
##       3:            NaN
##       4:            NaN
##       5:            NaN
##      ---               
## 4727773:       28806.43
## 4727774:       28846.44
## 4727775:       28879.06
## 4727776:       28893.70
## 4727777:       28909.17
```

Remove if nothing traded


```r
dt <- dt[!is.nan(Volume_BTC)]
dt
```

```
##           Timestamp     Open     High      Low    Close Volume_BTC Volume_Currency
##       1: 1325317920     4.39     4.39     4.39     4.39  0.4555809         2.00000
##       2: 1325346600     4.39     4.39     4.39     4.39 48.0000000       210.72000
##       3: 1325350740     4.50     4.57     4.50     4.57 37.8622972       171.38034
##       4: 1325350800     4.58     4.58     4.58     4.58  9.0000000        41.22000
##       5: 1325391360     4.58     4.58     4.58     4.58  1.5020000         6.87916
##      ---                                                                          
## 3484301: 1609372560 28801.47 28829.42 28785.64 28829.42  0.9652210     27804.57213
## 3484302: 1609372620 28829.42 28863.90 28829.42 28857.06  2.3688312     68332.35063
## 3484303: 1609372680 28850.49 28900.52 28850.49 28882.82  2.4665898     71232.78446
## 3484304: 1609372740 28910.54 28911.52 28867.60 28881.30  7.3327730    211870.91266
## 3484305: 1609372800 28893.21 28928.49 28893.21 28928.49  5.7576794    166449.70932
##          Weighted_Price
##       1:       4.390000
##       2:       4.390000
##       3:       4.526411
##       4:       4.580000
##       5:       4.580000
##      ---               
## 3484301:   28806.429798
## 3484302:   28846.441863
## 3484303:   28879.056266
## 3484304:   28893.695831
## 3484305:   28909.166061
```


```r
dt[,TimestampUTC := as.POSIXct(Timestamp,origin = "1970-01-01",tz = "UTC")]
```


## Temporal Trend


```r
dt[,date := lubridate::date(TimestampUTC)]
dt[,time := lubridate::local_time(TimestampUTC,units="hours")]
```


```r
keep_rws <- c(dt$date[-nrow(dt)] != dt$date[-1],FALSE)
dt_daily_close <- dt[keep_rws]
dt_daily_close
```

```
##        Timestamp     Open     High      Low    Close Volume_BTC Volume_Currency
##    1: 1325350800     4.58     4.58     4.58     4.58   9.000000        41.22000
##    2: 1325457900     5.00     5.00     5.00     5.00  10.100000        50.50000
##    3: 1325534640     5.00     5.00     5.00     5.00  19.048000        95.24000
##    4: 1325611620     5.29     5.29     5.29     5.29   4.010815        21.21721
##    5: 1325699460     5.37     5.57     5.37     5.57  43.312196       235.74707
##   ---                                                                          
## 3281: 1609027140 26437.09 26466.28 26435.67 26466.28   1.339213     35419.34286
## 3282: 1609113540 26217.19 26259.60 26217.19 26259.60   3.264809     85632.31260
## 3283: 1609199940 27037.78 27050.00 27024.52 27037.91   3.536079     95595.13092
## 3284: 1609286340 27371.72 27377.85 27355.99 27370.00   1.873968     51289.98054
## 3285: 1609372740 28910.54 28911.52 28867.60 28881.30   7.332773    211870.91266
##       Weighted_Price        TimestampUTC       date           time
##    1:       4.580000 2011-12-31 17:00:00 2011-12-31 17.00000 hours
##    2:       5.000000 2012-01-01 22:45:00 2012-01-01 22.75000 hours
##    3:       5.000000 2012-01-02 20:04:00 2012-01-02 20.06667 hours
##    4:       5.290000 2012-01-03 17:27:00 2012-01-03 17.45000 hours
##    5:       5.442972 2012-01-04 17:51:00 2012-01-04 17.85000 hours
##   ---                                                             
## 3281:   26447.887700 2020-12-26 23:59:00 2020-12-26 23.98333 hours
## 3282:   26228.891247 2020-12-27 23:59:00 2020-12-27 23.98333 hours
## 3283:   27034.218026 2020-12-28 23:59:00 2020-12-28 23.98333 hours
## 3284:   27369.727449 2020-12-29 23:59:00 2020-12-29 23.98333 hours
## 3285:   28893.695831 2020-12-30 23:59:00 2020-12-30 23.98333 hours
```


```r
ggplot(dt_daily_close,aes(x=date,y=Close)) +
  geom_line()
```

![plot of chunk unnamed-chunk-9](/assets/unnamed-chunk-9-1.png)


```r
ggplot(dt_daily_close,aes(x=date,y=log(Close))) +
  geom_line()
```

![plot of chunk unnamed-chunk-10](/assets/unnamed-chunk-10-1.png)


```r
dt_daily_ret <- dt_daily_close[,.(return = diff(log(Close)))]
dt_daily_ret[,date := dt_daily_close$date[-1]]
ggplot(dt_daily_ret,aes(x=date,y=return)) +
         geom_line()
```

![plot of chunk unnamed-chunk-11](/assets/unnamed-chunk-11-1.png)


```r
ggplot(dt_daily_ret,aes(x=return)) +
         geom_histogram(aes(y=..density..),col="white",bins=50) +
stat_function(fun = dnorm, args = list(mean = mean(dt_daily_ret$return),
                                       sd = sd(dt_daily_ret$return)))
```

![plot of chunk unnamed-chunk-12](/assets/unnamed-chunk-12-1.png)


```r
acf(dt_daily_ret$return)
```

![plot of chunk unnamed-chunk-13](/assets/unnamed-chunk-13-1.png)


```r
pacf(dt_daily_ret$return)
```

![plot of chunk unnamed-chunk-14](/assets/unnamed-chunk-14-1.png)


```r
acf(dt_daily_ret$return^2)
```

![plot of chunk unnamed-chunk-15](/assets/unnamed-chunk-15-1.png)


```r
Box.test(dt_daily_ret$return^2)
```

```
## 
## 	Box-Pierce test
## 
## data:  dt_daily_ret$return^2
## X-squared = 398.96, df = 1, p-value < 2.2e-16
```

### ARCH models


```r
a0 <- 0.2
a1 <- 0.4
n <- 1000L
ee <- rnorm(n+1)
zt <- numeric(length = n+1)
zt[1] <- ee[1]
for (i in 2:(n+1)) {
  ht <- a0 + a1*zt[i-1]^2
  zt[i] <- ee[i]*sqrt(ht)
}
plot(zt,type="l")
```

![plot of chunk unnamed-chunk-17](/assets/unnamed-chunk-17-1.png)


```r
acf(zt^2)
```

![plot of chunk unnamed-chunk-18](/assets/unnamed-chunk-18-1.png)



```r
library(fGarch)
```

```
## Loading required package: timeDate
```

```
## Loading required package: timeSeries
```

```
## Loading required package: fBasics
```

```r
m1 <- garchFit(~arma(0,0)+garch(2,0),dt_daily_ret$return)
```

```
## 
## Series Initialization:
##  ARMA Model:                arma
##  Formula Mean:              ~ arma(0, 0)
##  GARCH Model:               garch
##  Formula Variance:          ~ garch(2, 0)
##  ARMA Order:                0 0
##  Max ARMA Order:            0
##  GARCH Order:               2 0
##  Max GARCH Order:           2
##  Maximum Order:             2
##  Conditional Dist:          norm
##  h.start:                   3
##  llh.start:                 1
##  Length of Series:          3284
##  Recursion Init:            mci
##  Series Scale:              0.0463792
## 
## Parameter Initialization:
##  Initial Parameters:          $params
##  Limits of Transformations:   $U, $V
##  Which Parameters are Fixed?  $includes
##  Parameter Matrix:
##                      U           V     params includes
##     mu     -0.57443969   0.5744397 0.05744397     TRUE
##     omega   0.00000100 100.0000000 0.10000000     TRUE
##     alpha1  0.00000001   1.0000000 0.05000000     TRUE
##     alpha2  0.00000001   1.0000000 0.05000000     TRUE
##     gamma1 -0.99999999   1.0000000 0.10000000    FALSE
##     gamma2 -0.99999999   1.0000000 0.10000000    FALSE
##     delta   0.00000000   2.0000000 2.00000000    FALSE
##     skew    0.10000000  10.0000000 1.00000000    FALSE
##     shape   1.00000000  10.0000000 4.00000000    FALSE
##  Index List of Parameters to be Optimized:
##     mu  omega alpha1 alpha2 
##      1      2      3      4 
##  Persistence:                  0.1 
## 
## 
## --- START OF TRACE ---
## Selected Algorithm: nlminb 
## 
## R coded nlminb Solver: 
## 
##   0:     8046.2248: 0.0574440 0.100000 0.0500000 0.0500000
##   1:     4450.5719: 0.0574406 0.987635 0.373533 0.377766
##   2:     4441.8605: 0.0576335 0.762999 0.788219 0.543788
##   3:     4290.8192: 0.0589024 0.395512 0.562630 0.795928
##   4:     4233.2962: 0.0593005 0.359727 0.190320 0.464179
##   5:     4230.2769: 0.0594168 0.595855 0.349403 0.0531660
##   6:     4197.3473: 0.0593731 0.532303 0.319638 0.293113
##   7:     4189.0905: 0.0593673 0.493945 0.299758 0.276815
##   8:     4187.2702: 0.0588991 0.474727 0.266591 0.252396
##   9:     4187.1295: 0.0574901 0.507977 0.249612 0.240702
##  10:     4186.7721: 0.0564879 0.493179 0.247908 0.243303
##  11:     4186.7508: 0.0564780 0.496334 0.253101 0.252115
##  12:     4186.7051: 0.0569370 0.489443 0.251812 0.253377
##  13:     4186.6979: 0.0575401 0.491251 0.251173 0.254219
##  14:     4186.6975: 0.0569273 0.491562 0.250399 0.253936
##  15:     4186.6973: 0.0568848 0.491252 0.251229 0.254118
##  16:     4186.6972: 0.0569497 0.491307 0.251026 0.253907
##  17:     4186.6972: 0.0570156 0.491354 0.250879 0.254061
##  18:     4186.6972: 0.0570462 0.491377 0.250964 0.253975
##  19:     4186.6972: 0.0570402 0.491363 0.250953 0.253978
##  20:     4186.6972: 0.0570402 0.491363 0.250953 0.253978
## 
## Final Estimate of the Negative LLH:
##  LLH:  -5898.152    norm LLH:  -1.796027 
##          mu       omega      alpha1      alpha2 
## 0.002645480 0.001056936 0.250952557 0.253978481 
## 
## R-optimhess Difference Approximated Hessian Matrix:
##                   mu         omega       alpha1       alpha2
## mu     -2351850.0340     -94705.65    -130.9563    1516.1290
## omega    -94705.6536 -877581129.05 -373655.7934 -444900.1054
## alpha1     -130.9563    -373655.79   -1471.6057    -431.9747
## alpha2     1516.1290    -444900.11    -431.9747   -1191.6592
## attr(,"time")
## Time difference of 0.05965304 secs
## 
## --- END OF TRACE ---
## 
## 
## Time to Estimate Parameters:
##  Time difference of 0.4027209 secs
```

```
## Warning: Using formula(x) is deprecated when x is a character vector of length > 1.
##   Consider formula(paste(x, collapse = " ")) instead.
```

```r
summary(m1)
```

```
## 
## Title:
##  GARCH Modelling 
## 
## Call:
##  garchFit(formula = ~arma(0, 0) + garch(2, 0), data = dt_daily_ret$return) 
## 
## Mean and Variance Equation:
##  data ~ arma(0, 0) + garch(2, 0)
## <environment: 0x000001b4827aa428>
##  [data = dt_daily_ret$return]
## 
## Conditional Distribution:
##  norm 
## 
## Coefficient(s):
##        mu      omega     alpha1     alpha2  
## 0.0026455  0.0010569  0.2509526  0.2539785  
## 
## Std. Errors:
##  based on Hessian 
## 
## Error Analysis:
##         Estimate  Std. Error  t value Pr(>|t|)    
## mu     2.645e-03   6.524e-04    4.055 5.02e-05 ***
## omega  1.057e-03   3.843e-05   27.502  < 2e-16 ***
## alpha1 2.510e-01   2.827e-02    8.878  < 2e-16 ***
## alpha2 2.540e-01   3.296e-02    7.705 1.31e-14 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Log Likelihood:
##  5898.152    normalized:  1.796027 
## 
## Description:
##  Fri Jan 08 10:29:47 2021 by user: z5110862 
## 
## 
## Standardised Residuals Tests:
##                                 Statistic p-Value     
##  Jarque-Bera Test   R    Chi^2  58727.98  0           
##  Shapiro-Wilk Test  R    W      0.8800886 0           
##  Ljung-Box Test     R    Q(10)  26.3782   0.003263477 
##  Ljung-Box Test     R    Q(15)  39.05692  0.000628423 
##  Ljung-Box Test     R    Q(20)  49.41108  0.0002687736
##  Ljung-Box Test     R^2  Q(10)  14.15045  0.1662389   
##  Ljung-Box Test     R^2  Q(15)  18.71158  0.2271022   
##  Ljung-Box Test     R^2  Q(20)  20.63017  0.4191803   
##  LM Arch Test       R    TR^2   15.36755  0.2219489   
## 
## Information Criterion Statistics:
##       AIC       BIC       SIC      HQIC 
## -3.589618 -3.582192 -3.589621 -3.586959
```
