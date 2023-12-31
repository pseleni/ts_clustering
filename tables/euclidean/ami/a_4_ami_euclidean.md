AMI of UCR datasets for Algorithm 3 (the baseline computed by standard $k$-means) and Algorithm 1 (Sparse Time-Series Clustering)
with $(10^{-2}m / T)$-DTW and number of inducing points $m = \gamma \log_2 T$, for every $\gamma \in \{1, 2, 3, 4, 5\}$. The best AMI achieved is marked in bold. In parenthesis, we report the relative order of the corresponding AMI among the six ones reported, from $1$ (best) to $6$ (worst). In the last line, we report the average AMI (and the average relative order) for each column across all datasets.

| **dataset**                        | **baseline**       | **$1\cdot \log{T}$** | **$2\cdot \log{T}$** | **$3\cdot \log{T}$** | **$4\cdot \log{T}$** | **$5\cdot \log{T}$** |
|:-----------------------------------|:------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
| **Adiac**                          | **0.470**(1) | 0.452 (2)                     | 0.439 (3)                     | 0.383 (6)                     | 0.419 (5)                     | 0.435 (4)                     |
| **ArrowHead**                      | 0.243 (3)          | 0.191 (5)                     | 0.180 (6)                     | 0.249 (2)                     | 0.235 (4)                     | **0.252**(1)            |
| **Beef**                           | 0.208 (4)          | 0.183 (6)                     | **0.232**(1)            | 0.228 (2)                     | 0.219 (3)                     | 0.203 (5)                     |
| **BeetleFly**                      | 0.017 (4)          | **0.088**(1)            | -0.004 (6)                    | 0.042 (2)                     | 0.016 (5)                     | 0.023 (3)                     |
| **BirdChicken**                    | **0.034**(1) | 0.006 (5)                     | 0.005 (6)                     | 0.027 (3)                     | 0.034 (2)                     | 0.013 (4)                     |
| **Car**                            | 0.203 (5)          | **0.294**(1)            | 0.274 (2)                     | 0.233 (3)                     | 0.166 (6)                     | 0.204 (4)                     |
| **CBF**                            | 0.353 (3)          | 0.233 (6)                     | 0.331 (5)                     | 0.340 (4)                     | **0.369**(1)            | 0.356 (2)                     |
| **Coffee**                         | 0.424 (3)          | 0.211 (4)                     | 0.074 (6)                     | 0.088 (5)                     | 0.440 (2)                     | **0.574**(1)            |
| **Computers**                      | 0.000 (3)          | **0.002**(1)            | 0.001 (2)                     | -0.001 (5)                    | -0.001 (6)                    | -0.001 (4)                    |
| **CricketX**                       | 0.218 (5)          | 0.202 (6)                     | 0.222 (3)                     | **0.228**(1)            | 0.227 (2)                     | 0.220 (4)                     |
| **CricketY**                       | 0.272 (2)          | 0.231 (6)                     | 0.246 (5)                     | 0.267 (3)                     | 0.266 (4)                     | **0.275**(1)            |
| **CricketZ**                       | 0.221 (4)          | 0.219 (5)                     | 0.216 (6)                     | 0.224 (3)                     | 0.231 (2)                     | **0.231**(1)            |
| **DiatomSizeReduction**            | **0.813**(1) | 0.473 (5)                     | 0.464 (6)                     | 0.671 (4)                     | 0.732 (3)                     | 0.740 (2)                     |
| **DistalPhalanxOutlineCorrect**    | -0.001 (6)         | 0.000 (3)                     | **0.004**(1)            | 0.003 (2)                     | -0.001 (5)                    | -0.000 (4)                    |
| **DistalPhalanxOutlineAgeGroup**   | 0.311 (3)          | 0.321 (2)                     | 0.303 (5)                     | 0.287 (6)                     | **0.346**(1)            | 0.306 (4)                     |
| **DistalPhalanxTW**                | 0.440 (2)          | 0.430 (4)                     | **0.443**(1)            | 0.416 (6)                     | 0.437 (3)                     | 0.420 (5)                     |
| **Earthquakes**                    | **0.002**(1) | -0.001 (5)                    | -0.001 (6)                    | -0.001 (4)                    | -0.001 (2)                    | -0.001 (3)                    |
| **ECG200**                         | 0.097 (4)          | 0.109 (2)                     | **0.176**(1)            | 0.067 (6)                     | 0.080 (5)                     | 0.101 (3)                     |
| **ECGFiveDays**                    | 0.000 (3)          | **0.087**(1)            | 0.008 (2)                     | -0.001 (6)                    | -0.000 (5)                    | -0.000 (4)                    |
| **FaceAll**                        | 0.348 (2)          | 0.213 (6)                     | 0.279 (5)                     | 0.297 (4)                     | 0.317 (3)                     | **0.353**(1)            |
| **FaceFour**                       | 0.412 (2)          | 0.087 (6)                     | 0.273 (5)                     | 0.328 (3)                     | 0.312 (4)                     | **0.422**(1)            |
| **FacesUCR**                       | **0.351**(1) | 0.209 (6)                     | 0.281 (5)                     | 0.296 (4)                     | 0.302 (3)                     | 0.340 (2)                     |
| **FiftyWords**                     | 0.476 (2)          | 0.426 (6)                     | 0.449 (5)                     | 0.465 (4)                     | 0.468 (3)                     | **0.478**(1)            |
| **Fish**                           | 0.280 (2)          | **0.289**(1)            | 0.265 (4)                     | 0.243 (6)                     | 0.269 (3)                     | 0.260 (5)                     |
| **GunPoint**                       | -0.004 (6)         | **-0.000**(1)           | -0.003 (2)                    | -0.003 (3)                    | -0.004 (5)                    | -0.004 (4)                    |
| **Ham**                            | **0.038**(1) | 0.002 (5)                     | 0.003 (4)                     | 0.027 (2)                     | -0.003 (6)                    | 0.010 (3)                     |
| **Herring**                        | 0.001 (2)          | -0.001 (3)                    | -0.001 (4)                    | -0.005 (5)                    | **0.002**(1)            | -0.005 (6)                    |
| **InsectWingbeatSound**            | **0.514**(1) | 0.352 (6)                     | 0.405 (5)                     | 0.425 (4)                     | 0.463 (3)                     | 0.464 (2)                     |
| **ItalyPowerDemand**               | 0.001 (4)          | **0.041**(1)            | 0.017 (2)                     | 0.001 (5)                     | 0.002 (3)                     | -0.000 (6)                    |
| **LargeKitchenAppliances**         | 0.025 (6)          | 0.031 (4)                     | 0.038 (2)                     | **0.040**(1)            | 0.038 (3)                     | 0.030 (5)                     |
| **Lightning2**                     | 0.045 (5)          | 0.022 (6)                     | 0.066 (4)                     | 0.068 (3)                     | 0.068 (2)                     | **0.068**(1)            |
| **Lightning7**                     | **0.360**(1) | 0.357 (3)                     | 0.343 (6)                     | 0.348 (5)                     | 0.354 (4)                     | 0.357 (2)                     |
| **Meat**                           | **0.637**(1) | 0.382 (3)                     | 0.161 (5)                     | 0.065 (6)                     | 0.498 (2)                     | 0.288 (4)                     |
| **MedicalImages**                  | **0.234**(1) | 0.134 (6)                     | 0.217 (4)                     | 0.201 (5)                     | 0.225 (2)                     | 0.222 (3)                     |
| **MiddlePhalanxOutlineAgeGroup**   | **0.394**(1) | 0.347 (4)                     | 0.282 (5)                     | 0.251 (6)                     | 0.386 (3)                     | 0.394 (2)                     |
| **MiddlePhalanxOutlineCorrect**    | -0.000 (3)         | **0.013**(1)            | 0.007 (2)                     | -0.001 (5)                    | -0.000 (4)                    | -0.001 (6)                    |
| **MiddlePhalanxTW**                | **0.406**(1) | 0.401 (3)                     | 0.317 (6)                     | 0.326 (5)                     | 0.391 (4)                     | 0.405 (2)                     |
| **MoteStrain**                     | **0.305**(1) | 0.280 (2)                     | 0.265 (3)                     | 0.239 (6)                     | 0.255 (5)                     | 0.262 (4)                     |
| **OliveOil**                       | **0.543**(1) | 0.240 (2)                     | 0.166 (3)                     | 0.050 (4)                     | -0.023 (6)                    | -0.018 (5)                    |
| **OSULeaf**                        | **0.195**(1) | 0.095 (6)                     | 0.167 (5)                     | 0.188 (3)                     | 0.176 (4)                     | 0.190 (2)                     |
| **PhalangesOutlinesCorrect**       | **0.010**(1) | -0.000 (6)                    | 0.003 (4)                     | 0.002 (5)                     | 0.007 (3)                     | 0.010 (2)                     |
| **Plane**                          | 0.832 (2)          | 0.726 (6)                     | 0.827 (3)                     | 0.782 (5)                     | 0.821 (4)                     | **0.840**(1)            |
| **ProximalPhalanxOutlineCorrect**  | **0.085**(1) | 0.072 (3)                     | 0.035 (5)                     | 0.032 (6)                     | 0.038 (4)                     | 0.084 (2)                     |
| **ProximalPhalanxOutlineAgeGroup** | **0.463**(1) | 0.450 (3)                     | 0.380 (5)                     | 0.240 (6)                     | 0.426 (4)                     | 0.453 (2)                     |
| **ProximalPhalanxTW**              | 0.487 (2)          | 0.460 (3)                     | 0.458 (4)                     | 0.437 (5)                     | 0.432 (6)                     | **0.505**(1)            |
| **RefrigerationDevices**           | 0.001 (6)          | **0.014**(1)            | 0.009 (4)                     | 0.011 (3)                     | 0.013 (2)                     | 0.009 (5)                     |
| **ShapeletSim**                    | -0.001 (2)         | **-0.001**(1)           | -0.002 (4)                    | -0.001 (3)                    | -0.002 (5)                    | -0.003 (6)                    |
| **ShapesAll**                      | **0.559**(1) | 0.439 (6)                     | 0.497 (5)                     | 0.536 (4)                     | 0.544 (3)                     | 0.555 (2)                     |
| **SmallKitchenAppliances**         | 0.010 (6)          | 0.068 (2)                     | **0.074**(1)            | 0.065 (4)                     | 0.059 (5)                     | 0.068 (3)                     |
| **SonyAIBORobotSurface1**          | 0.406 (2)          | 0.016 (6)                     | 0.185 (4)                     | 0.067 (5)                     | 0.242 (3)                     | **0.437**(1)            |
| **SonyAIBORobotSurface2**          | 0.238 (3)          | 0.010 (6)                     | 0.225 (5)                     | **0.263**(1)            | 0.238 (2)                     | 0.238 (4)                     |
| **Strawberry**                     | 0.123 (3)          | **0.137**(1)            | 0.006 (5)                     | 0.130 (2)                     | 0.002 (6)                     | 0.073 (4)                     |
| **SwedishLeaf**                    | **0.515**(1) | 0.307 (6)                     | 0.449 (3)                     | 0.430 (5)                     | 0.441 (4)                     | 0.474 (2)                     |
| **Symbols**                        | 0.765 (4)          | 0.727 (6)                     | 0.768 (3)                     | **0.781**(1)            | 0.761 (5)                     | 0.778 (2)                     |
| **SyntheticControl**               | 0.770 (4)          | 0.640 (6)                     | 0.770 (5)                     | 0.779 (3)                     | 0.779 (2)                     | **0.780**(1)            |
| **ToeSegmentation1**               | -0.002 (4)         | **0.002**(1)            | -0.002 (6)                    | -0.002 (2)                    | -0.002 (3)                    | -0.002 (5)                    |
| **ToeSegmentation2**               | 0.002 (2)          | **0.003**(1)            | -0.000 (6)                    | 0.001 (4)                     | 0.001 (3)                     | -0.000 (5)                    |
| **Trace**                          | 0.508 (6)          | **0.538**(1)            | 0.527 (2)                     | 0.513 (3)                     | 0.512 (5)                     | 0.513 (4)                     |
| **TwoLeadECG**                     | 0.003 (5)          | 0.021 (2)                     | 0.005 (4)                     | 0.016 (3)                     | 0.001 (6)                     | **0.034**(1)            |
| **TwoPatterns**                    | 0.019 (5)          | 0.007 (6)                     | 0.020 (3)                     | 0.021 (2)                     | **0.024**(1)            | 0.019 (4)                     |
| **Wine**                           | -0.006 (4)         | -0.006 (6)                    | -0.004 (2)                    | -0.006 (5)                    | **-0.002**(1)           | -0.005 (3)                    |
| **WordSynonyms**                   | 0.332 (2)          | 0.318 (5)                     | 0.315 (6)                     | 0.321 (4)                     | 0.325 (3)                     | **0.333**(1)            |
| **Worms**                          | 0.025 (2)          | **0.036**(1)            | 0.024 (4)                     | 0.016 (6)                     | 0.018 (5)                     | 0.024 (3)                     |
| **avg**                            | 0.254 (2.730)      | 0.201 (3.746)                 | 0.209 (4.000)                 | 0.207 (3.952)                 | 0.228 (3.556)                 | 0.239 (3.016)                 |