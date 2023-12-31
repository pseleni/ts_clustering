Best AMI of UCR datasets for Algorithm 3 (the baseline computed by standard $k$-means) and Algorithm 1 (Sparse Time-Series Clustering)
with number of inducing points $m = \gamma \log_2 T$, for $\gamma \in \{1, 2, 3, 4, 5\}$ and $(\alpha m / T)$-DTW. Averages are computed over AMI values of 10 different runs (for the baseline) and bests over AMI values of 10 different runs for each different value of $\alpha \{ 0, 10^{-4}, 10^{-3}, 10^{-2**\}}$ for the sparse framework. The best average AMI is reported in bold. In parenthesis, we report the relative order of the corresponding average AMI among the six ones reported, from $1$ (best) to $6$ (worst). In the last line, we report the average AMI (and the average relative order) for each column across all datasets. 

| **dataset**                        | **baseline**       | **$1\cdot \log{T}$** | **$2\cdot \log{T}$** | **$3\cdot \log{T}$** | **$4\cdot \log{T}$** | **$5\cdot \log{T}$** |
|:-----------------------------------|:------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
| **Adiac**                          | **0.487**(1) | 0.452 (2)                     | 0.445 (3)                     | 0.400 (6)                     | 0.428 (5)                     | 0.433 (4)                     |
| **ArrowHead**                      | 0.245 (4)          | 0.256 (2)                     | 0.231 (6)                     | **0.262**(1)            | 0.236 (5)                     | 0.249 (3)                     |
| **Beef**                           | **0.223**(1) | 0.185 (6)                     | 0.207 (2)                     | 0.198 (4)                     | 0.198 (3)                     | 0.197 (5)                     |
| **BeetleFly**                      | 0.080 (2)          | 0.030 (6)                     | 0.048 (5)                     | 0.074 (3)                     | 0.067 (4)                     | **0.100**(1)            |
| **BirdChicken**                    | 0.012 (5)          | **0.047**(1)            | 0.005 (6)                     | 0.024 (2)                     | 0.013 (4)                     | 0.018 (3)                     |
| **Car**                            | 0.161 (2)          | **0.331**(1)            | 0.132 (6)                     | 0.148 (5)                     | 0.156 (3)                     | 0.156 (4)                     |
| **CBF**                            | 0.715 (3)          | 0.582 (6)                     | **0.727**(1)            | 0.717 (2)                     | 0.644 (4)                     | 0.590 (5)                     |
| **Coffee**                         | 0.566 (2)          | 0.211 (6)                     | 0.530 (3)                     | 0.514 (4)                     | 0.345 (5)                     | **0.723**(1)            |
| **Computers**                      | 0.035 (5)          | 0.018 (6)                     | 0.046 (2)                     | 0.041 (4)                     | **0.053**(1)            | 0.044 (3)                     |
| **CricketX**                       | **0.404**(1) | 0.241 (6)                     | 0.317 (4)                     | 0.310 (5)                     | 0.348 (3)                     | 0.357 (2)                     |
| **CricketY**                       | **0.401**(1) | 0.281 (6)                     | 0.346 (5)                     | 0.382 (2)                     | 0.370 (4)                     | 0.380 (3)                     |
| **CricketZ**                       | **0.401**(1) | 0.247 (6)                     | 0.331 (5)                     | 0.336 (4)                     | 0.336 (3)                     | 0.370 (2)                     |
| **DiatomSizeReduction**            | **0.916**(1) | 0.445 (6)                     | 0.648 (4)                     | 0.767 (2)                     | 0.711 (3)                     | 0.564 (5)                     |
| **DistalPhalanxOutlineCorrect**    | 0.003 (3)          | **0.006**(1)            | 0.004 (2)                     | -0.001 (6)                    | -0.001 (5)                    | -0.000 (4)                    |
| **DistalPhalanxOutlineAgeGroup**   | 0.380 (2)          | 0.321 (6)                     | **0.396**(1)            | 0.349 (5)                     | 0.355 (4)                     | 0.369 (3)                     |
| **DistalPhalanxTW**                | **0.504**(1) | 0.472 (5)                     | 0.480 (3)                     | 0.470 (6)                     | 0.475 (4)                     | 0.481 (2)                     |
| **Earthquakes**                    | **0.035**(1) | 0.003 (4)                     | 0.003 (3)                     | 0.001 (6)                     | 0.004 (2)                     | 0.002 (5)                     |
| **ECG200**                         | 0.061 (4)          | **0.089**(1)            | 0.081 (2)                     | 0.066 (3)                     | 0.052 (6)                     | 0.058 (5)                     |
| **ECGFiveDays**                    | 0.006 (5)          | **0.088**(1)            | 0.035 (4)                     | 0.073 (2)                     | 0.003 (6)                     | 0.057 (3)                     |
| **FaceAll**                        | 0.627 (2)          | 0.228 (6)                     | 0.414 (5)                     | 0.551 (4)                     | 0.589 (3)                     | **0.688**(1)            |
| **FaceFour**                       | 0.417 (3)          | 0.037 (6)                     | 0.427 (2)                     | 0.371 (5)                     | 0.415 (4)                     | **0.524**(1)            |
| **FacesUCR**                       | 0.622 (2)          | 0.226 (6)                     | 0.408 (5)                     | 0.550 (4)                     | 0.582 (3)                     | **0.685**(1)            |
| **FiftyWords**                     | 0.524 (5)          | 0.469 (6)                     | 0.560 (4)                     | 0.594 (3)                     | 0.610 (2)                     | **0.611**(1)            |
| **Fish**                           | **0.393**(1) | 0.313 (4)                     | 0.289 (6)                     | 0.339 (3)                     | 0.308 (5)                     | 0.348 (2)                     |
| **GunPoint**                       | -0.003 (4)         | **0.026**(1)            | -0.003 (6)                    | -0.002 (3)                    | -0.002 (2)                    | -0.003 (5)                    |
| **Ham**                            | 0.024 (4)          | -0.002 (6)                    | 0.026 (3)                     | **0.035**(1)            | 0.030 (2)                     | 0.021 (5)                     |
| **Herring**                        | 0.010 (3)          | -0.004 (6)                    | -0.003 (5)                    | 0.008 (4)                     | **0.020**(1)            | 0.017 (2)                     |
| **InsectWingbeatSound**            | 0.138 (6)          | 0.346 (5)                     | 0.408 (4)                     | 0.427 (3)                     | 0.437 (2)                     | **0.438**(1)            |
| **ItalyPowerDemand**               | 0.006 (2)          | **0.039**(1)            | 0.003 (3)                     | 0.003 (4)                     | 0.002 (6)                     | 0.002 (5)                     |
| **LargeKitchenAppliances**         | **0.172**(1) | 0.073 (4)                     | 0.078 (2)                     | 0.071 (5)                     | 0.056 (6)                     | 0.074 (3)                     |
| **Lightning2**                     | 0.059 (2)          | 0.049 (4)                     | 0.050 (3)                     | 0.032 (5)                     | 0.022 (6)                     | **0.084**(1)            |
| **Lightning7**                     | 0.421 (2)          | 0.392 (4)                     | 0.378 (5)                     | 0.405 (3)                     | 0.377 (6)                     | **0.422**(1)            |
| **Meat**                           | **0.686**(1) | 0.387 (3)                     | 0.200 (5)                     | 0.178 (6)                     | 0.516 (2)                     | 0.352 (4)                     |
| **MedicalImages**                  | **0.301**(1) | 0.211 (6)                     | 0.238 (5)                     | 0.248 (4)                     | 0.257 (3)                     | 0.274 (2)                     |
| **MiddlePhalanxOutlineAgeGroup**   | **0.396**(1) | 0.347 (6)                     | 0.361 (4)                     | 0.356 (5)                     | 0.383 (3)                     | 0.392 (2)                     |
| **MiddlePhalanxOutlineCorrect**    | 0.003 (3)          | **0.011**(1)            | 0.006 (2)                     | 0.002 (4)                     | -0.000 (6)                    | -0.000 (5)                    |
| **MiddlePhalanxTW**                | 0.399 (2)          | **0.401**(1)            | 0.363 (6)                     | 0.390 (5)                     | 0.392 (4)                     | 0.396 (3)                     |
| **MoteStrain**                     | 0.019 (6)          | 0.324 (4)                     | 0.330 (3)                     | 0.313 (5)                     | 0.342 (2)                     | **0.371**(1)            |
| **OliveOil**                       | **0.485**(1) | 0.245 (2)                     | 0.157 (3)                     | 0.034 (5)                     | -0.040 (6)                    | 0.048 (4)                     |
| **OSULeaf**                        | 0.219 (3)          | 0.106 (6)                     | 0.226 (2)                     | 0.203 (5)                     | 0.210 (4)                     | **0.229**(1)            |
| **PhalangesOutlinesCorrect**       | 0.003 (5)          | **0.014**(1)            | 0.002 (6)                     | 0.006 (4)                     | 0.009 (2)                     | 0.009 (3)                     |
| **Plane**                          | **0.914**(1) | 0.804 (6)                     | 0.856 (5)                     | 0.877 (4)                     | 0.906 (3)                     | 0.910 (2)                     |
| **ProximalPhalanxOutlineCorrect**  | 0.065 (4)          | 0.072 (3)                     | 0.058 (6)                     | 0.085 (2)                     | 0.063 (5)                     | **0.086**(1)            |
| **ProximalPhalanxOutlineAgeGroup** | **0.481**(1) | 0.467 (4)                     | 0.472 (2)                     | 0.381 (6)                     | 0.458 (5)                     | 0.469 (3)                     |
| **ProximalPhalanxTW**              | **0.516**(1) | 0.489 (4)                     | 0.510 (2)                     | 0.495 (3)                     | 0.486 (5)                     | 0.481 (6)                     |
| **RefrigerationDevices**           | **0.078**(1) | 0.012 (6)                     | 0.015 (5)                     | 0.032 (4)                     | 0.042 (2)                     | 0.035 (3)                     |
| **ShapeletSim**                    | 0.004 (3)          | -0.001 (5)                    | -0.002 (6)                    | 0.002 (4)                     | 0.015 (2)                     | **0.038**(1)            |
| **ShapesAll**                      | 0.566 (4)          | 0.482 (6)                     | 0.553 (5)                     | 0.582 (3)                     | 0.589 (2)                     | **0.591**(1)            |
| **SmallKitchenAppliances**         | **0.224**(1) | 0.069 (6)                     | 0.077 (5)                     | 0.086 (4)                     | 0.096 (3)                     | 0.098 (2)                     |
| **SonyAIBORobotSurface1**          | **0.626**(1) | 0.461 (6)                     | 0.482 (4)                     | 0.499 (2)                     | 0.469 (5)                     | 0.491 (3)                     |
| **SonyAIBORobotSurface2**          | 0.073 (5)          | 0.064 (6)                     | 0.290 (2)                     | **0.293**(1)            | 0.251 (3)                     | 0.232 (4)                     |
| **Strawberry**                     | 0.114 (3)          | **0.137**(1)            | 0.055 (6)                     | 0.127 (2)                     | 0.088 (5)                     | 0.095 (4)                     |
| **SwedishLeaf**                    | 0.559 (4)          | 0.353 (6)                     | 0.537 (5)                     | 0.590 (3)                     | 0.625 (2)                     | **0.636**(1)            |
| **Symbols**                        | 0.820 (3)          | 0.802 (5)                     | 0.792 (6)                     | **0.826**(1)            | 0.821 (2)                     | 0.808 (4)                     |
| **SyntheticControl**               | **0.884**(1) | 0.786 (6)                     | 0.837 (5)                     | 0.866 (2)                     | 0.861 (4)                     | 0.864 (3)                     |
| **ToeSegmentation1**               | 0.015 (5)          | 0.003 (6)                     | 0.018 (3)                     | **0.036**(1)            | 0.024 (2)                     | 0.017 (4)                     |
| **ToeSegmentation2**               | 0.015 (6)          | 0.037 (3)                     | **0.043**(1)            | 0.034 (4)                     | 0.042 (2)                     | 0.032 (5)                     |
| **Trace**                          | **0.702**(1) | 0.631 (6)                     | 0.679 (5)                     | 0.687 (4)                     | 0.695 (2)                     | 0.691 (3)                     |
| **TwoLeadECG**                     | 0.055 (5)          | 0.012 (6)                     | **0.736**(1)            | 0.110 (4)                     | 0.469 (2)                     | 0.336 (3)                     |
| **TwoPatterns**                    | **0.858**(1) | 0.357 (6)                     | 0.711 (5)                     | 0.841 (3)                     | 0.833 (4)                     | 0.848 (2)                     |
| **Wine**                           | -0.004 (3)         | -0.006 (6)                    | -0.004 (4)                    | -0.003 (2)                    | **-0.002**(1)           | -0.006 (5)                    |
| **WordSynonyms**                   | 0.408 (5)          | 0.358 (6)                     | 0.433 (4)                     | 0.465 (3)                     | **0.479**(1)            | 0.479 (2)                     |
| **Worms**                          | 0.140 (3)          | 0.048 (6)                     | 0.107 (5)                     | 0.134 (4)                     | 0.144 (2)                     | **0.144**(1)            |
| **avg**                            | 0.312 (2.651)      | 0.238 (4.476)                 | 0.289 (3.937)                 | 0.290 (3.619)                 | 0.298 (3.460)                 | 0.310 (2.857)                 |