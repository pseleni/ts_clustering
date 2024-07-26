# ts_clustering

This repository contains the code for our paper:

**Efficient Time-Series Clustering through Sparse Gaussian Modeling** <br />
*Dimitris Fotakis, Panagiotis Patsilinakos, Eleni Psaroudaki, Michalis Xefteris*<br />
Paper: https://www.mdpi.com/1999-4893/17/2/61


```
@article{fotakis2024efficient,
  title={Efficient Time-Series Clustering through Sparse Gaussian Modeling},
  author={Fotakis, Dimitris and Patsilinakos, Panagiotis and Psaroudaki, Eleni and Xefteris, Michalis},
  journal={Algorithms},
  volume={17},
  number={2},
  pages={61},
  year={2024},
  publisher={MDPI}
}

```
## Structure

    .
    ├── images                  # All images
    │   ├── dtw                 # Images for dtw metric
    │   |    ├── ami            # AMI images
    │   |    ├── ari            # ARI images
    │   ├── euclidean           # Images for euclidean metric
    │   |    ├── ami            # AMI images
    │   |    ├── ari            # ARI images
    │   ├── runtime             # Runtime images
    │   └── additional          # Additional images
    ├── src                     # Contains the code and the results
    │   ├── analytics           # Contains the results for each a-m combination
    │   ├── results             # Contains the avg and std over a for each m and run-times
    │   └── ...                 # source code
    ├── tables                  # Tables in tex and md form 
    │   ├── dtw                 # Tables for dtw metric
    │   |    ├── ami            # AMI tables
    │   |    ├── ari            # ARI tables
    │   ├── euclidean           # Tables for euclidean metric
    │   |    ├── ami            # AMI tables
    │   |    ├── ari            # ARI tables
    │   └── runtime             # Runtime tables
    └── ...                     # README, yml


#### Find which dataset's results are not available 
For our convenience, we have created a bash script that checks the files in the results folder and prints all the combinations of (dataset, metric, with/without preprocessing) that are missing. To run and save the results to a file simply:
```
bash ./findMissing.sh > missing.txt
```
or in Linux environment run 
```
chmod u+x findMissing.sh  
```
and then simply
```
./findMissing.sh > missing.txt
```
