# Efficient Time-Series Clustering through Sparse Gaussian Modeling
This repository contains the code for ''Efficient Time-Series Clustering through Sparse Gaussian Modeling'' paper. 


## Structure

    .
    ├── analytics               # Contains the results for each a-m combination
    ├── images                  # All images
    │   ├── dtw                 # Images for dtw metric
    │   |    ├── ami            # AMI images
    │   |    ├── ari            # ARI images
    │   ├── euclidean           # Images for euclidean metric
    │   |    ├── ami            # AMI images
    │   |    ├── ari            # ARI images
    │   ├── runtime             # Runtime images
    │   └── additional          # Additional images
    ├── results                 # Contains the avg and std over a for each m and run-times
    ├── tables                  # Tables in tex and md form 
    │   ├── dtw                 # Tables for dtw metric
    │   |    ├── ami            # AMI tables
    │   |    ├── ari            # ARI tables
    │   ├── euclidean           # Tables for euclidean metric
    │   |    ├── ami            # AMI tables
    │   |    ├── ari            # ARI tables
    │   └── runtime             # Runtime tables
    └── ...




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
