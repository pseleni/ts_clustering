# Time Series Clustering
The code for Time Series Clustering using Sparse Gaussian Processes Framework


### Find which dataset's results are not available 
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
