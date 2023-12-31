# -*- coding: utf-8 -*-

from full_kmeans import run_kmeans,run_kmeans_without
from gaussian_kmeans import run_set,run_set_without
from multiprocessing import Pool, current_process
from tslearn.datasets import UCR_UEA_datasets 

import csv
import os 
import logging

LOG_FILE = "my_log.log"

def init_logger(log_file):
    logger = logging.getLogger(current_process().name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="w")
        formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def worker_init(log_file):
    global logger
    logger = init_logger(log_file)

def workload(preprocess, metric, data, runs=10, outputFolder='defaultResults', basePath ='defaultAnalytic' ):
    # log_file = "my_log.log"
    logger = logging.getLogger(current_process().name)

    
    # preprocess == 0 means with preprocess
    # preprocess == 1 means without preprocess
    km = [run_kmeans, run_kmeans_without]
    rs = [run_set, run_set_without]
    
    func1 = km[preprocess]
    func2 = rs[preprocess]
    
    file = f"{outputFolder}/{data}_{metric}"
    # if preprocess==1 (==True) : 
    if preprocess == 0:
        file = f"{file}_preprocessed.csv"
    else :
        file = f"{file}.csv"
    
    logger.info(f"File: {file}") 

    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)   
        logger.info("Created file.")
          

    with open(file, 'a', newline='') as csvfile:   
        writer = csv.writer(csvfile)
        writer.writerow(func1(data, runs, metric))
        logger.info("k-means results written.")

    for i in range(1,6):
        with open(file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(func2(data, i, runs, metric, basePath))
            logger.info(f"Gaussian k-means {i} results written.")
    
    logger.info("Finished")

# datasets = ['ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF',
#  'Car', 'Coffee', 'DiatomSizeReduction', 'DistalPhalanxTW', 'ECG200',
#  'ECGFiveDays', 'FaceFour', 'Fish', 'LargeKitchenAppliances',
#  'Lightning2', 'Lightning7', 'Meat', 'OSULeaf', 'OliveOil', 'Plane', 
#  'ProximalPhalanxTW', 'ShapeletSim', 'SonyAIBORobotSurface1',
#  'SonyAIBORobotSurface2', 'Trace']
#datasets = UCR_UEA_datasets().list_multivariate_datasets()
#datasets = ['AtrialFibrillation']


#datasets = ['Coffee', 'ECG200', 'OliveOil', 'Car', 'ArrowHead', 'Meat', 'Plane', 'ProximalPhalanxTW','CBF','Fish', 'BeetleFly', 'BirdChicken', 'DiatomSizeReduction', 'DistalPhalanxTW', 'OSULeaf', 'ShapeletSim', 'Trace', 'Lightning2', 'Lightning7', 'ECGFiveDays', 'FaceFour', 'LargeKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', ]
metrics = ['euclidean','dtw']#, 'softdtw']
header = ['points', 'adj_rand_mean', 'adj_rand_std', 'adj_mut_mean', 'adj_mut_std', 'run_time', 'run_time_base']


def main(processes = 4, runs = 10, outputFolder='defaultResults'):
    basePath = 'defaultAnalytic'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if not os.path.exists(basePath):
        os.makedirs(basePath)
    datasets = UCR_UEA_datasets().list_univariate_datasets()
    # datasets = ['ArrowHead']
    choose = [0,1]
    pool = Pool(processes)
    mappings = []
    fname = "missing.txt"

    if (not fname):
        for data in datasets:
            for metric in metrics:
                for preprocess in choose: 
                    mappings.append((preprocess, metric, data, ))
    else :
        # Open the file for reading
        with open(fname, "r") as file:
            # Read each line in the file
            for line in file.readlines():
                # Split each line into three parts using a comma as the delimiter
                parts = line.strip().split(',')
                
                # Check if there are exactly three parts in the line
                if len(parts) == 3:
                    # Convert the first part to an integer (assuming it's a number)
                    number = int(parts[0])
                    string1 = parts[1].strip()
                    string2 = parts[2].strip()
                    
                    # Create a tuple and append it to the list
                    mappings.append((number, string1, string2))
    try:
        with Pool(processes, initializer=worker_init, initargs=(LOG_FILE,)) as pool:
            pool.starmap(workload, mappings)    
    except KeyboardInterrupt:
        logger.error('Caught KeyboardInterrupt, terminating workers')
        pool.terminate()
        pool.join()
    else:
        logger.info("Completed all tasks")

if __name__ == '__main__':
    main(18, 10)    
