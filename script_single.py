# -*- coding: utf-8 -*-

from full_kmeans import run_kmeans,run_kmeans_without
from gaussian_kmeans import run_set,run_set_without
from multiprocessing import Pool, current_process
from tslearn.datasets import UCR_UEA_datasets 

import csv
import shutil
import os 
import logging
from logging.handlers import RotatingFileHandler

LOG_FILE = "CinCECGTorso_dtw.log"

def init_logger(log_file):
    logger = logging.getLogger(current_process().name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = RotatingFileHandler(log_file, maxBytes=20*1024*1024, backupCount=1000)
        formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def worker_init(log_file):
    global logger
    logger = init_logger(log_file)

def workload(preprocess, metric, data, runs=10, outputFolder='lastResults', basePath ='lastAnalytics' ):
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


def main(preprocess, metric, data,  runs = 10, outputFolder=''):
    basePath = 'lastAnalytics'
    # if not os.path.exists(outputFolder):
    #     os.makedirs(outputFolder)
    # if not os.path.exists(basePath):
    #     os.makedirs(basePath)
    # datasets = UCR_UEA_datasets().list_univariate_datasets()
    # datasets = ['ArrowHead']
    # choose = [0,1]
    #ChlorineConcentration_dtw.csv
    file = f"{outputFolder}/{data}_{metric}"
    # if preprocess==1 (==True) : 
    if preprocess == 0:
        LOG_FILE = f"{file}_preprocessed.csv"
    else :
        LOG_FILE = f"{file}.csv"
    
    try:
        if not os.path.exists('logs'):
            os.makedirs('logs')
        logfile = f"logs/{LOG_FILE}"
        if os.path.exists(logfile):
            os.remove(logfile)
        worker_init(f"logs/{LOG_FILE}")
        workload(preprocess,metric, data)    
    except Exception as e:
        logger.error(e)
    
if __name__ == '__main__':
    main(1, "euclidean", "CricketX") 
