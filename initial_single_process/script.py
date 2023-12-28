# -*- coding: utf-8 -*-

from full_kmeans import run_kmeans
from gaussian_kmeans import run_set


datasets = ['ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF',
 'Car', 'Coffee', 'DiatomSizeReduction', 'DistalPhalanxTW', 'ECG200',
 'ECGFiveDays', 'FaceFour', 'Fish', 'LargeKitchenAppliances',
 'Lightning2', 'Lightning7', 'Meat', 'OSULeaf', 'OliveOil', 'Plane', 
 'ProximalPhalanxTW', 'ShapeletSim', 'SonyAIBORobotSurface1',
 'SonyAIBORobotSurface2', 'Trace']
datasets = ['ArrowHead','BeetleFly', 'BirdChicken', 'CBF']

#datasets = ['Coffee', 'ECG200', 'OliveOil', 'Car', 'ArrowHead', 'Meat', 'Plane', 'ProximalPhalanxTW','CBF','Fish', 'BeetleFly', 'BirdChicken', 'DiatomSizeReduction', 'DistalPhalanxTW', 'OSULeaf', 'ShapeletSim', 'Trace', 'Lightning2', 'Lightning7', 'ECGFiveDays', 'FaceFour', 'LargeKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', ]
metrics = ['euclidean','dtw']#, 'softdtw']
header = ['points', 'adj_rand_mean', 'adj_rand_std', 'adj_mut_mean', 'adj_mut_std', 'run_time']

import csv

for data in datasets:
    for metric in metrics:
        with open(data+'_'+metric+'.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)        
            writer.writerow(run_kmeans(data, 2, metric))
            for i in range(1,6):
                writer.writerow(run_set(data, i, 2, metric))
            print('next')
        
    