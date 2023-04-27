# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot  as plt

datasets = ['ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF',
 'Car', 'Coffee', 'DiatomSizeReduction', 'DistalPhalanxTW', 'ECG200',
 'ECGFiveDays', 'FaceFour', 'Fish', 'LargeKitchenAppliances',
 'Lightning2', 'Lightning7', 'Meat', 'OSULeaf', 'OliveOil', 'Plane', 
 'ProximalPhalanxTW', 'ShapeletSim', 'SonyAIBORobotSurface1',
 'SonyAIBORobotSurface2', 'Trace']

datasets = ['Beef']
metrics = ['euclidean','dtw']#, 'softdtw']
header = ['points', 'adj_rand_mean', 'adj_rand_std', 'adj_mut_mean', 'adj_mut_std', 'run_time']
images = [('adj_rand_mean', 'adj_rand_std'), 
          ('adj_mut_mean', 'adj_mut_std')]    
for column in images:
    for metric in metrics: 
        y = ['original', '1log2or','2log2or','3log2or','4log2or','5log2or']
        
        for dataset in datasets:
            filename = dataset+'_'+metric+'.csv'
            df = pd.read_csv(filename)
            mean = df[column[0]].tolist()
            std = df[column[1]].tolist()
                    
            plt.errorbar(y, mean, std, linestyle='None', marker='^', label=dataset)
        plt.title("{:s} - {:s} - Adjusted Rand Index ".format(dataset, metric), fontsize=15)
        plt.ylabel('ARI', fontsize=14)
        plt.xlabel('Weights', fontsize=14)
        plt.show()          