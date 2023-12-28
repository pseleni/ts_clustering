# -*- coding: utf-8 -*-

import time
import numpy as np

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.utils import shuffle

def run_kmeans(dataset, runs, metric = 'dtw'):
    print("{:s}:".format(dataset))

    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
    data_tmp = np.r_[X_train, X_test]
    true_labels = np.r_[y_train, y_test]
    data_tmp, true_labels = shuffle(data_tmp, true_labels, random_state=0)
    N = np.unique(true_labels).size
    data = TimeSeriesScalerMinMax(value_range=(-1., 1.)).fit_transform(data_tmp)
    
    adj_rand_index = []
    adjusted_mutual_info = []

    start = time.process_time()
    for i in range(runs):
        dba_km = TimeSeriesKMeans(n_clusters=N,
                                  n_init=1,
                                  metric=metric,                          
                                  max_iter_barycenter=100, init="random")
        y_pred = dba_km.fit_predict(data)
        
        adj_rand_index.append(adjusted_rand_score(true_labels, y_pred))
        adjusted_mutual_info.append(adjusted_mutual_info_score(true_labels, y_pred))
        
    adj_rand_index =  np.array(adj_rand_index)
    adj_rand_index_mean = np.mean(adj_rand_index)
    adj_rand_index_std = np.std(adj_rand_index, ddof = 1)
    
    adjusted_mutual_info = np.array(adjusted_mutual_info)
    adjusted_mutual_info_mean = np.mean(adjusted_mutual_info)
    adjusted_mutual_info_std = np.std(adjusted_mutual_info, ddof = 1)
    
    end = time.process_time()
    run_time = end-start
        
    print("Adjusted Rand Index - mean: {:f}, std: {:f}".format(adj_rand_index_mean, adj_rand_index_std))
    print("Adjusted Mutual Info - mean: {:f}, std: {:f}".format(adjusted_mutual_info_mean, adjusted_mutual_info_std))        
    print("Running time: {:f} sec".format(run_time))
    return (data.shape[0], round(adj_rand_index_mean,3), 
            round(adj_rand_index_std,3), 
            round(adjusted_mutual_info_mean,3), 
            round(adjusted_mutual_info_std,3), 
            round(run_time,4))


# datasets = ['Beef', 'Coffee', 'ECG200', 'OliveOil', 'Car', 'ArrowHead', 'Meat', 'Plane', 'ProximalPhalanxTW','CBF','Fish', 'BeetleFly', 'BirdChicken', 'DiatomSizeReduction', 'DistalPhalanxTW', 'OSULeaf', 'ShapeletSim', 'Trace', 'Lightning2', 'Lightning7', 'ECGFiveDays', 'FaceFour', 'LargeKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', ]
# metrics = ['euclidean','dtw', 'softdtw']
# header = ['points', 'adj_rand_mean', 'adj_rand_std', 'adj_mut_mean', 'adj_mut_std', 'run_time']

# import csv

# for data in datasets:
#     for metric in metrics:
#         with open(data+'_'+metric+'.csv', 'a') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(header)
#             writer.writerow(run_kmeans(data, 4, metric))
#             csvfile.close()
    