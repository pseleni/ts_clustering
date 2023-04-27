# -*- coding: utf-8 -*-

import csv
import gpytorch
import numpy as np
import time
import torch

from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from math import log2,sqrt
from numpy import newaxis
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.utils import shuffle
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import UCR_UEA_datasets

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        global inducing_p
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(MaternKernel(1.5))
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=torch.linspace(torch.min(train_x), torch.max(train_x), inducing_p), likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def run_set(dataset, ind_points, runs, metric = 'dtw'):
    global inducing_p
    
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
    
    data = np.r_[X_train, X_test]
    true_labels = np.r_[y_train, y_test]
    
    # train_y = X_train[0,:,0]  
    # for indx in range(data.shape[0]):
        # train_y = np.vstack([train_y, data[indx,:,0]])
    
    train_y = np.vstack([data[indx,:,0] for indx in range(data.shape[0])])
    
    
    data, train_y, true_labels = shuffle(data, train_y, true_labels, random_state=0)
    
    train_x = torch.from_numpy(np.arange(1,train_y.shape[1]+1)).float()
    train_y = torch.from_numpy(train_y).float()
    
    train_x = (train_x - train_x.mean(0)) / train_x.std(0)
    #Normalization [-1,1]
    for i in range(train_y.shape[0]):
        train_y[i] = train_y[i] - torch.min(train_y[i])
        train_y[i] = 2 * (train_y[i] / torch.max(train_y[i])) - 1  
    
    inducing_p = ind_points*(int(log2(train_y.shape[1]))+1)
    start = time.process_time()
    # initialize likelihood and model
    likelihood = []
    model = []
    for i in range(train_y.shape[0]):
        likelihood.append(gpytorch.likelihoods.GaussianLikelihood())
        model.append(GPRegressionModel(train_x, train_y[i], likelihood[i]))
    
    training_iter = 50
    for i in range(train_y.shape[0]):
        # Find optimal model hyperparameters
        model[i].train()
        likelihood[i].train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
                {'params': list(model[i].parameters())[:4]},
                {'params': list(model[i].parameters())[4], 'lr': (0.1/ind_points)}]                                 
                                         , lr=0.1)    # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood[i], model[i])

        for it in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model[i](train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y[i])
            loss.backward()        
            optimizer.step()
            
    # Get into evaluation (predictive posterior) mode
    for i in range(train_y.shape[0]):
        model[i].eval()
        likelihood[i].eval()
        
    end = time.process_time()
    run_time = end-start
    print("Running time -SGPR: {:f} sec".format(run_time))
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood

    w_s_b = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0] 
    w_s = [b* data.shape[0] / ind_points for b in w_s_b]
    N = np.unique(true_labels).size

    print("{:s}:\n".format(dataset))    
    adj_rand_indices_random = []
    adjusted_mutual_info_random = []
    r_time = []
    for w in w_s:
        adj_rand_index = []
        adjusted_mutual_info = []
        
        start = time.process_time()
        X = model[0](model[0].covar_module.inducing_points).mean.detach().numpy().astype('float64')
        tX = model[0].covar_module.inducing_points.detach().numpy()[:,-1].astype('float64')
        t_X = np.multiply(sqrt(w),tX)
        x_tmp = np.column_stack((X,t_X))
        time_series = x_tmp[newaxis,:,:]
        
        for indx in range(1, data.shape[0]):
            X = model[indx](model[indx].covar_module.inducing_points).mean.detach().numpy().astype('float64')
            tX = model[indx].covar_module.inducing_points.detach().numpy()[:,-1].astype('float64')
            t_X = np.multiply(sqrt(w),tX)
            x_tmp = np.column_stack((X,t_X))
            time_series = np.concatenate([time_series, x_tmp[None,:,:]], axis=0)
        
        for i in range(runs):
            dba_km = TimeSeriesKMeans(n_clusters=N,
                                      n_init=1,
                                      metric=metric,                          
                                      max_iter_barycenter=100, init="random")
            y_pred = dba_km.fit_predict(time_series)
            
            adj_rand_index.append(adjusted_rand_score(true_labels, y_pred)) 
            adjusted_mutual_info.append(adjusted_mutual_info_score(true_labels, y_pred))
            
            
        end = time.process_time()
        run_time = (end-start)/len(w_s)
        r_time.append(run_time)
        
        adj_rand_index = np.array(adj_rand_index)
        adj_rand_index_mean = np.mean(adj_rand_index)
        adj_rand_index_std = np.std(adj_rand_index, ddof = 1)
        adj_rand_indices_random.append(adj_rand_index_mean)
        
        adjusted_mutual_info = np.array(adjusted_mutual_info)
        adjusted_mutual_info_mean = np.mean(adjusted_mutual_info)
        adjusted_mutual_info_std = np.std(adjusted_mutual_info, ddof = 1)
        adjusted_mutual_info_random.append(adjusted_mutual_info_mean)
        
        print("Weight: {:f} (random) ----> Adjusted Rand Index - mean: {:f}, std: {:f}".format(w, adj_rand_index_mean, adj_rand_index_std))
        print("Weight: {:f} (random) ----> Adjusted Mutual Info - mean: {:f}, std: {:f}".format(w, adjusted_mutual_info_mean, adjusted_mutual_info_std))        
        print("Running time - k-means: {:f} sec".format(run_time))
    
    
    m_ari = np.mean(adj_rand_indices_random)
    s_ari = np.std(adj_rand_indices_random)
    
    m_ami = np.mean(adjusted_mutual_info_random)
    s_ami = np.std(adjusted_mutual_info_random)
    
    name = dataset+'_'+ metric+'_adj_rand_ind.csv'
    with open(name, 'a', newline='') as f:
        writer = csv.writer(f)
        wr = adj_rand_indices_random.copy()
        wr.insert(0, inducing_p)
        writer.writerow(wr)
    
    name = dataset+'_'+metric+'_adj_mutual_info.csv'
    with open(name, 'a', newline='') as f:
        writer = csv.writer(f)
        wr = adjusted_mutual_info_random.copy()
        wr.insert(0, inducing_p)
        writer.writerow(wr)# print(r_time)
    
    return inducing_p, round(m_ari,3), round(s_ari,3), \
        round(m_ami,3), round(s_ami,3), round(np.mean(r_time),4)

