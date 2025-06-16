import os
import sys
import pandas as pd 
import numpy as np
import scipy as sp
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from utils import generate_data, community_detection, LR, LR1, CLSE, permute_columns, run_netcoh, np_cv_rules

path_result = 'result/ex1/'
os.makedirs(path_result, exist_ok=True)

n_simu = 200
sigma = .5
res = []
coef = 'full'

for K in tqdm([3,4,5], desc='K', leave=True):
    for n in tqdm(np.arange(100, 1001, 100), desc='n', leave=False):
        for seed in tqdm(range(n_simu), desc='seed', leave=False):
            Z, B, P, A, beta, x, y = generate_data(n=n, K=K, sigma=sigma, seed=seed, coef=coef)

            Z_hat = community_detection(A, K)[1]
            Z_hat = permute_columns(Z_hat, Z)
            err_Z = np.sum(np.abs(Z - Z_hat))/2

            for method in [CLSE, LR, LR1]:
                hbeta = method(y, x, A, Z_hat)

                err_est = np.mean((hbeta - beta)**2)
                err_pred = np.mean((((Z @ beta @ Z.T - Z_hat @ hbeta @ Z_hat.T) * A) @ x)**2)
                # err_sig = np.mean((((Z @ (hbeta - beta) @ Z.T) * A) @ x @ x.T)**2)
                res.append([method.__name__, n, K, seed, err_est, err_pred, err_Z])

            with np_cv_rules.context():
                y_hat = run_netcoh(y, x, A)['yhat']
            err_pred = np.mean((((Z @ beta @ Z.T) * A) @ x - y_hat)**2)
            res.append(['netcoh', n, K, seed, None, err_pred, err_Z])

df = pd.DataFrame(res, columns=['method', 'n', 'K', 'seed', 'err_est', 'err_pred', 'err_Z'])
df.to_csv(path_result + 'res_coef_{}_sigma_{:.01f}.csv'.format(coef,sigma), index=False)
