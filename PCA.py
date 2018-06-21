import numpy as np 

def load_dataset(filename):
    fr = open(filename)
    string_arr = [line.strip().split() for line in fr.readlines()]
    dat_arr = [list(map(float,line)) for line in string_arr]
    return np.mat(dat_arr)

def pca(datamat, n):
    mean_vals = np.mean(datamat, axis=0)
    mean_removed = datamat - mean_vals
    cov_mat = np.mat(np.cov(mean_removed, rowvar=0))
    eig_vals, eig_vects = np.linalg.eig(cov_mat)
    eig_val_ind = np.argsort(-eig_vals)
    eig_val_ind = eig_val_ind[:n]
    red_eig_vects = eig_vects[:,eig_val_ind]
    new_datamat = mean_removed * red_eig_vects
    recon_mat = new_datamat * red_eig_vects.T + mean_vals
    return new_datamat, recon_mat

def replace_nan_with_mean(datamat):
    n = np.shape(datamat)[1]
    for i in range(n):
        mean_val = np.mean(datamat[np.nonzero(~np.isnan(datamat[:,i].A))[0],i])
        datamat[np.nonzero(np.isnan(datamat[:,i].A))[0],i] = mean_val
    return datamat