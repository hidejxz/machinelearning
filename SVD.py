import numpy as np 
from numpy import linalg as la 


def datasimple():
    return[[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]]


def datasimple2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]



def eclud_sim(a,b):
    return 1.0 / (1.0+la.norm(a-b))

def pears_sim(a,b):
    if len(a)<3:
        return 1.0
    return 0.5 + 0.5*np.corrcoef(a,b,rowvar=0)[0][1]

def cos_sim(a,b):
    num = float(a.T*b)
    denom = la.norm(a)*la.norm(b)
    return 0.5 + 0.5*(num/denom)

def stand_est(datamat, user, sim_meas, item):
    n = np.shape(datamat)[1]
    sim_total = 0.0
    rat_sim_total = 0.0
    for j in range(n):
        user_rating = datamat[user,j]
        if user_rating==0 or j==item:
            continue
        overlap = np.nonzero(np.logical_and(
            datamat[:,item].A>0, datamat[:,j].A>0))[0]
        if len(overlap) == 0:
            similarity = 0
        else: similarity = sim_meas(datamat[overlap, item],datamat[overlap, j])
        sim_total += similarity
        rat_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total/sim_total

def svd_est(datamat, user, sim_meas, item):
    n = np.shape(datamat)[1]
    sim_total = 0.0
    rat_sim_total = 0.0
    U, Sigma,VT = la.svd(datamat)
    Sig = np.mat(np.eye(4)*Sigma[:4])
    xformed_Items = datamat.T * U[:,:4] * Sig.I
    for j in range(n):
        user_rating = datamat[user,j]
        if user_rating==0 or j==item:
            continue
        similarity = sim_meas(datamat[item,:].T,datamat[j,:].T)
        sim_total += similarity
        rat_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total/sim_total



def recommend(datamat, user, N=3, sim_meas=cos_sim, est_method=stand_est):
    unrated_items_ind = np.nonzero(datamat[user,:].A==0)[1]
    if len(unrated_items_ind) == 0:
        return 'you rated everthing'
    itemscores = []
    for item in unrated_items_ind:
        est_scores = est_method(datamat, user, sim_meas, item)
        itemscores.append((item,est_scores))
    return sorted(itemscores, key = lambda x:x[1], reverse = True)[:N]


def print_mat(datamat, thresh=0.8):
    m,n = np.shape(datamat)
    print('shape: %d * %d' % (m,n))
    for i in range(32):
        a = []
        for k in range(32):
            if float(datamat[i,k]) > thresh:
                print(1, end='')
            else:
                print(0, end='')
        print('')


def img_compress(num_sv=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        new_row = []
        for i in range(32):
            new_row.append(int(line[i]))
        myl.append(new_row)
    datamat = np.mat(myl)
    print('*'*5,'original matrix', '*'*5)
    print_mat(datamat, thresh=0.8)
    U, Sigma, VT = la.svd(datamat)
    sig_recon = np.mat(np.zeros((num_sv, num_sv)))
    for k in range(num_sv):
        sig_recon[k,k] = Sigma[k]
    recon_mat = U[:,:num_sv]*sig_recon*VT[:num_sv,:]
    print('*'*5,'reconstructed matrix using %d singular values' % num_sv, '*'*5)
    print_mat(recon_mat, thresh)




