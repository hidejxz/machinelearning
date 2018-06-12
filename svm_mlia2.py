import numpy as np 
import random


class opt_struct:
    def __init__(self, data, labels, C, tol):
        self.data = np.mat(data)
        self.labels = np.mat(labels)
        self.C = C
        self.tol = tol
        self.m = np.shape(data)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.ecache = np.mat(np.zeros((self.m,2)))

class svm_mlia():
    
    def load_dataset(self, filename):        
        data = []
        labels = []
        fr = open(filename)
        for line in fr.readlines():
            line_arr = line.strip().split('\t')
            data.append([float(line_arr[0]),float(line_arr[1])])
            labels.append(float(line_arr[2]))
        return data, labels

    def select_jrand(self, i, m):
        j=i
        while (j==i):
            j = int(random.uniform(0,m))
        return j

    def clip_alpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L 
        return aj


    def err(self, alphas, label_mat, data_mat, b, x, y):
        #alphas = mat(m,1)
        #label_mat = mat(m,1)
        #data_mat = mat(m,n)
        #x = mat(1,n)
        #b = 1
        fx = float(np.multiply(alphas,label_mat).T * (data_mat*x.T)) + b
        return  fx - float(y)

    '''   
    def calc_Ek(self, alphas, label_mat, data_mat, b, x, y):
        #alphas = mat(m,1)
        #label_mat = mat(m,1)
        #data_mat = mat(m,n)
        #x = mat(1,n)
        #b = 1
        fx = float(np.multiply(alphas,label_mat).T * (data_mat*x.T)) + b
        return  fx - float(y)
    '''



    def bond(self, alphas, label_mat, C, i, j):
        '''
        alphas = mat(m,1)
        label_mat = mat(m,1)
        '''

        if label_mat[i] != label_mat[j]:
            L = max(0.0, alphas[j] - alphas[i])
            H = min(C, C + alphas[j] - alphas[i])
        else:
            L = max(0.0, alphas[j] + alphas[i] - C)
            H = min(C, alphas[j] + alphas[i])
        return L, H

    def eta(self, data_mat, i, j):
        '''
        data_mat = mat(m,n)
        '''
        eta = -2.0 * data_mat[i]*data_mat[j].T \
            + data_mat[i]*data_mat[i].T \
            + data_mat[j]*data_mat[j].T
        return eta


    def b_new(self, x, y, alphas, b_old, i, j, Ei, Ej, alpha_i_old, alpha_j_old):
        '''
        b1 = b - Ei - label_mat[i]*(alphas[i] - alpha_i_old)*data_mat[i]*data_mat[i].T \
            - label_mat[j]*(alphas[j] - alpha_j_old)*data_mat[i]*data_mat[j].T
        b2 = b - Ej - label_mat[i]*(alphas[i] - alpha_i_old)*data_mat[i]*data_mat[j].T \
            - label_mat[j]*(alphas[j] - alpha_j_old)*data_mat[j]*data_mat[j].T
        '''

        diff_alpha_i = alphas[i] - alpha_i_old
        diff_alpha_j = alphas[j] - alpha_j_old

        K11 = x[i] * x[i].T
        K12 = x[i] * x[j].T
        K22 = x[j] * x[j].T

        b1 = - Ei - y[i]*diff_alpha_i*K11 - y[j]*diff_alpha_j*K12 + b_old
        b2 = - Ej - y[i]*diff_alpha_i*K12 - y[j]*diff_alpha_j*K22 + b_old
        return b1, b2


    def smo_simple(self, data, labels, C, tol, max_iter):
        os = opt_struct(data, labels, C, tol)
        data_mat = np.mat(data)
        label_mat = np.mat(labels).T
        b = 0.0
        m,n = np.shape(data_mat)
        alphas = np.mat(np.zeros((m,1)))
        iter = 0
        while (iter < max_iter):
            print(('*'*14+'iteration %d start'+'*'*14) % iter)
            alpha_pairs_changed = 0
            for i in range(m):
                print(('*'*7+'the %d sample'+'*'*7) % i)
                #fXi = float(np.multiply(alphas,label_mat).T*(data_mat*data_mat[i].T)) + b
                #Ei = fXi - float(label_mat[i])
                Ei = self.err(alphas, label_mat, data_mat, \
                    b, data_mat[i], label_mat[i])
                if ((label_mat[i]*Ei < -tol) and (alphas[i] < C)) or \
                    ((label_mat[i]*Ei > tol) and (alphas[i] > 0)):
                    j = self.select_jrand(i,m)
                    #fXj = float(np.multiply(alphas,label_mat).T*(data_mat*data_mat[j].T)) + b
                    #Ej = fXj - float(label_mat[j])
                    Ej = self.err(alphas, label_mat, data_mat, \
                        b, data_mat[j], label_mat[j])
                    alpha_i_old = alphas[i].copy()
                    alpha_j_old = alphas[j].copy()
                    
                    L, H = self.bond(alphas, label_mat, C, i, j)
                    '''
                    if (label_mat[i] != label_mat[j]):
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])
                    '''
                    if L== H:
                        print('L==H')
                        continue
                    eta = self.eta(data_mat, i, j)
                    #eta = 2.0 * data_mat[i]*data_mat[j].T - data_mat[i]*data_mat[i].T - data_mat[j]*data_mat[j].T
                    if eta <= 0:
                        print('eta<=0')
                        continue
                    alphas[j] += label_mat[j]*(Ei - Ej)/eta
                    alphas[j] = self.clip_alpha(alphas[j], H, L)
                    #print('diff: ',alphas[j]-alpha_j_old)
                    if (abs(alphas[j]-alpha_j_old)<0.00001):
                        print('j not moving enough')
                        continue
                    alphas[i] += label_mat[i]*label_mat[j]*(alpha_j_old - alphas[j])
                    
                    b1, b2 = self.b_new(data_mat, label_mat, alphas, b, i, j, Ei, Ej, alpha_i_old, alpha_j_old)
                    '''

                    b1 = b - Ei- label_mat[i]*(alphas[i]-alpha_i_old)*data_mat[i]*data_mat[i].T - label_mat[j]*(alphas[j]-alpha_j_old)*data_mat[i]*data_mat[j].T
                    b2 = b - Ej- label_mat[i]*(alphas[i]-alpha_i_old)*data_mat[i]*data_mat[j].T - label_mat[j]*(alphas[j]-alpha_j_old)*data_mat[j]*data_mat[j].T
                    '''
                    if (alphas[i] > 0) and (alphas[i] < C):
                        b = b1
                    elif (alphas[j] > 0) and (alphas[j] < C):
                        b = b2
                    else: b = (b1+b2)/2.0

                    alpha_pairs_changed += 1
                    print('iter: %d i:%d j:%d, pairs changed %d' % \
                        (iter, i, j, alpha_pairs_changed)) 

            if alpha_pairs_changed == 0:
                iter += 1
            else:
                iter = 0
            print('iteration number: %d' % iter)
        return b, alphas





