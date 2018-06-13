import numpy as np 
import random


def load_dataset(filename):        
    X = []
    y = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        X.append([float(line_arr[0]),float(line_arr[1])])
        y.append(float(line_arr[2]))
    return X, y

def load_img(dirname):
    from os import listdir
    file_list = listdir(dirname) 
    #print(file_list)
    m = len(file_list)
    data_mat = np.zeros((m,1024))
    labels = []
    for i in range(m):
        file_name = file_list[i]
        file_str = file_name.split('.')[0]
        class_num = int(file_str.split('_')[0])
        if class_num == 9:
            labels.append(-1)
        else:
            labels.append(1)
        data_mat[i,:] = img2vector(dirname+'/'+file_list[i])
    return data_mat, labels

def img2vector(filename):
    return_vec = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vec[0,32*i+j] = int(line_str[j])
    return return_vec

class svm_mlia():

    class opt_struct():
        def __init__(self, X, y, C, tol, ktup, kernel):
            self.X = np.mat(X)
            self.y = np.mat(y).T
            self.C = C
            self.tol = tol
            self.m = np.shape(X)[0]
            self.n = np.shape(X)[1]
            self.alphas = np.mat(np.zeros((self.m,1)))
            self.b = 0
            self.ecache = np.mat(np.zeros((self.m,2)))
            self.ktup = ktup
            self.K = np.mat(np.zeros((self.m,self.m)))
            for i in range(self.m):
                self.K[:,i] = kernel(self.X, self.X[i,:], ktup)

    def kernel(self, X, A, ktup):
        '''
        output: mat m*1
        '''
        m,n = np.shape(X)
        K = np.mat(np.zeros((m,1)))
        if ktup[0] == 'lin':
            K = X * A.T
        elif ktup[0] == 'rbf':
            for j in range(m):
                delta = X[j,:] - A
                K[j] = delta * delta.T 
            K = np.exp(K/(-2*ktup[1]**2))
        else:
            raise NameError('can not recognize')
        return K

    def select_jrand(self, i, m):
        j=i
        while (j==i):
            j = int(random.uniform(0,m))
        return j

    def select_j(self, i, os, Ei):
        max_k = -1
        max_deltaE = 0
        os.ecache[i] = [1,Ei]
        valid_list = np.nonzero(os.ecache[:,0])[0]
        j = -1
        Ej = 0
        if len(valid_list) > 1:
            for k in valid_list:
                if k == i:
                    continue
                Ek = self.calc_Ek(os, k)
                deltaE = np.abs(Ei - Ek)
                if deltaE > max_deltaE:
                    max_k = k
                    max_deltaE = deltaE
                    Ej = Ek
            return max_k, Ej
        else:
            j = self.select_jrand(i, os.m)
            Ej = self.calc_Ek(os, j)
        return j, Ej

    def updateEk(self, os, k):
        Ek = self.calc_Ek(os, k)
        os.ecache[k] = [1,Ek]


    def clip_alpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L 
        return aj


    def calc_Ek(self, os, i):
        #alphas = mat(m,1)
        #label_mat = mat(m,1)
        #data_mat = mat(m,n)
        #x = mat(1,n)
        #b = 1
        fx = float(np.multiply(os.alphas,os.y).T 
            * os.K[:,i]) + os.b
        return  fx - float(os.y[i])
    



    def bond(self, os, i, j):
        '''
        alphas = mat(m,1)
        label_mat = mat(m,1)
        '''

        if os.y[i] != os.y[j]:
            L = max(0.0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0.0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        return L, H

    def eta(self, os, i, j):
        '''
        data_mat = mat(m,n)
        '''
        eta = -2.0*os.K[i, j] + os.K[i, i] + os.K[j, j]
        return eta


    def b_new(self, os, i, j, Ei, Ej, alpha_i_old, alpha_j_old):

        diff_alpha_i = os.alphas[i] - alpha_i_old
        diff_alpha_j = os.alphas[j] - alpha_j_old
        '''
        K11 = os.X[i] * os.X[i].T
        K12 = os.X[i] * os.X[j].T
        K22 = os.X[j] * os.X[j].T
        '''
        b1 = - Ei - os.y[i]*diff_alpha_i*os.K[i, i] \
            - os.y[j]*diff_alpha_j*os.K[i, j] + os.b
        b2 = - Ej - os.y[i]*diff_alpha_i*os.K[i, j] \
            - os.y[j]*diff_alpha_j*os.K[j, j] + os.b
        return b1, b2


    def smo_simple(self, X, y, C, tol, max_iter):
        os = self.opt_struct(X, y, C, tol, ktup, self.kernel)
        iter = 0
        while (iter < max_iter):
            print(('*'*14+'iteration %d start'+'*'*14) % iter)
            alpha_pairs_changed = 0
            for i in range(os.m):
                print(('*'*7+'the %d sample'+'*'*7) % i)
                Ei = self.calc_Ek(os, i)
                if ((labels[i]*Ei < -tol) and (os.alphas[i] < C)) or \
                    ((labels[i]*Ei > tol) and (os.alphas[i] > 0)):
                    j = self.select_jrand(i,os.m)
                    Ej = self.calc_Ek(os, j)
                    alpha_i_old = os.alphas[i].copy()
                    alpha_j_old = os.alphas[j].copy()
                    
                    L, H = self.bond(os, i, j)
                    if L== H:
                        print('L==H')
                        continue

                    eta = self.eta(os, i, j)
                    if eta <= 0:
                        print('eta<=0')
                        continue

                    os.alphas[j] += os.y[j]*(Ei - Ej)/eta
                    os.alphas[j] = self.clip_alpha(os.alphas[j], H, L)
                    if (abs(os.alphas[j]-alpha_j_old)<0.00001):
                        print('j not moving enough')
                        continue

                    os.alphas[i] += os.y[i]*os.y[j]* \
                        (alpha_j_old - os.alphas[j])
                    
                    b1, b2 = self.b_new(os, i, j, Ei, Ej, \
                                        alpha_i_old, alpha_j_old)
 
                    if (os.alphas[i] > 0) and (os.alphas[i] < C):
                        os.b = b1
                    elif (os.alphas[j] > 0) and (os.alphas[j] < C):
                        os.b = b2
                    else: os.b = (b1+b2)/2.0

                    alpha_pairs_changed += 1
                    print('iter: %d i:%d j:%d, pairs changed %d' % \
                        (iter, i, j, alpha_pairs_changed)) 

            if alpha_pairs_changed == 0:
                iter += 1
            else:
                iter = 0
            print('iteration number: %d' % iter)
        return os

    def inner_loop(self, i, os):
        print(('*'*7+'the %d sample'+'*'*7) % i)
        Ei = self.calc_Ek(os, i)
        if ((os.y[i]*Ei < -os.tol) and (os.alphas[i] < os.C)) or \
            ((os.y[i]*Ei > os.tol) and (os.alphas[i] > 0)):
            j, Ej = self.select_j(i, os, Ei)
            alpha_i_old = os.alphas[i].copy()
            alpha_j_old = os.alphas[j].copy()
            
            L, H = self.bond(os, i, j)
            if L== H:
                print('L==H')
                return 0

            eta = self.eta(os, i, j)
            if eta <= 0:
                print('eta<=0')
                return 0

            os.alphas[j] += os.y[j]*(Ei - Ej)/eta
            os.alphas[j] = self.clip_alpha(os.alphas[j], H, L)
            self.updateEk(os,j)

            if (abs(os.alphas[j]-alpha_j_old)<0.00001):
                print('j not moving enough')
                return 0

            os.alphas[i] += os.y[i]*os.y[j]* \
                (alpha_j_old - os.alphas[j])
            self.updateEk(os,i)   
            
            b1, b2 = self.b_new(os, i, j, Ei, Ej, \
                                alpha_i_old, alpha_j_old)

            if (os.alphas[i] > 0) and (os.alphas[i] < os.C):
                os.b = b1
            elif (os.alphas[j] > 0) and (os.alphas[j] < os.C):
                os.b = b2
            else: os.b = (b1+b2)/2.0
            return 1
        else:
            return 0

    def smoP(self, X, y, C, tol, max_iter, ktup):
        self.os = self.opt_struct(X, y, C, tol, ktup, self.kernel)
        iter = 0
        entire_set = True
        alpha_pairs_changed = 0
        while (iter < max_iter) and \
            ((alpha_pairs_changed > 0) or entire_set):
            alpha_pairs_changed = 0
            if entire_set:
                for i in range(self.os.m):
                    alpha_pairs_changed += self.inner_loop(i, self.os)
                    print('fullset, iter:%d i:%d, paris changed %d' % \
                        (iter, i, alpha_pairs_changed))
            else:
                non_bound = np.nonzero((self.os.alphas.A>0) * (self.os.alphas.A<C))[0]
                for i in non_bound:
                    alpha_pairs_changed += self.inner_loop(i, self.os)
                    print('non-bound, iter:%d i:%d, paris changed %d' % \
                        (iter, i, alpha_pairs_changed))
            iter+=1         
            if entire_set: 
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True
            print('iteration number: %d' % iter)

        #return os
 
    def pred(self, data, labels):
        sv_index = np.nonzero(self.os.alphas.A>0)[0]
        sv_x = np.mat(self.os.X)[sv_index]
        sv_y = np.mat(self.os.y)[sv_index]
        sv_alphas = np.mat(self.os.alphas)[sv_index]
        x_valid = np.mat(data)
        if len(labels) > 0:
            y_valid = np.mat(labels).T
        error_cnt = 0
        result_list = []
        '''
        if os.ktup[0] == 'lin':
        for i in sv_index:
            w += np.multiply(os.alphas[i]*os.y[i],os.X[i].T)
        return np.mat(data_in)*w+os.b
        '''
        for i in range(len(x_valid)):
            result = np.sign(np.multiply(sv_alphas, sv_y).T * \
                    self.kernel(sv_x,x_valid[i],self.os.ktup) + self.os.b)
            if (len(labels) > 0) and (result != y_valid[i]):
                error_cnt+=1
            result_list.append(result)
        if len(labels) > 0:
            print('error rate: %.2f' % float(error_cnt/len(y_valid)))
        return result_list
