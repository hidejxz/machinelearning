import numpy as np 
import random


class opt_struct:
    def __init__(self, data, labels, C, tol):
        self.data_mat = np.mat(data)
        self.label_mat = np.mat(labels).T
        self.C = C
        self.tol = tol
        self.m = np.shape(data)[0]
        self.n = np.shape(data)[1]
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
        fx = float(np.multiply(os.alphas,os.label_mat).T 
            * (os.data_mat*os.data_mat[i].T)) + os.b
        return  fx - float(os.label_mat[i])
    



    def bond(self, os, i, j):
        '''
        alphas = mat(m,1)
        label_mat = mat(m,1)
        '''

        if os.label_mat[i] != os.label_mat[j]:
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
        eta = -2.0 * os.data_mat[i]*os.data_mat[j].T \
            + os.data_mat[i]*os.data_mat[i].T \
            + os.data_mat[j]*os.data_mat[j].T
        return eta


    def b_new(self, os, i, j, Ei, Ej, alpha_i_old, alpha_j_old):

        diff_alpha_i = os.alphas[i] - alpha_i_old
        diff_alpha_j = os.alphas[j] - alpha_j_old

        K11 = os.data_mat[i] * os.data_mat[i].T
        K12 = os.data_mat[i] * os.data_mat[j].T
        K22 = os.data_mat[j] * os.data_mat[j].T

        b1 = - Ei - os.label_mat[i]*diff_alpha_i*K11 \
            - os.label_mat[j]*diff_alpha_j*K12 + os.b
        b2 = - Ej - os.label_mat[i]*diff_alpha_i*K12 \
            - os.label_mat[j]*diff_alpha_j*K22 + os.b
        return b1, b2


    def smo_simple(self, data, labels, C, tol, max_iter):
        os = opt_struct(data, labels, C, tol)
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

                    os.alphas[j] += os.label_mat[j]*(Ei - Ej)/eta
                    os.alphas[j] = self.clip_alpha(os.alphas[j], H, L)
                    if (abs(os.alphas[j]-alpha_j_old)<0.00001):
                        print('j not moving enough')
                        continue

                    os.alphas[i] += os.label_mat[i]*os.label_mat[j]* \
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
        return os.b, os.alphas

    def inner_loop(self, i, os):
        print(('*'*7+'the %d sample'+'*'*7) % i)
        Ei = self.calc_Ek(os, i)
        if ((os.label_mat[i]*Ei < -os.tol) and (os.alphas[i] < os.C)) or \
            ((os.label_mat[i]*Ei > os.tol) and (os.alphas[i] > 0)):
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

            os.alphas[j] += os.label_mat[j]*(Ei - Ej)/eta
            os.alphas[j] = self.clip_alpha(os.alphas[j], H, L)
            self.updateEk(os,j)

            if (abs(os.alphas[j]-alpha_j_old)<0.00001):
                print('j not moving enough')
                return 0

            os.alphas[i] += os.label_mat[i]*os.label_mat[j]* \
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

    def smoP(self, data, labels, C, tol, max_iter):
        os = opt_struct(data, labels, C, tol)
        iter = 0
        entire_set = True
        alpha_pairs_changed = 0
        while (iter < max_iter) and \
            ((alpha_pairs_changed > 0) or entire_set):
            alpha_pairs_changed = 0
            if entire_set:
                for i in range(os.m):
                    alpha_pairs_changed += self.inner_loop(i, os)
                    print('fullset, iter:%d i:%d, paris changed %d' % \
                        (iter, i, alpha_pairs_changed))
            else:
                non_bound = np.nonzero((os.alphas.A>0) * (os.alphas.A<C))[0]
                for i in non_bound:
                    alpha_pairs_changed += self.inner_loop(i, os)
                    print('non-bound, iter:%d i:%d, paris changed %d' % \
                        (iter, i, alpha_pairs_changed))
            iter+=1         
            if entire_set: 
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True
            print('iteration number: %d' % iter)
        w = np.zeros((os.n,1))
        for i in range(os.m):
            w += np.multiply(os.alphas[i]*os.label_mat[i],os.data_mat[i].T)
        return os.b, os.alphas, np.mat(w)

    def pred(self, w, b, data_in):
        return np.mat(data_in)*w+b

