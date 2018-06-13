import numpy as np 
import random


def load_dataset(filename):
    #output:list        
    X = []
    y = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        X.append([float(line_arr[0]),float(line_arr[1])])
        y.append(float(line_arr[2]))
    return X, y


class opt_struct:
    def __init__(self, X, y, C, tol, ktup):
        self.X = np.mat(X)
        self.y = np.mat(y).T
        self.C = C
        self.tol = tol
        self.m = np.shape(X)[0]
        self.n = np.shape(X)[1]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.ecache = np.mat(np.zeros((self.m,2)))
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = self.kernel(self.X, self.X[i,:], ktup)

    def kernel(self, X, A, ktup):
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



class svm():
    def __init__(self, C, tol, ktup):
        
        self.C = C
        self.tol = tol
        self.b = 0
        self.ktup = ktup


    def kernel(self, X, A, ktup):
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

    def select_jrand(self, i):
        j=i
        while (j==i):
            j = int(random.uniform(0,self.m))
        return j

    def select_j(self, i, Ei):
        max_k = -1
        max_deltaE = 0
        self.ecache[i] = [1,Ei]
        valid_list = np.nonzero(self.ecache[:,0])[0]
        j = -1
        Ej = 0
        if len(valid_list) > 1:
            for k in valid_list:
                if k == i:
                    continue
                Ek = self.calc_Ek(k)
                deltaE = np.abs(Ei - Ek)
                if deltaE > max_deltaE:
                    max_k = k
                    max_deltaE = deltaE
                    Ej = Ek
            return max_k, Ej
        else:
            j = self.select_jrand(i, self.m)
            Ej = self.calc_Ek(j)
        return j, Ej

    def updateEk(self, k):
        Ek = self.calc_Ek(k)
        self.ecache[k] = [1,Ek]


    def clip_alpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L 
        return aj


    def calc_Ek(self, i):
        #alphas = mat(m,1)
        #label_mat = mat(m,1)
        #data_mat = mat(m,n)
        #x = mat(1,n)
        #b = 1
        fx = float(np.multiply(self.alphas,self.y).T 
            * self.K[:,i]) + self.b
        return  fx - float(self.y[i])
    



    def bond(self, i, j):
        '''
        alphas = mat(m,1)
        label_mat = mat(m,1)
        '''

        if self.y[i] != self.y[j]:
            L = max(0.0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        else:
            L = max(0.0, self.alphas[j] + self.alphas[i] - self.C)
            H = min(self.C, self.alphas[j] + self.alphas[i])
        return L, H

    def eta(self, i, j):
        '''
        data_mat = mat(m,n)
        '''
        '''
        eta = -2.0 * os.data_mat[i]*os.data_mat[j].T \
            + os.data_mat[i]*os.data_mat[i].T \
            + os.data_mat[j]*os.data_mat[j].T
        '''
        eta = -2.0*self.K[i, j] + self.K[i, i] + self.K[j, j]
        return eta


    def b_new(self, i, j, Ei, Ej, alpha_i_old, alpha_j_old):

        diff_alpha_i = self.alphas[i] - alpha_i_old
        diff_alpha_j = self.alphas[j] - alpha_j_old
        '''
        K11 = self.data_mat[i] * self.data_mat[i].T
        K12 = self.data_mat[i] * self.data_mat[j].T
        K22 = self.data_mat[j] * self.data_mat[j].T
        '''
        b1 = - Ei - self.y[i]*diff_alpha_i*self.K[i, i] \
            - self.y[j]*diff_alpha_j*self.K[i, j] + self.b
        b2 = - Ej - self.y[i]*diff_alpha_i*self.K[i, j] \
            - self.y[j]*diff_alpha_j*self.K[j, j] + self.b
        return b1, b2


    def smo_simple(self, X, y):
        '''
        input: X list m*n
               y list 1*m
        output b float
               alphas mat m*1
        '''
        os = opt_struct(X, y, self.C, self.tol, self.ktup)
        iter = 0
        while (iter < self.max_iter):
            print(('*'*14+'iteration %d start'+'*'*14) % iter)
            alpha_pairs_changed = 0
            for i in range(self.m):
                print(('*'*7+'the %d sample'+'*'*7) % i)
                Ei = self.calc_Ek(i)
                if ((self.y[i]*Ei < -self.tol) and 
                    (self.alphas[i] < self.C)) \
                    or ((self.y[i]*Ei > self.tol) and 
                        (self.alphas[i] > 0)):
                    j = self.select_jrand(i)
                    Ej = self.calc_Ek(j)
                    alpha_i_old = self.alphas[i].copy()
                    alpha_j_old = self.alphas[j].copy()
                    
                    L, H = self.bond(i, j)
                    if L== H:
                        print('L==H')
                        continue

                    eta = self.eta(i, j)
                    if eta <= 0:
                        print('eta<=0')
                        continue

                    self.alphas[j] += self.y[j]*(Ei - Ej)/eta
                    self.alphas[j] = self.clip_alpha(self.alphas[j], H, L)
                    if (abs(self.alphas[j]-alpha_j_old)<0.00001):
                        print('j not moving enough')
                        continue

                    self.alphas[i] += self.y[i]*self.y[j]* \
                        (alpha_j_old - self.alphas[j])
                    
                    b1, b2 = self.b_new(i, j, Ei, Ej, \
                                        alpha_i_old, alpha_j_old)
 
                    if (self.alphas[i] > 0) and (self.alphas[i] < C):
                        self.b = b1
                    elif (self.alphas[j] > 0) and (self.alphas[j] < C):
                        self.b = b2
                    else: self.b = (b1+b2)/2.0

                    alpha_pairs_changed += 1
                    print('iter: %d i:%d j:%d, pairs changed %d' % \
                        (iter, i, j, alpha_pairs_changed)) 

            if alpha_pairs_changed == 0:
                iter += 1
            else:
                iter = 0
            print('iteration number: %d' % iter)
        return self.b, self.alphas

    def inner_loop(self, i):
        print(('*'*7+'the %d sample'+'*'*7) % i)
        Ei = self.calc_Ek(i)
        if ((self.y[i]*Ei < -self.tol) and (self.alphas[i] < self.C)) or \
            ((self.y[i]*Ei > self.tol) and (self.alphas[i] > 0)):
            j, Ej = self.select_j(i, Ei)
            alpha_i_old = self.alphas[i].copy()
            alpha_j_old = self.alphas[j].copy()
            
            L, H = self.bond(os, i, j)
            if L== H:
                print('L==H')
                return 0

            eta = self.eta(os, i, j)
            if eta <= 0:
                print('eta<=0')
                return 0

            self.alphas[j] += self.y[j]*(Ei - Ej)/eta
            self.alphas[j] = self.clip_alpha(self.alphas[j], H, L)
            self.updateEk(j)

            if (abs(self.alphas[j]-alpha_j_old)<0.00001):
                print('j not moving enough')
                return 0

            self.alphas[i] += self.y[i]*self.y[j]* \
                (alpha_j_old - self.alphas[j])
            self.updateEk(i)   
            
            b1, b2 = self.b_new(i, j, Ei, Ej, alpha_i_old, alpha_j_old)

            if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
                self.b = b1
            elif (self.alphas[j] > 0) and (self.alphas[j] < self.C):
                self.b = b2
            else: self.b = (b1+b2)/2.0
            return 1
        else:
            return 0

    def smoP(self):
        os = opt_struct(data, labels, C, tol, ktup)
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

        return os.b, os.alphas
        '''
        w = np.zeros((os.n,1))
        if ktup[0] = 'lin':
            for i in range(os.m):
                w += np.multiply(os.alphas[i]*os.label_mat[i],os.data_mat[i].T)
            return os.b, os.alphas, np.mat(w)
        elif ktup[0] = 'rbf':
            return os.b, os.alphas
        else:
            raise NameError('can not recognize')
        '''
        

fx = float(np.multiply(os.alphas,os.label_mat).T 
            * os.K[:,i]) + os.b


    def pred(self, alphas, b, data_in, ktup):
        

        if ktup[0] = 'lin':
            return np.mat(data_in)*w+b
        elif ktup[0] = 'rbf':
            return os.b, os.alphas
        else:
            raise NameError('can not recognize')
        
    '''
    def kernel(self, X, A, ktup):
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
    '''
