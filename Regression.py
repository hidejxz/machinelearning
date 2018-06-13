from numpy import *


def load_dataset(filename):
    num_feat = len(open(filename).readline().split())
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat-1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat

class Regression():


    def stand_regression(self, X, y):
        X = mat(X)
        y = mat(y).T
        xTx = X.T*X
        if linalg.det(xTx) == 0.0:
            print('Error')
            return
        w = xTx.I * X.T * y
        return w

    def lwlr(self, xi, X, y, k=1):
        xi = mat(xi)
        X = mat(X)
        y = mat(y).T 
        m = shape(X)[0]
        weights = mat(eye((m)))
        y_hat = []
        for i in range(shape(xi)[0]):
            for j in range(m):
                diff = xi[i] - X[j]
                weights[j,j] = exp((diff*diff.T)/(-2.0*k**2))
            xTx = X.T * weights * X
            if linalg.det(xTx) == 0.0:
                print('Error')
                return
            result = xi[i] * xTx.I * X.T * weights * y
            y_hat.append(result[0,0])
        return y_hat



