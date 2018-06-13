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

def preprocessing(X, y):
    y_mean = mean(y,axis = 0)
    y_arr = y - y_mean
    x_mean = mean(X,axis = 0)
    x_var = var(X,axis = 0)
    x_arr = (X-x_mean)/x_var
    return x_arr, y_arr

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

    def ridge_regression(self, X, y, lam = 0.2):
        X = mat(X)
        y = mat(y).T
        denom = X.T*X + eye(shape(X)[1])*lam
        if linalg.det(denom) == 0.0:
            print('Error')
            return
        w = denom.I * X.T * y
        return w

    def rss_error(self, y, y_hat):
        return ((array(y) - array(y_hat))**2).sum()

    def stage_wise(self, X, y, eps = 0.01, iter = 100):
        X = mat(X)
        y = mat(y).T
        m,n = shape(X)
        w_mat = mat(zeros((iter,n)))
        ws = mat(zeros((n,1)))
        ws_max = ws.copy()
        for i in range(iter):
            print(ws)
            lowest_error = inf 
            for j in range(n):
                for sign in [-1, 1]:
                    ws_test = ws.copy()
                    ws_test[j] += eps*sign
                    y_test = X * ws_test
                    rsse = self.rss_error(y,y_test)
                    if rsse < lowest_error:
                        lowest_error = rsse
                        ws_max = ws_test
                ws = ws_max.copy()
            w_mat[i] = ws.T
        return w_mat.round(4)

