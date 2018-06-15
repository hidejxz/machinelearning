import numpy as np 

def load_dataset(filename):
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split()
        flt_line = list(map(float, cur_line))
        data_mat.append(flt_line)
    return data_mat

def reg_leaf(X):
    X = np.array(X)
    return np.mean(X[:,-1])

def reg_err(X):
    X = np.array(X)
    return np.var(X[:,-1]) * np.shape(X)[0]

def linear_solve(dataset):
    
    m,n = np.shape(dataset)
    X = np.mat(np.ones((m,n)))
    X[:,1:n] = dataset[:,0:n-1].copy()
    y = dataset[:,-1].copy()
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('Error')
    ws = xTx.I*X.T*y
    return ws, X, y

def model_leaf(dataset):
    dataset = np.mat(dataset)
    ws, X, y = linear_solve(dataset)
    return ws

def model_err(dataset):
    dataset = np.mat(dataset)
    ws, X, y = linear_solve(dataset)
    y_hat = X * ws
    return np.sum(np.power(y-y_hat,2))

def reg_tree_eval(model, indata):
    return float(model)

def model_tree_eval(model, indata):
    n = np.shape(indata)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = indata
    return float(X*model)


class RegTree():

    def binary_split(self, X, feature, value):
        X = np.array(X)
        mat0 = X[np.nonzero(X[:,feature] > value)[0]]
        mat1 = X[np.nonzero(X[:,feature] <= value)[0]]
        return mat0, mat1

    def create_tree(self, X, leaf_type = reg_leaf, err_type = reg_err, 
                    ops = (1,4)):
        X = np.array(X)
        feat, val = self.best_split(X, leaf_type, err_type, ops)
        if feat == None:
            return val 
        ret_tree = {}
        ret_tree['sp_ind'] = feat 
        ret_tree['sp_val'] = val 
        lset, rset = self.binary_split(X, feat, val)
        ret_tree['left'] = self.create_tree(lset, leaf_type, err_type, ops)
        ret_tree['right'] = self.create_tree(rset, leaf_type, err_type, ops)
        return ret_tree


    def best_split(self, X, leaf_type = reg_leaf, err_type = reg_err, 
                   ops = (1,4)):
        X = np.array(X)
        tol_s = ops[0]
        tol_n = ops[1]
        if len(set(np.array(X)[:,-1])) == 1:
            return None, leaf_type(X)
        m,n = np.shape(X)
        S = err_type(X)
        best_S = np.inf
        best_index = -1
        best_value = 0
        for feat_index in range(n-1):
            for sp_val in set(X[:,feat_index]):
                mat0, mat1 = self.binary_split(X, feat_index, sp_val)
                if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n):
                    continue
                new_S = err_type(mat0) + err_type(mat1)
                if new_S < best_S:
                    best_index = feat_index
                    best_value = sp_val
                    best_S = new_S
        if (S - best_S) < tol_s:
            return None, leaf_type(X)
        mat0, mat1 = self.binary_split(X, best_index, best_value)
        if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n):
            return None, leaf_type(X)
        return best_index, best_value

    def is_tree(self, obj):
        return (type(obj).__name__ == 'dict')

    def get_mean(self, tree):
        if self.is_tree(tree['right']):
            tree['right'] = self.get_mean(tree['right'])
        if self.is_tree(tree['left']):
            tree['left'] = self.get_mean(tree['left'])
        return (tree['left']+tree['right'])/2.0

    def prune(self, tree, test_data):
        test_data = np.array(test_data)
        if np.shape(test_data)[0] == 0:
            return self.get_mean(tree)
        if (self.is_tree(tree['right'])) or (self.is_tree(tree['left'])):
            lset, rset = self.binary_split(test_data, tree['sp_ind'], 
                                           tree['sp_val'])
        if self.is_tree(tree['left']):
            tree['left'] = self.prune(tree['left'], lset)
        if self.is_tree(tree['right']):
            tree['right'] = self.prune(tree['right'], lset)
        if not self.is_tree(tree['left']) and not self.is_tree(tree['right']):
            lset, rset = self.binary_split(test_data, tree['sp_ind'], 
                                           tree['sp_val'])
            error_no_merge = np.sum(np.power(lset[:,-1] - tree['left'],2)) \
                             + np.sum(np.power(rset[:,-1] - tree['right'],2))
            tree_mean = (tree['left'] + tree['right'])/2.0
            error_merge = np.sum(np.power(test_data[:,1] - tree_mean,2))
            if error_merge < error_no_merge:
                print('merging')
                return tree_mean
            else: 
                return tree
        else:
            return tree

    def tree_forecast(self, tree, indata, model_eval = reg_tree_eval):
        if not self.is_tree(tree):
            return model_eval(tree, indata)
        if indata[tree['sp_ind']] > tree['sp_val']:
            if self.is_tree(tree['left']):
                return self.tree_forecast(tree['left'], indata, model_eval)
            else:
                return model_eval(tree['left'], indata)
        else:
            if self.is_tree(tree['right']):
                return self.tree_forecast(tree['right'], indata, model_eval)
            else:
                return model_eval(tree['right'], indata)

    def create_forecast(self, tree, test_data, model_eval = reg_tree_eval):
        m = len(test_data)
        y_hat = np.mat(np.zeros((m,1)))
        for i in range(m):
            y_hat[i] = (self.tree_forecast(tree, np.mat(test_data[i]), 
                        model_eval))
        return y_hat










