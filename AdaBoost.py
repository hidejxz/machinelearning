import numpy as np 

class AdaBoost():

    def stump_classify(self, data_mat, dimen, thresh_val, thresh_ineq):
        ret_array = np.ones((np.shape(data_mat)[0],1))
        if thresh_ineq == 'lt':
            ret_array[data_mat[:,dimen] <= thresh_val] = -1.0
        else:
            ret_array[data_mat[:,dimen] > thresh_val] = -1.0
        return ret_array

    def build_stump(self, data_mat, class_labels, D):
        data_mat = np.mat(data_mat)
        label_mat = np.mat(class_labels).T
        m,n = np.shape(data_mat)
        num_steps = 10.0
        best_stump = {}
        best_class_est = np.mat(np.zeros((m,1)))
        min_error = np.inf
        for i in range(n):
            range_min = data_mat[:,i].min()
            range_max = data_mat[:,i].max()
            step_size = (range_max - range_min)/num_steps
            for j in range(-1,int(num_steps)+1):
                for inequal in ['lt', 'gt']:
                    thresh_val = (range_min + float(j) * step_size)
                    predicted_vals = self.stump_classify(data_mat, i, thresh_val, inequal)
                    err_mat = np.mat(np.ones((m,1)))
                    err_mat[predicted_vals == label_mat] = 0
                    weighted_err = D.T*err_mat
                    print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" 
                          %(i, thresh_val, inequal, weighted_err))
                    if weighted_err < min_error:
                        min_error = weighted_err
                        best_class_est = predicted_vals.copy()
                        best_stump['dim'] = i
                        best_stump['thresh'] = thresh_val
                        best_stump['ineq'] = inequal
        return best_stump, min_error, best_class_est

    def adaboost_train_ds(self, data_mat, class_labels, num_iter = 40):
        weak_class_arr = []
        m = np.shape(data_mat)[0]
        D = np.mat(np.ones((m,1))/m)
        agg_class_est = np.mat(np.zeros((m,1)))
        for i in range(num_iter):
            best_stump, error, class_est = self.build_stump(data_mat, class_labels, D)
            print("D:", D.T)
            alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
            best_stump['alpha'] = alpha
            weak_class_arr.append(best_stump)
            print("class_est: ",class_est.T)
            expon = np.multiply(-1*alpha*np.mat(class_labels).T, class_est)
            D = np.multiply(D,np.exp(expon))
            D = D/D.sum()
            agg_class_est += alpha*class_est
            print("agg_class_est: ", agg_class_est.T)
            agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m,1))) 
            error_rate = agg_errors.sum()/m
            print("total error: ",error_rate,"\n")
            if error_rate == 0.0: break
        return weak_class_arr