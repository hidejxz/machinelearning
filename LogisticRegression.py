import numpy as np
import matplotlib.pyplot as plt
import random

class LogisticRegression:


    def load_dataset(self, filename):
        x = []
        y = []
        fr = open(filename)
        for line in fr.readlines():
            curr_arr = line.strip().split()
            line_arr = [1.0]
            n = len(curr_arr) - 1
            for i in range(n):
                line_arr.append(float(curr_arr[i]))
            x.append(line_arr)
            y.append(int(float(curr_arr[-1])))
        return x, y

    def sigmoid(self, z):
        if z >= 0:
            return 1.0 / (1+np.exp(-z))
        else:
            return np.exp(z) / (1+np.exp(z))

    '''        
    def grad_ascent(self, x, y, alpha = 0.001, num_iter = 500):
        data_mat = np.mat(x)
        label_mat = np.mat(y).transpose()
        m,n = shape(data_mat)
        w = ones((n,1))
        for k in range(num_iter):
            h = self.sigmoid(data_mat * w)
            e = label_mat - h
            w += alpha * data_mat.transpose() * e
        return w
    '''

    def stoc_grad_ascent(self, x, y, num_iter = 1000):
        m,n = np.shape(x)
        w = np.ones(n)
        for i in range(num_iter):
            data_index = list(range(m))
            for k in range(m):
                alpha = 4 / (1.0+i+k) + 0.01
                rand_index1 = int(random.uniform(0, len(data_index)))
                rand_index = data_index[rand_index1]
                h = self.sigmoid(sum(x[rand_index] * w))
                e = y[rand_index] - h
                w += alpha * np.array(x[rand_index]) * e
                del(data_index[rand_index1])
        return w

    def predict(self, x, w):
        result = []
        for i in x:
            prob = self.sigmoid(sum(np.array(i)*w))
            if prob >= 0.5:
                result.append(1)
            else:
                result.append(0)
        return result

    def valid(self, x_train, y_train, x_test, y_test, num_valid = 10, num_iter = 1000):
        error_rate_list = []
        for i in range(num_valid):
            w = self.stoc_grad_ascent(x_train, y_train, num_iter = num_iter)
            result = self.predict(x_test, w)
            error_rate = sum(abs(np.array(result)-np.array(y_test)))/len(y_test)
            error_rate_list.append(error_rate)
        return error_rate_list, np.average(error_rate_list)
            

    def plot(self, w, data_mat, label_mat):
        data_arr = np.array(data_mat)
        n = shape(data_arr)[0]
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(n):
            if int(label_mat[i]) == 1:
                xcord1.append(data_arr[i,1])
                ycord1.append(data_arr[i,2])
            else:
                xcord2.append(data_arr[i,1])
                ycord2.append(data_arr[i,2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
        ax.scatter(xcord2, ycord2, s = 30, c = 'green')
        x = arange(-3.0, 3.0, 0.1)
        y = (- w[0] - w[1]*x) / w[2]
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()  

