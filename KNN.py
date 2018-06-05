import numpy as np 
import operator

class KNN:

    def classify(self, x, dataset, labels, k):
        dataset_size = dataset.shape[0]
        diff_mat = np.tile(x, (dataset_size, 1)) - dataset
        sq_diff_mat = diff_mat**2
        sq_distances = sq_diff_mat.sum(axis = 1)
        distances = sq_distances**0.5
        sorted_dist_indicies = distances.argsort()
        class_count = {}
        for i in range(k):
            vote_label = labels[sorted_dist_indicies[i]]
            class_count[vote_label] = class_count.get(vote_label,0) + 1
        sorted_class_count = sorted(class_count.items(), 
                                    key = operator.itemgetter(1), 
                                    reverse = True)
        return sorted_class_count[0][0]

    def load_dataset(self,filename):
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
            label_mat.append(int(cur_line[-1]))
        return np.array(data_mat), np.array(label_mat)

    def auto_norm(self, dataset):
        min_val = dataset.min(0)
        max_val = dataset.max(0)
        ranges = max_val - min_val
        m = dataset.shape[0]
        norm_dataset = dataset - np.tile(min_val, (m,1))
        norm_dataset /= np.tile(ranges, (m,1))
        return norm_dataset, ranges, min_val

    def dating_class_test(self, data_mat, label_mat):
        rate = 0.1
        norm_dataset, ranges, min_val = self.auto_norm(data_mat)
        m = 