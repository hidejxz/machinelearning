import numpy as np 
import operator

class kNN:

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
        norm_mat, ranges, min_val = self.auto_norm(data_mat)
        m = norm_mat.shape[0]
        num_test_vec = int(m*rate)
        error_count = 0.0
        for i in range(num_test_vec):
            classifier_result = self.classify(norm_mat[i,:], 
                                              norm_mat[num_test_vec:m,:],
                                              label_mat[num_test_vec:m],3)
            print('the classifier came back with: %d, the real answer is %d'
                  %(classifier_result, label_mat[i]))
            if classifier_result != label_mat[i]:
                error_count += 1
        print('the total error rate is %f' % (error_count/float(num_test_vec)))

    def classify_person(self, data_mat, label_mat):
        result_list = ['not at all','in small doses','in large doses']
        percent_tats = float(input('percentage of time spent playing video games?: '))
        ffmiles = float(input('frequent flier miles earned per year?: '))
        ice_cream = float(input('liters of ice cream consumed per year?: '))
        norm_mat, ranges, min_val = self.auto_norm(data_mat)
        in_arr = np.array([ffmiles, percent_tats, ice_cream])
        classifier_result = self.classify((in_arr-min_val)/ranges,norm_mat,label_mat,3)
        print('You will probably like this person: ', result_list[classifier_result-1])

    def img2vector(self, filename):
        return_vec = np.zeros((1,1024))
        fr = open(filename)
        for i in range(32):
            line_str = fr.readline()
            for j in range(32):
                return_vec[0,32*i+j] = int(line_str[j])
        return return_vec

    def load_img(self, dirname):
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
            labels.append(class_num)
            data_mat[i,:] = self.img2vector(dirname+'/'+file_list[i])
        return data_mat, labels

    def handwriting(self, train_dir, test_dir, k):
        train_data,train_label = self.load_img(train_dir)
        test_data,test_label = self.load_img(test_dir)
        error_count = 0.0
        m = len(test_label)
        for i in range(m):
            result = self.classify(test_data[i],train_data,train_label,k)
            if result != test_label[i]:
                error_count += 1
        print('the total number of errors is %d, error rate is %f' %
              (error_count,error_count/float(m)))







