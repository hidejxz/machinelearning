import numpy as np
import operator
from collections import Counter

class Tree:

    def entropy(self, dataset):
        m = len(dataset)
        labels = [example[-1] for example in dataset]
        label_counts = dict(Counter(labels))
        ent = 0.0
        for key in label_counts:
            prob = float(label_counts[key])/m
            ent -= prob * np.log2(prob)
        return ent

    def split_dataset(self, dataset, axis, value):
        ret_dataset = []
        for i in dataset:
            if i[axis] == value:
                reduced_feat = i[:axis]
                reduced_feat.extend(i[axis+1:])
                ret_dataset.append(reduced_feat)
        return ret_dataset

    def choose_split_feature(self,dataset):
        num_feat = len(dataset[0]) - 1
        base_ent = self.entropy(dataset)
        best_infogain = 0.0
        best_feat = -1
        for i in range(num_feat):
            feat_list = [example[i] for example in dataset]
            unique_vals = set(feat_list)
            new_ent = 0.0
            for value in unique_vals:
                sub_dataset = self.split_dataset(dataset, i, value)
                prob = len(sub_dataset)/float(len(dataset))
                new_ent += prob * self.entropy(sub_dataset)
            infogain = base_ent - new_ent
            if (infogain > best_infogain):
                best_infogain = infogain
                best_feat = i
        return best_feat

    def vote(class_list):
        dic = dict(Counter(class_list))
        sorted_class = sorted(dic.items(), key = operator.itemgetter(1), reverse = True)
        return sorted_class[0][0]

    def create_tree(self,dataset, labels):
        class_list = [example[-1] for example in dataset]
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(dataset[0]) == 1:
            return self.vote(class_list)
        best_feat_index = self.choose_split_feature(dataset)
        best_feat_label = labels[best_feat_index]
        mytree = {best_feat_label:{}}
        del(labels[best_feat_index])
        feat_values = [example[best_feat_index] for example in dataset]
        unique_vals = set(feat_values)
        for value in unique_vals:
            sub_labels = labels[:]
            sub_dataset = self.split_dataset(dataset, best_feat_index, value)
            mytree[best_feat_label][value] = self.create_tree(sub_dataset,sub_labels)
        return mytree 

    def classify(self, input_tree, feat_labels, test_vec):
        first_str = list(input_tree.keys())[0]
        second_dict = input_tree[first_str]
        feat_index = feat_labels.index(first_str)
        for key in second_dict.keys():
            if test_vec[feat_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = self.classify(second_dict[key], feat_labels, test_vec)
                else:
                    class_label = second_dict[key]
        return class_label
    




