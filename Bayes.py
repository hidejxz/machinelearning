import numpy as np
import random

class Bayes():

    def load_dataset(self):
        posting_list = [
            ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
            ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
            ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
            ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
            ['mr', 'licks', 'ate', 'my', 'steak', 'how','to', 'stop', 'him'],
            ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        class_vec = [0,1,0,1,0,1]
        return posting_list, class_vec

    def create_vocab_list(self, dataset):
        vocab_set = set([])
        for document in dataset:
            vocab_set = vocab_set | set(document)
        return list(vocab_set)

    def words2vec(self, vocab_list, input_set,is_bag = 0):
        return_vec = [0]*len(vocab_list)
        for word in input_set:
            if word in vocab_list:
                if is_bag == 1:
                    return_vec[vocab_list.index(word)] += 1
                else:
                    return_vec[vocab_list.index(word)] = 1
            else:
                print('the word: %s is not in Vocabulary!' % word)
        return return_vec

    def naive_bayes(self, data_mat, labels):
        
        '''
        m,n = np.shape(data_mat)
        p_abusive = sum(labels)/float(m)
        p0_num = np.zeros(n)
        p1_num = np.zeros(n)
        p0_denom = 0.0
        p1_denom = 0.0
        for i in range(m):
            if labels[i] == 1:
                p1_num += data_mat[i]
                p1_denom += sum(data_mat[i])
            else:
                p0_num += data_mat[i]
                p0_denom += sum(data_mat[i])
        p1_vec = p1_num/p1_denom
        p0_vec = p0_num/p0_denom
        '''
        m,n = np.shape(data_mat)
        p_abusive = sum(labels)/float(m)
        train_mat = np.mat(data_mat)
        train_mat0 = train_mat[np.nonzero(np.array(labels)==0)[0],:]
        p0_vec = np.log(np.array(
            (train_mat0.sum(axis = 0)+1)/(train_mat0.sum()+n))[0])
        train_mat1 = train_mat[np.nonzero(np.array(labels)==1)[0],:]
        p1_vec = np.log(np.array(
            (train_mat1.sum(axis = 0)+1)/(train_mat1.sum()+n))[0])
        

        return p0_vec, p1_vec, p_abusive
        
    def classify_nb(self, vec, p0_vec, p1_vec, p_label1):
        p1 = sum(vec*p1_vec) + np.log(p_label1)
        p0 = sum(vec*p0_vec) + np.log(1.0-p_label1)
        return 1 if p1>p0 else 0


    def text_parse(self, big_string):
        import re 
        list_of_tokens = re.split(r'\W', big_string)
        return [tok.lower() for tok in list_of_tokens if len(tok) > 2]

    def load_emails(self):
        doclist = []
        classlist = []
        fulltext = []
        for i in range(1,26):
            wordlist = self.text_parse(
                open('email/spam/%d.txt' % i).read())
            doclist.append(wordlist)
            fulltext.extend(wordlist)
            classlist.append(1)
            wordlist = self.text_parse(
                open('email/ham/%d.txt' % i).read())
            doclist.append(wordlist)
            fulltext.extend(wordlist)
            classlist.append(0)
        return doclist, classlist, fulltext

    def split_train_test(self, m, q):
        trainset = list(range(50))
        testset = []
        for i in range(int(m*q)):
            rand_index = int(random.uniform(0,len(trainset)))
            testset.append(trainset[rand_index])
            del(trainset[rand_index])
        return trainset, testset


    def spam_test(self):
        doclist, classlist, fulltext = self.load_emails()
        vocab_list = self.create_vocab_list(doclist)
        trainset, testset = self.split_train_test(50, 0.2)
        x_train = []
        y_train = []
        for docindex in trainset:
            x_train.append(self.words2vec(
                vocab_list, doclist[docindex], is_bag = 1))
            y_train.append(classlist[docindex])
        p0v, p1v, pspam = self.naive_bayes(x_train,y_train)
        error_cnt = 0
        for docindex in testset:
            word_vector = self.words2vec(
                vocab_list, doclist[docindex], is_bag = 1)
            result = self.classify_nb(
                np.array(word_vector), p0v, p1v, pspam)
            if result != classlist[docindex]:
                error_cnt += 1
        error_rate = float(error_cnt)/len(testset)
        print('error rate: %.4f' % error_rate)
        
















