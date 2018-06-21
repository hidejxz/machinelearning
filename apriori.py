'''
D: the whole dataset
Ck: all possiable elements, C1 = [1,2,3] C2 = [[1,2],[2,3],[1,3]]
Lk: Ck which greater than min_support
ret_list: [L1, L2, L3...]
'''

def load_dataset():
    return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]

def create_C1(dataset):
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

def scan_D(D, Ck, min_support):
    ss_Cnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can in ss_Cnt: 
                    ss_Cnt[can] += 1
                else:
                    ss_Cnt[can] = 1
    num_items = float(len(D))
    Lk = []
    support_data = {}
    for key in ss_Cnt:
        support = ss_Cnt[key]/num_items
        if support >= min_support:
            #ret_list.insert(0,key)
            Lk.append(key)
        support_data[key] = support
    return Lk, support_data

def apriori_gen(Lk):
    k = len(Lk[0])
    ret_list = []
    len_Lk = len(Lk)
    for i in range(len_Lk):
        for j in range(i+1, len_Lk):
            L1 = list(Lk[i])[:k-1]
            L2 = list(Lk[j])[:k-1]
            L1.sort()
            L2.sort()
            if L1 == L2:
                ret_list.append(Lk[i] | Lk[j])
    return ret_list


def apriori(dataset, min_support = 0.5):
    C1 = create_C1(dataset)
    D = list(map(set,dataset))
    L1, support_data = scan_D(D, C1, min_support)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = apriori_gen(L[k-2])
        Lk, supk = scan_D(D, Ck, min_support)
        support_data.update(supk)
        L.append(Lk)
        k += 1
    return L, support_data

def generate_rules(L, support_data, min_conf = 0.7):
    big_rule_list = []
    for i in range(1, len(L)):
        #print('*'*10,i,'*'*10)
        for freq_set in L[i]:
            #print('-'*10 ,'freq_set: %s' % freq_set, '-'*10)
            H1 = [frozenset([item]) for item in freq_set]
            if (i > 1):
                rules_from_conseq(freq_set, H1, support_data, \
                    big_rule_list, min_conf)
                #print('big_rule_list',big_rule_list)
            else:
                calc_conf(freq_set, H1, support_data, big_rule_list, min_conf)
    return big_rule_list

def calc_conf(freq_set, H, support_data, brl, min_conf = 0.7):
    pruned_H = []
    for conseq in H:
        conf = round(support_data[freq_set]/support_data[freq_set-conseq],2)
        if conf >= min_conf:
            print(freq_set-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freq_set-conseq, conseq, conf))
            pruned_H.append(conseq)
    return pruned_H

def rules_from_conseq(freq_set, H, support_data, brl, min_conf = 0.7):
    m = len(H[0])
    if len(freq_set) > m:
        hmp1 = calc_conf(freq_set, H, support_data, brl, min_conf)
        if len(hmp1) > 1:
            hmp1 = apriori_gen(hmp1)
            rules_from_conseq(freq_set, hmp1, support_data, brl, min_conf)
    return













