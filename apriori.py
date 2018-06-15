

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
    ret_list = []
    support_data = {}
    for key in ss_Cnt:
        support = ss_Cnt[key]/num_items
        if support >= min_support:
            ret_list.insert(0,key)
        support_data[key] = support
    return ret_list, support_data