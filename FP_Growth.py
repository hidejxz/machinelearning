from operator import itemgetter

class TreeNode:
    def __init__(self, name_value, num, parent_node):
        self.name = name_value
        self.count = num
        self.node_link = None 
        self.parent = parent_node
        self.children = {}

    def inc(self, num):
        self.count += num

    def disp(self, ind = 1):
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)



def load_simple_data():
    dataset = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return dataset

def create_init_set(dataset):
    ret_dict = {}
    for trans in dataset:
        ret_dict[frozenset(trans)] = 1
    return ret_dict

def create_tree(dataset, min_sup = 1):
    header = {}
    for trans in dataset:
        for item in trans:
            header[item] = header.get(item,0) + dataset[trans]
    key_list = list(header.keys())
    for k in key_list:
        if header[k] < min_sup:
            del(header[k])
    freq_itemset = set(header.keys())
    if len(freq_itemset) == 0:
        return None, None
    for k in header:
        header[k] = [header[k], None]
    ret_tree = TreeNode('Null Set', 1, None)
    for transet, count in dataset.items():
        local_D = {}
        for item in transet:
            if item in freq_itemset:
                local_D[item] = header[item][0]
        if len(local_D) > 0:
            ordered_items = [v[0] for v in sorted(local_D.items(), 
                            key = itemgetter(0,1), reverse = True)]
                            #key = lambda p: p[1], reverse = True)]
            update_tree(ordered_items, ret_tree, header, count)

        
    return ret_tree, header

def update_tree(items, in_tree, header, count):
    if items[0] in in_tree.children:
        in_tree.children[items[0]].inc(count)
    else:
        in_tree.children[items[0]] = TreeNode(items[0], count, in_tree)
        if header[items[0]][1] == None:
            header[items[0]][1] = in_tree.children[items[0]]
        else:
            update_header(header[items[0]][1], 
                          in_tree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1:], in_tree.children[items[0]], header, 
                    count)

def update_header(node_to_test, target_node):
    while node_to_test.node_link is not None:
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node

def ascend_tree(leaf_node, prefix_path):
    if leaf_node.parent is not None:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefix_path)

def find_prefix_path(base_pat, tree_node):
    cond_pats = {}
    while tree_node is not None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            cond_pats[frozenset(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    return cond_pats

def mine_tree(in_tree, header, min_sup, prefix, freq_itemlist):
    bigL = [v[0] for v in sorted(header.items(), 
            key = lambda p: (p[1][0],p[0]))]
    print(bigL)
    for base_pat in bigL:
        new_freqset = prefix.copy()
        new_freqset.add(base_pat)
        freq_itemlist.append(new_freqset)
        cond_patt_bases = find_prefix_path(base_pat, header[base_pat][1])
        cond_tree, my_head = create_tree(cond_patt_bases, min_sup)
        if my_head is not None:
            print('conditional tree for: ', new_freqset)
            cond_tree.disp(1)
            mine_tree(cond_tree, my_head, min_sup, new_freqset, 
                      freq_itemlist)







