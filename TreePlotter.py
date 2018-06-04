import matplotlib.pyplot as plt 

class TreePlotter:
    decision_node = dict(boxstyle = 'sawtooth', fc = '0.8')
    leaf_node = dict(boxstyle = 'round4', fc = '0.8')
    arrow_args = dict(arrowstyle = '<-')

    def plot_node(self, node_txt, center_pt, parent_pt, node_type):
        self.ax1.annotate(node_txt, xy = parent_pt, xycoords = 'axes fraction', 
            xytext = center_pt, textcoords = 'axes fraction', va = 'center', 
            ha = 'center', bbox = node_type, arrowprops = self.arrow_args)

    def create_plot(self,intree):
        fig = plt.figure(1, facecolor = 'white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon = False, **axprops)
        self.totalw = float(self.get_num_leafs(intree))
        self.totald = float(self.get_tree_depth(intree))
        self.xoff = -0.5/self.totalw
        self.yoff = 1.0
        self.plot_tree(intree, (0.5, 1.0), '')
        plt.show()

    def get_num_leafs(self, mytree):
        num_leafs = 0
        first_str = list(mytree.keys())[0]
        second_dict = mytree[first_str]
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                num_leafs += self.get_num_leafs(second_dict[key])
            else:
                num_leafs += 1
        return num_leafs

    def get_tree_depth(self, mytree):
        max_depth = 0
        first_str = list(mytree.keys())[0]
        second_dict = mytree[first_str]
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                this_depth = 1 + self.get_tree_depth(second_dict[key])
            else:
                this_depth = 1
        return this_depth

    def plot_mid_text(self, cntr_pt, parent_pt, txtstring):
        xmid = (parent_pt[0]-cntr_pt[0])/2.0 + cntr_pt[0]
        ymid = (parent_pt[1]-cntr_pt[1])/2.0 + cntr_pt[1]
        self.ax1.text(xmid, ymid, txtstring)

    def plot_tree(self, mytree, parent_pt, nodetxt):
        num_leafs = self.get_num_leafs(mytree)
        depth = self.get_tree_depth(mytree)
        first_str = list(mytree.keys())[0]
        cntr_pt = (self.xoff + (1.0 + float(num_leafs))/2.0/self.totalw,self.yoff)
        self.plot_mid_text(cntr_pt, parent_pt, nodetxt)
        self.plot_node(first_str, cntr_pt, parent_pt, self.decision_node)
        second_dict = mytree[first_str]
        self.yoff -= 1.0/self.totald
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                self.plot_tree(second_dict[key], cntr_pt, str(key))
            else:
                self.xoff += 1.0/self.totalw
                self.plot_node(second_dict[key], (self.xoff, self.yoff), 
                    cntr_pt, self.leaf_node)
                self.plot_mid_text((self.xoff,self.yoff), cntr_pt, str(key))
        self.yoff += 1.0/self.totald

