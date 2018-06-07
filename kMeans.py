import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class kMeans():

    def load_dataset(self, filename):
        data_mat = []
        fr = open(filename)
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            flt_line = list(map(np.float,cur_line))
            data_mat.append(flt_line)
        return data_mat

    def euler_dist(self, vec_a, vec_b):
        return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))

    def rand_cent(self, dataset, k):
        n = np.shape(dataset)[1]
        centroids = np.mat(np.zeros((k,n)))
        for j in range(n):
            min_vals = min(dataset[:,j])
            ranges = float(max(dataset[:,j]) - min_vals)
            centroids[:,j] = min_vals + ranges*np.random.rand(k,1)
        return centroids

    def kmeans(self, dataset, k, dist_meas=euler_dist, create_cent=rand_cent):
        m = np.shape(dataset)[0]
        cluster_assment = np.mat(np.zeros((m,2)))
        centroids = create_cent(self,dataset,k)
        cluster_changed = True
        while cluster_changed:
            cluster_changed = False
            for i in range(m):
                min_dist = np.inf
                min_index = -1
                for j in range(k):
                    dist = dist_meas(self, centroids[j,:],dataset[i,:])
                    if dist < min_dist:
                        min_dist = dist
                        min_index = j
                if cluster_assment[i,0] != min_index:
                    cluster_changed = True
                cluster_assment[i,:] = min_index, min_dist**2
            print(centroids)
            for cent in range(k):
                cluster = dataset[np.nonzero(cluster_assment[:,0].A == cent)[0]]
                centroids[cent,:] = np.mean(cluster, axis=0)
        return centroids, cluster_assment

    def bikmeans(self, dataset, k, dist_meas=euler_dist):
        m = np.shape(dataset)[0]
        cluster_assment = np.mat(np.zeros((m,2)))
        centroid0 = np.mean(dataset, axis=0).tolist()[0]
        cent_list = [centroid0]
        for j in range(m):
            cluster_assment[j,1] = dist_meas(self,
                np.mat(centroid0), dataset[j,:])**2
        while (len(cent_list) < k):
            min_sse = np.inf 
            for i in range(len(cent_list)):
                cur_cluster = dataset[
                    np.nonzero(cluster_assment[:,0].A == i)[0],:]
                cur_centroids, cur_cluster_assment = \
                    self.kmeans(cur_cluster, 2, dist_meas)
                print('bb',type(cur_centroids),'bb')
                cur_sse = np.sum(cur_cluster_assment[:,1])
                other_sse = np.sum(
                    cluster_assment[np.nonzero(cluster_assment[:,0].A != i)[0],1])
                print('sseSplit: %.2f, notSplit: %.2f' 
                    % (cur_sse, other_sse))
                if (cur_sse + other_sse) < min_sse:
                    best_cent_to_split = i
                    best_new_cents = cur_centroids
                    best_cluster_assment = cur_cluster_assment.copy()
                    min_sse = cur_sse + other_sse
            best_cluster_assment[
                np.nonzero(best_cluster_assment[:,0].A == 1)[0],0
                ] = len(cent_list)
            best_cluster_assment[
                np.nonzero(best_cluster_assment[:,0].A == 0)[0],0
                ] = best_cent_to_split
            print('best_cent_to_split: %d' % best_cent_to_split)
            print('len of best_cluster_assment: %d' % len(best_cluster_assment))
            print('aa',best_new_cents,'aa')
            cent_list[best_cent_to_split] = best_new_cents[0,:].tolist()
            cent_list.append(best_new_cents[1,:].tolist())
            cluster_assment[
                np.nonzero(cluster_assment[:,0].A == best_cent_to_split)[0],:
                ] = best_cluster_assment
        return cent_list, cluster_assment

    def dist_slc(self, vec_a, vec_b):
        a = np.sin(vec_a[0,1]*np.pi/180) * np.sin(vec_b[0,1]*np.pi/180)
        b = np.cos(vec_a[0,1]*np.pi/180) * np.cos(vec_b[0,1]*np.pi/180) \
            * np.cos(np.pi * (vec_b[0,0]-vec_a[0,0])/180)
        return np.arccos(a + b)*6371.0


    def cluster_clubs(self, k=5, dist_meas=dist_slc):
        dat_list = []
        for line in open('places.txt').readlines():
            line_arr = line.split('\t')
            dat_list.append([float(line_arr[4]), float(line_arr[3])])
        dataset = np.mat(dat_list)
        centroids, cluster_assment = self.bikmeans(dataset, k, dist_meas=dist_meas)
        fig = plt.figure()
        rect = [0.1, 0.1, 0.8, 0.8]
        scatter_markers = ['s','o','^', '8', 'p', 'd', 'v', 'h', '>', '<']
        axprops = dict(xticks=[], yticks=[])
        ax0 = fig.add_axes(rect, label = 'ax0', **axprops)
        imgp = plt.imread('Portland.png')
        ax0.imshow(imgp)
        ax1 = fig.add_axes(rect)
        for i in range(k):
            cur_cluster = dataset[np.nonzero(cluster_assment[:,0].A==i)[0],:]
            marker_style = scatter_markers[i % len(scatter_markers)]
            ax1.scatter(cur_cluster[:,0].flatten().A[0],
                        cur_cluster[:,1].flatten().A[0],
                        marker = marker_style, s = 90)
        print(centroids)
        ax1.scatter(centroids[:,0].flatten().A[0],
                    centroids[:,1].flatten().A[0], marker='+', s=300)
        plt.show()





