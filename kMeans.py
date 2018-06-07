import numpy as np

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
                    dist = dist_meas(self,centroids[j,:],dataset[i,:])
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







