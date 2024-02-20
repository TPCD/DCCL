import numpy as np
import copy
import random
from project_utils.cluster_utils import cluster_acc
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import check_random_state
import torch

def pairwise_distance(data1, data2, batch_size=None):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''
    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)

    if batch_size == None:
        dis = (A-B)**2
        #return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        #  torch.cuda.empty_cache()
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if(i+batch_size < data1.shape[0]):
                dis_batch = (A[i:i+batch_size]-B)**2
                dis_batch = dis_batch.sum(dim=-1)
                dis[i:i+batch_size] = dis_batch
                i = i+batch_size
                #  torch.cuda.empty_cache()
            elif(i+batch_size >= data1.shape[0]):
                dis_final = (A[i:] - B)**2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
    #  torch.cuda.empty_cache()
    return dis


class K_Means:

    def __init__(self, k=3, tolerance=1e-4, max_iterations=100, init='k-means++',
                 n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=None, mode=None):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pairwise_batch_size = pairwise_batch_size
        self.mode = mode

    def split_for_val(self, l_feats, l_targets, val_prop=0.2):

        np.random.seed(0)

        # Reserve some labelled examples for validation
        num_val_instances = int(val_prop * len(l_targets))
        val_idxs = np.random.choice(range(len(l_targets)), size=(num_val_instances), replace=False)
        val_idxs.sort()
        remaining_idxs = list(set(range(len(l_targets))) - set(val_idxs.tolist()))
        remaining_idxs.sort()
        remaining_idxs = np.array(remaining_idxs)

        val_l_targets = l_targets[val_idxs]
        val_l_feats = l_feats[val_idxs]

        remaining_l_targets = l_targets[remaining_idxs]
        remaining_l_feats = l_feats[remaining_idxs]

        return remaining_l_feats, remaining_l_targets, val_l_feats, val_l_targets


    def kpp(self, X, pre_centers=None, k=10, random_state=None):
        random_state = check_random_state(random_state)

        if pre_centers is not None:

            C = pre_centers

        else:

            C = X[random_state.randint(0, len(X))]

        C = C.view(-1, X.shape[1])

        while C.shape[0] < k:

            dist = pairwise_distance(X, C, self.pairwise_batch_size)
            dist = dist.view(-1, C.shape[0])
            d2, _ = torch.min(dist, dim=1)
            prob = d2/d2.sum()
            cum_prob = torch.cumsum(prob, dim=0)
            r = random_state.rand()

            if len((cum_prob >= r).nonzero()) == 0:
                debug = 0
            else:
                ind = (cum_prob >= r).nonzero()[0][0]
            C = torch.cat((C, X[ind].view(1, -1)), dim=0)

        return C


    def fit_once(self, X, random_state):

        centers = torch.zeros(self.k, X.shape[1]).type_as(X)
        labels = -torch.ones(len(X))
        #initialize the centers, the first 'k' elements in the dataset will be our initial centers

        if self.init == 'k-means++':
            centers = self.kpp(X, k=self.k, random_state=random_state)

        elif self.init == 'random':

            random_state = check_random_state(self.random_state)
            idx = random_state.choice(len(X), self.k, replace=False)
            for i in range(self.k):
                centers[i] = X[idx[i]]

        else:
            for i in range(self.k):
                centers[i] = X[i]

        #begin iterations

        best_labels, best_inertia, best_centers = None, None, None
        for i in range(self.max_iterations):

            centers_old = centers.clone()
            dist = pairwise_distance(X, centers, self.pairwise_batch_size)
            mindist, labels = torch.min(dist, dim=1)
            inertia = mindist.sum()

            for idx in range(self.k):
                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(X, 0, selected)
                centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
            if center_shift ** 2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1


    def fit_mix_once(self, u_feats, l_feats, l_targets, random_state):

        def supp_idxs(c):
            return l_targets.eq(c).nonzero().squeeze(1)

        l_classes = torch.unique(l_targets)
        support_idxs = list(map(supp_idxs, l_classes))
        l_centers = torch.stack([l_feats[idx_list].mean(0) for idx_list in support_idxs])
        cat_feats = torch.cat((l_feats, u_feats))

        centers = torch.zeros([self.k, cat_feats.shape[1]]).type_as(cat_feats)
        centers[:len(l_classes)] = l_centers

        labels = -torch.ones(len(cat_feats)).type_as(cat_feats).long()

        l_classes = l_classes.cpu().long().numpy()
        l_targets = l_targets.cpu().long().numpy()
        l_num = len(l_targets)
        cid2ncid = {cid:ncid for ncid, cid in enumerate(l_classes)}  # Create the mapping table for New cid (ncid)
        for i in range(l_num):
            labels[i] = cid2ncid[l_targets[i]]

        #initialize the centers, the first 'k' elements in the dataset will be our initial centers
        centers = self.kpp(u_feats, l_centers, k=self.k, random_state=random_state)

        # Begin iterations
        best_labels, best_inertia, best_centers = None, None, None
        for it in range(self.max_iterations):
            centers_old = centers.clone()

            dist = pairwise_distance(u_feats, centers, self.pairwise_batch_size)
            u_mindist, u_labels = torch.min(dist, dim=1)
            u_inertia = u_mindist.sum()
            l_mindist = torch.sum((l_feats - centers[labels[:l_num]])**2, dim=1)
            l_inertia = l_mindist.sum()
            inertia = u_inertia + l_inertia
            labels[l_num:] = u_labels

            for idx in range(self.k):

                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(cat_feats, 0, selected)
                centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))

            if center_shift ** 2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1


    def fit(self, X):
        random_state = check_random_state(self.random_state)
        best_inertia = None
        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):
                labels, inertia, centers, n_iters = self.fit_once(X, random_state)
                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters
        else:
            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(self.fit_once)(X, seed) for seed in seeds)
            # Get results with the lowest inertia
            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]


    def fit_mix(self, u_feats, l_feats, l_targets):

        random_state = check_random_state(self.random_state)
        best_inertia = None
        fit_func = self.fit_mix_once

        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):

                labels, inertia, centers, n_iters = fit_func(u_feats, l_feats, l_targets, random_state)

                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters

        else:

            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(fit_func)(u_feats, l_feats, l_targets, seed)
                                                              for seed in seeds)
            # Get results with the lowest inertia

            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]


def main():

    import matplotlib.pyplot as plt
    from matplotlib import style
    import pandas as pd
    style.use('ggplot')
    from sklearn.datasets import make_blobs
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
    X, y = make_blobs(n_samples=500,
                      n_features=2,
                      centers=4,
                      cluster_std=1,
                      center_box=(-10.0, 10.0),
                      shuffle=True,
                      random_state=1)  # For reproducibility

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    #  X = torch.from_numpy(X).float().to(device)


    y = np.array(y)
    l_targets = y[y>1]
    l_feats = X[y>1]
    u_feats = X[y<2]
    cat_feats = np.concatenate((l_feats, u_feats))
    y = np.concatenate((y[y>1], y[y<2]))
    cat_feats = torch.from_numpy(cat_feats).to(device)
    u_feats = torch.from_numpy(u_feats).to(device)
    l_feats = torch.from_numpy(l_feats).to(device)
    l_targets = torch.from_numpy(l_targets).to(device)

    km = K_Means(k=4, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=10)

    #  km.fit(X)

    km.fit_mix(u_feats, l_feats, l_targets)
    #  X = X.cpu()
    X = cat_feats.cpu()
    centers = km.cluster_centers_.cpu()
    pred = km.labels_.cpu()
    print('nmi', nmi_score(pred, y))

    # Plotting starts here
    colors = 10*["g", "c", "b", "k", "r", "m"]

    for i in range(len(X)):
        x = X[i]
        plt.scatter(x[0], x[1], color = colors[pred[i]],s = 10)

    for i in range(4):
        plt.scatter(centers[i][0], centers[i][1], s = 130, marker = "*", color='r')
    plt.show()

if __name__ == "__main__":
    main()