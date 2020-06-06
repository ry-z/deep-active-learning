import numpy as np
from .strategy import Strategy
import pickle
from datetime import datetime
from torch.distributions import MultivariateNormal, kl_divergence
from sklearn.mixture import GaussianMixture

class Gaussian(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, n_components, n_subset):
        super(Gaussian, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.n_components = n_components
        self.n_subset = n_subset
        
    def getKL(self, gmm_p, gmm_q, X):
        log_p_X= gmm_p.score_samples(X[0])
        log_q_X= gmm_q.score_samples(X[0])
        return log_p_X.mean() - log_q_X.mean()
    
    def subset_select(self):
        l_set = self.X[self.idxs_lb] #select labelled set
        u_set = self.X[~self.idxs_lb] #select unlabeled set
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        chunks = [u_set[x:x+self.n_subset] for x in range(0, len(u_set), self.n_subset)]
        idxs_chunks = [idxs_unlabeled[x:x+self.n_subset] for x in range(0, len(idxs_unlabeled), self.n_subset)]
        nsamples, nx, ny = self.X.shape
        g_whole = GaussianMixture(self.n_components).fit(self.X.reshape((nsamples, nx*ny)))
        gen_X = g_whole.sample(n_samples=5000)
        kl_list = []
        for i in range(0, len(chunks)):
            subset = np.concatenate((l_set, chunks[i]), axis=0)
            nsubsamples, nx, ny = subset.shape
            g_subset = GaussianMixture(self.n_components).fit(subset.reshape((nsubsamples, nx*ny)))
            kl_list.append(self.getKL(g_whole, g_subset, gen_X))
        
        selected_chunk_idx = np.argmin(kl_list)
        return idxs_chunks[selected_chunk_idx]
        
    def query(self, n):
        selected_chunk_idxs = self.subset_select()
        idxs_chunk_unlabeled = np.arange(self.n_pool)[selected_chunk_idxs]
        probs = self.predict_prob(self.X[selected_chunk_idxs], self.Y[selected_chunk_idxs])
        U = probs.max(1)[0]
        #print(type(idxs_chunk_unlabeled[U.sort()[1]:n]))
        return idxs_chunk_unlabeled[U.sort()[1][:n]]
        
        
        
