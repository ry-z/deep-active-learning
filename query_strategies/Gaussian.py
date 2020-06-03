import numpy as np
from .strategy import Strategy
import pickle
from datetime import datetime
from torch.distributions import MultivariateNormal, kl_divergence
import sklearn.mixture.GMM

class Gaussian(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, tor=1e-4, n_components, n_subset):
		super(GaussianDis, self).__init__(X, Y, idxs_lb, net, handler, args)
		self.tor = tor
		self.global_dis = None
		self.local_dis = []
		self.n_components = n_components
		
    def _getKL(gmm_p, gmm_q, n_samples=10**5):
	    self.X = gmm_p.sample(n_samples)
		log_p_X, _ = gmm_p.score_samples(X)
		log_q_X, _ = gmm_q.score_samples(X)
		return log_p_X.mean() - log_q_X.mean()
	
	def subset_select(self):
	    l_set = self.X[idxs_lb] #select labelled set
		u_set = self.X[~idxs_lb] #select unlabeled set
		idxs_unlabeled = np.arange(self.n_pool)[~idxs_lb]
		chunks = [u_set[x:x+n_subset] for x in range(0, len(u_set), n_subset)]
		idxs_chunks = [idxs_unlabeled[x:x+n_subset] for x in range(0, len(idxs_unlabeled), n_subset)]
	    g_whole = GMM(self.n_components).fit(self.X)
		kl_list = []
		for i in range(0, len(chunks)):
		    subset = np.concatenate((l_set, chunks[i]), axis=0)
		    g_subset = GMM(self.n_components).fit(subset)
			kl_list.append(_getKL(g_whole, g_subset))
		
		selected_chunk_idx = np.argmin(kl_list)
		return idxs_chunks[selected_chunk_index]]
		
	def query(self, n):
	    selected_chunk_idxs = subset_select()
		idxs_chunk_unlabeled = np.arange(self.n_pool)[selected_chunk_idxs]
		probs = self.predict_prob(self.X[selected_chunk_idxs], self.Y[selected_chunk_idxs])
		U = probs.max(1)[0]
		return idxs_chunk_unlabeled[U.sort()[1]:n]
		
		
		