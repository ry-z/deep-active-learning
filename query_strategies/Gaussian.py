import numpy as np
from .strategy import Strategy
import pickle
from datetime import datetime
from torch.distributions import MultivariateNormal, kl_divergence
import sklearn.mixture.GMM

class GaussianDis(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, tor=1e-4, n_components, n_subset):
		super(GaussianDis, self).__init__(X, Y, idxs_lb, net, handler, args)
		self.tor = tor
		self.global_dis = None
		self.local_dis = []
		self.n_components = n_components
		
    def _getKL(mu1, var1, mu2, var2):
	    p = MultivariateNormal(mu1, var1)
		q = MultivariateNormal(mu2, var2)
		kl_loss = kl_divergence(p, q)
		return kl_loss
	
	def subset_select(self):
	    l_set = X[idxs_lb==True] #select labelled set
		u_set = X[idxs_lb==False] #select unlabeled set
		chunks = [u_set[x:x+n_subset] for x in range(0, len(u_set), n_subset)]
	    g_whole = GMM(self.n_components).fit(X)
		kl_list = []
		for i in range(0, len(chunks)):
		    subset = np.concatenate((l_set, chunks[i]), axis=0)
		    g_subset = GMM(self.n_components).fit(subset)
			kl_list.append(_getKL())
		
		selected_chunk = chunks[np.argmin(kl_list)]
		return selected_chunk
		