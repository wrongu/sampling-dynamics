if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
from graphical_models import BayesNet, DiscreteVariable
from sampling import gibbs_sample
from models import m_deep_bistable, m_wide_bistable
from util import load_or_run
from counting import steady_state

def KL(p1, p2):
	"""computes KL divergence(p1, p2) (where p1 and p2 are discrete
	probability vectors of the same length)"""
	return np.dot(p1, np.log(p1 / p2))

def test_deep_kl_lag(m, p=.99, trials=100, eps=1e-3, n_samples=300, pre_burnin=100):
	net = m_deep_bistable(m+1, p) # using m+1 since leaf node is used for evidence

	# compute steady state
	model_nodes = net._nodes[:-1]
	evidence_node = net._nodes[-1]
	ss_distributions = steady_state(net, {evidence_node : 1}, model_nodes, eps)

	# accumulate data (KL divergence) in nodes x time_steps x trials
	data = np.zeros((m, n_samples, trials))

	for t in range(trials):
		counts = {node: eps * np.ones(node.size()) for node in model_nodes}

		def do_divergence(i, net):
			for j,n in enumerate(model_nodes):
				counts[n][n.state_index()] += 1
				# compute divergence between current estimate and ss_distribution
				data[j][i][t] = KL(ss_distributions[n], counts[n] / counts[n].sum())

		# prepare net by burning-in to 0-evidence
		gibbs_sample(net, {evidence_node : 0}, None, 0, pre_burnin)

		# get data on 1-evidence (no burnin this time; demonstrating lag)
		gibbs_sample(net, {evidence_node : 1}, do_divergence, n_samples, 0)

	return data

def test_deep_likelihood_lag(m, p=.99, trials=100, eps=1e-3, n_samples=300, pre_burnin=100):
	net = m_deep_bistable(m+1, p)

	# compute steady state
	model_nodes = net._nodes[:-1]
	evidence_node = net._nodes[-1]
	ss_distributions = steady_state(net, {evidence_node : 1}, model_nodes, eps)

	# accumulate data (sample likelihood) in nodes x time_steps x trials
	data = np.zeros((m-1, n_samples, trials))

	for t in range(trials):

		def do_likelihood(i, net):
			for j,n in enumerate(model_nodes):
				data[j][i][t] = ss_distributions[n][n.state_index()]

		# prepare net by burning-in to 0-evidence
		gibbs_sample(net, {evidence_node : 0}, None, 0, pre_burnin)

		# get data on 1-evidence (no burnin this time; demonstrating lag)
		gibbs_sample(net, {evidence_node : 1}, do_likelihood, n_samples, 0)

	return data

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--recompute', dest='recompute', action='store_true', default=False)
	parser.add_argument('--samples', type=int, default=30)
	parser.add_argument('--prob', type=float, default=0.9)
	parser.add_argument('--trials', type=int, default=300)
	parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
	parser.add_argument('--m-max', dest='m_max', type=int, default=7)

	args = parser.parse_args()

	recompute = args.recompute
	samples = args.samples
	prob = args.prob
	trials = args.trials
	plot = args.plot
	m_max = args.m_max

	# KL Divergence Test
	for m in range(2,m_max+1):
		print m

		filename = 'test_deep_kl_lag[%d].npy' % m
		data = load_or_run(filename, lambda: test_deep_kl_lag(m, trials=trials, n_samples=samples, p=prob), force_recompute=recompute)

		if plot:
			# get mean over trials
			mean = np.mean(data, axis=2)
			variance = np.var(data, axis=2)
	
			fig = plt.figure()
			ax = fig.add_subplot(111)
			# plot in reverse since data[0] is 'top' layer
			for layer in range(data.shape[0]+1, 1, -1):
				plt.errorbar(range(samples), mean[layer-2,:].transpose(), variance[layer-2,:].transpose())
			plt.legend(['layer %d' % l for l in range(2,m+1)])
			plt.savefig('results_KL_m=%d.png' % m)
			plt.close()

	# Likelihood Test
	for m in range(2,m_max+1):
		print m

		filename = 'test_deep_likelihood_lag[%d].npy' % m
		data = load_or_run(filename, lambda: test_deep_likelihood_lag(m, trials=trials, n_samples=samples, p=prob), force_recompute=recompute)

		# first, normalize each trajectory of likelihoods to fill the [0,1] range
		# (if Likelihood of samples varies between {0.3,0.7} for one node and {0.1,0.9} for another,
		# 	it is difficult to compare convergence rates.)
		# (todo: this should gotten from ss_distributions directly rather than extracted as min/max)
		min_likelihood = np.zeros(data.shape)
		max_likelihood = np.zeros(data.shape)
		for layer in range(data.shape[0]):
			min_likelihood[layer,:] = data[layer].min()
			max_likelihood[layer,:] = data[layer].max()

		if plot:
			data = (data - min_likelihood) / (max_likelihood - min_likelihood)
	
			# get mean over trials
			mean = np.mean(data, axis=2)
			variance = np.var(data, axis=2)
	
			fig = plt.figure()
			ax = fig.add_subplot(111)
			# plot in reverse since data[0] is 'top' layer
			for layer in range(data.shape[0]+1, 1, -1):
				plt.errorbar(range(samples), mean[layer-2,:].transpose(), variance[layer-2,:].transpose())
			plt.legend(['layer %d' % l for l in range(2,m+1)])
			plt.savefig('results_likelihood_m=%d.png' % m)
			plt.close()

