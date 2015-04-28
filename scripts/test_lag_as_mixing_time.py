if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
import argparse
from models import k_deep_bistable
from visualize import plot_net_layerwise
from util import load_or_run
from counting import *

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', dest='recompute', action='store_true', default=False)
parser.add_argument('--prob', dest='p', type=float, default=0.96)
parser.add_argument('--eps', dest='eps', type=float, default=0.05)
parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
parser.add_argument('--k-max', dest='k_max', type=int, default=7)

args = parser.parse_args()

max_t = 1000
k_min = 2
n_layers = args.k_max - k_min + 1
layers = range(k_min, args.k_max+1)

mixing_times = np.zeros(n_layers)
for K in layers:
	print K
	net = k_deep_bistable(K, args.p)
	ev = net.get_node_by_name('X1')
	P = load_or_run('transition_matrix_K%d_p%.3f' % (K, args.p), lambda: construct_markov_transition_matrix(net, conditioned_on={ev: 1}))

	# S_start and S_target are marginal distributions conditioned on {ev:0} and {ev:1} respectively.
	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_target = analytic_marginal_states(net, conditioned_on={ev: 1})

	mixing_times[K-k_min], _ = mixing_time(S_start, S_target, P, eps=args.eps)

if args.plot:
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.plot(layers, mixing_times, '-ko')
	ax.set_xlim([0,args.k_max+1])
	plt.xlabel('model depth')
	plt.ylabel('mixing time (samples to variational distance < eps)')
	plt.title('Mixing Time as a funciton of depth')
	plt.savefig('plots/mixing_time.png')
	plt.close()

# Part 2: mixing time "by layer" to show that early layers are slowed down by the presence of deep layers
mixing_time_by_layer = np.zeros((n_layers, n_layers)) # mtbl[i,j] = time of node i when net has j total
for K in layers:
	print K
	net = k_deep_bistable(K, args.p)
	ev = net.get_node_by_name('X1')
	P = load_or_run('transition_matrix_K%d_p%.3f' % (K, args.p), lambda: construct_markov_transition_matrix(net, conditioned_on={ev: 1}))

	# S_start and S_target are marginal distributions conditioned on {ev:0} and {ev:1} respectively.
	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_target = analytic_marginal_states(net, conditioned_on={ev: 1})

	for layer in range(k_min, K+1):
		node = net.get_node_by_name('X%d' % (layer))
		node_marginal_start  = node_marginal(net, S_start,  node)
		node_marginal_target = node_marginal(net, S_target, node)

		# run mixing time algorithm 
		S = S_start.copy()
		d = variational_distance(node_marginal_start, node_marginal_target)
		i = 0
		while d > args.eps and i < max_t-1:
			i += 1
			S = np.dot(P, S)
			marg = node_marginal(net, S, node)
			d = variational_distance(marg, node_marginal_target)

		mixing_time_by_layer[K-k_min,layer-k_min] = i

if args.plot:
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	for K in layers:
		ax.plot(np.arange(K,args.k_max+1), mixing_time_by_layer[K-2:n_layers,K-2],'-o')
	plt.title('Mixing Time (per layer) as a function of depth')
	plt.xlabel('model depth')
	plt.ylabel('mixing time (samples to variational distance < eps)')
	plt.legend(['layer %d' % K for K in layers], loc='upper left')
	plt.savefig('plots/mixing_time_by_layer.png')
	plt.close()