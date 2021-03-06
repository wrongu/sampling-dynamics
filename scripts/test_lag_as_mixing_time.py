if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
import argparse
from models import m_deep_bistable
from visualize import plot_net_layerwise
from util import load_or_run
from counting import *

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', dest='recompute', action='store_true', default=False)
parser.add_argument('--compare-p', dest='cmp_p', action='store_true', default=False)
parser.add_argument('--marg', dest='marg', type=float, default=0.9)
parser.add_argument('--eps', dest='eps', type=float, default=0.05)
parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
parser.add_argument('--m-max', dest='m_max', type=int, default=7)

args = parser.parse_args()

max_t = 1000
m_min = 2
n_layers = args.m_max - m_min + 1
layers = range(m_min, args.m_max+1)

mixing_times = np.zeros(n_layers)
for M in layers:
	net = m_deep_bistable(M, marg=args.marg)
	ev = net.get_node_by_name('X1')
	p = ev.get_table()[0,0]
	A = load_or_run('transition_matrix_M%d_p%.3f_ev1' % (M, p),
		lambda: construct_markov_transition_matrix(net, conditioned_on={ev:1}),
		force_recompute=args.recompute)

	# S_start and S_target are marginal distributions conditioned on {ev:0} and {ev:1} respectively.
	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_target = analytic_marginal_states(net, conditioned_on={ev: 1})

	mixing_times[M-m_min], _ = mixing_time(S_start, S_target, A, eps=args.eps)

if args.cmp_p:
	p = 0.96
	mixing_times_rho_const = np.zeros(n_layers)
	for M in layers:
		net = m_deep_bistable(M, p=0.96)
		ev = net.get_node_by_name('X1')
		A = load_or_run('transition_matrix_M%d_p%.3f_ev1' % (M, p),
			lambda: construct_markov_transition_matrix(net, conditioned_on={ev:1}),
			force_recompute=args.recompute)

		# S_start and S_target are marginal distributions conditioned on {ev:0} and {ev:1} respectively.
		S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
		S_target = analytic_marginal_states(net, conditioned_on={ev: 1})

		mixing_times_rho_const[M-m_min], _ = mixing_time(S_start, S_target, A, eps=args.eps)

if args.plot:
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.plot(layers, mixing_times, '-bo')
	if args.cmp_p:
		plt.plot(layers, mixing_times_rho_const, '--b^')
		plt.legend(['P constant', 'rho constant'], loc='lower right')
	ax.set_xlim([0,args.m_max+1])
	ax.set_ylim([0,40])
	plt.xlabel('model depth')
	plt.ylabel('mixing time')
	plt.savefig('plots/mixing_time.png')
	plt.close()

# Part 2: mixing time "by layer" to show that early layers are slowed down by the presence of deep layers
mixing_time_by_layer = np.zeros((n_layers, n_layers)) # mtbl[i,j] = time of node i when net has j total
for M in layers:
	net = m_deep_bistable(M, marg=args.marg)
	ev = net.get_node_by_name('X1')
	p = ev.get_table()[0,0]
	A = load_or_run('transition_matrix_M%d_p%.3f_ev1' % (M, p), lambda: construct_markov_transition_matrix(net, conditioned_on={ev:1}))

	# S_start and S_target are marginal distributions conditioned on {ev:0} and {ev:1} respectively.
	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_target = analytic_marginal_states(net, conditioned_on={ev: 1})

	for layer in range(m_min, M+1):
		node = net.get_node_by_name('X%d' % (layer))
		node_marginal_start  = node_marginal(net, S_start,  node)
		node_marginal_target = node_marginal(net, S_target, node)

		# run mixing time algorithm 
		S = S_start.copy()
		d = variational_distance(node_marginal_start, node_marginal_target)
		i = 0
		while d > args.eps and i < max_t-1:
			i += 1
			S = np.dot(A, S)
			marg = node_marginal(net, S, node)
			d = variational_distance(marg, node_marginal_target)

		mixing_time_by_layer[M-m_min,layer-m_min] = i

if args.cmp_p:
	p = 0.96
	mixing_times_by_layer_rho_const = np.zeros((n_layers, n_layers))
	for M in layers:
		net = m_deep_bistable(M, p)
		ev = net.get_node_by_name('X1')
		A = load_or_run('transition_matrix_M%d_p%.3f_ev1' % (M, p), lambda: construct_markov_transition_matrix(net, conditioned_on={ev:1}))

		# S_start and S_target are marginal distributions conditioned on {ev:0} and {ev:1} respectively.
		S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
		S_target = analytic_marginal_states(net, conditioned_on={ev: 1})

		for layer in range(m_min, M+1):
			node = net.get_node_by_name('X%d' % (layer))
			node_marginal_start  = node_marginal(net, S_start,  node)
			node_marginal_target = node_marginal(net, S_target, node)

			# run mixing time algorithm 
			S = S_start.copy()
			d = variational_distance(node_marginal_start, node_marginal_target)
			i = 0
			while d > args.eps and i < max_t-1:
				i += 1
				S = np.dot(A, S)
				marg = node_marginal(net, S, node)
				d = variational_distance(marg, node_marginal_target)

			mixing_times_by_layer_rho_const[M-m_min,layer-m_min] = i

if args.plot:
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	for M in layers:
		ax.plot(np.arange(M,args.m_max+1), mixing_time_by_layer[M-2:n_layers,M-2],'-o',label='$x_%d$' % M)
	if args.cmp_p:
		ax.set_color_cycle(None)
		for M in layers:
			ax.plot(np.arange(M,args.m_max+1), mixing_times_by_layer_rho_const[M-2:n_layers,M-2],'--^')

	ax.set_xlim([0,args.m_max+1])
	ax.set_ylim([0,40])
	plt.xlabel('model depth')
	plt.ylabel('mixing time')
	plt.legend(loc='upper left')
	plt.savefig('plots/mixing_time_by_layer.png')
	plt.close()