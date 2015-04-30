if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
from graphical_models import DiscreteVariable, BayesNet
from collections import defaultdict
from models import m_deep_bistable
from util import load_or_run
from sampling import *
from counting import *

def analytic_switching_times(net, init_distribution, target, transition=None, max_t=None, eps=1e-4):
	"""analytically compute switching time distribution for the given net to transition from
	the distribution of initial states 'init_distribution' to the target state.

	`target` must be a tuple of (function, value) such that function(net)=value defines when "switched"
	(this is more general that switching into some {node:value} configuration)

	The algorithm terminates when P(T > t) < eps (that is, when the fraction
	(1-eps) of times have been accounted for) OR at max_t, if it is defined. Whichever comes
	first.

	Warning: this algorithm computes the transition table for the full joint distribution, and
	should not be used for large networks

	Disclaimer: although 'analytic', this algorithm is only as accurate as init_distribution
	"""

	n_states = count_states(net)

	if transition is None:
		transition = construct_markov_transition_matrix(net)

	S = init_distribution

	# iteratively transition S by P, counting at each step how much
	# probability mass moved into target and clearing those
	# values (so flipping back and forth across threshold isn't counted
	# multiple times)

	target_ids = id_subset(net, *target)

	switching_time_distribution = []

	while S.sum() > eps:
		transitioned_mass = S[target_ids].sum()
		switching_time_distribution.append(transitioned_mass)
		S[target_ids] = 0
		S = np.dot(transition, S)

		if max_t is not None and len(switching_time_distribution) >= max_t:
			break

	return np.array(switching_time_distribution)

def sampled_switching_times(net, target, max_t=10000, burnin=50, trials=10000):
	# map from time to count (will be converted to an array later)
	times = defaultdict(lambda: 0)

	target_ids = id_subset(net, *target)

	# gibbs sampler callback (counts switch if it happened and breaks from sampling loop)
	def do_check_switch(i, net):
		switched = state_to_id(net) in target_ids
		if switched:
			times[i] = times[i] + 1
		# returning True halts the remaining samples
		return switched

	def is_init_state(i,net):
		return state_to_id(net) not in target_ids

	# initialize net to a reasonable state
	gibbs_sample(net, {}, None, 0, burnin)

	for t in range(trials):
		# run until in an initialization state (target not true)
		# NOTE by doing this, we get a better comparison with 'analytic' using 'sample_recently_switched_states' for initialization
		gibbs_sample(net, {}, is_init_state, max_t, 0)

		# run sampler until switched (no evidence)
		gibbs_sample(net, {}, do_check_switch, max_t, 0)

	# convert times to a distribution
	max_t = max(times.keys())

	times_array = np.zeros(max_t+1)
	for k,v in times.iteritems():
		times_array[k] = v
	return times_array / times_array.sum()

def sample_recently_switched_states(net, state_fn, max_iterations=50000):
	"""estimate distribution of states immediately following a transition from state_fn(net)=1 to state_fn(net)=0

	i.e. this estimates the distribution of states where the 'clock starts' for computing switching times
	"""
	n_states = count_states(net)

	S = np.zeros(n_states)

	class state_tracking_counter(object):
		def __init__(self):
			self.last_state = -1

		def __call__(self, i, net):
			current_state = state_fn(net)
			if self.last_state == 1 and current_state == 0:
				S[state_to_id(net, net.state_vector())] += 1
			self.last_state = current_state

	gibbs_sample(net, {}, state_tracking_counter(), max_iterations, 1)

	print "sample_recently_switched_states got %d examples" % S.sum()

	return S / S.sum()

def analytic_recently_switched_states(net, state_fn, state_at_t0, P=None):
	N = count_states(net)
	if P is None: P = construct_markov_transition_matrix(net)

	# find which state ids correspond to the 'initial' or 'time zero' state
	state0_ids = id_subset(net, state_fn, state_at_t0)

	S = analytic_marginal_states(net)

	# distribution over just-switched states (from S to S') is proportional to
	# 	P(S=j)*P(S'=i|S=j)
	# i.e. transition probability from marginal to any of state0_ids *but not from any state0_ids*
	S_recently_switched = np.zeros(N)
	S[state0_ids] = 0.
	S_recently_switched[state0_ids] = np.dot(P, S)[state0_ids]
	return S_recently_switched / S_recently_switched.sum()


def top_node_percept(net):
	return net._nodes[0].get_value()

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--recompute', dest='recompute', action='store_true', default=False)
	parser.add_argument('--marg', dest='marg', type=float, default=0.9)
	parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
	parser.add_argument('--k-max', dest='k_max', type=int, default=7)
	parser.add_argument('--samples', dest='samples', type=int, default=5000)
	parser.add_argument('--T', dest='max_t', type=int, default=10000)
	args = parser.parse_args()

	# Make plots that verify 'analytic' switching time algorithm (compare with sampling)
	for k in range(2, args.k_max+1):
		print k
		net = m_deep_bistable(k, marg=args.marg)
		p = net.get_node_by_name('X1').get_table()[0,0]
		nodes = net._nodes
		print '-init-'
		P = load_or_run('transition_matrix_K%d_p%.3f_noev' % (k, p), lambda: construct_markov_transition_matrix(net), force_recompute=args.recompute)
		S_start = analytic_recently_switched_states(net, top_node_percept, 0, P)
		print '-sample-'
		empirical = load_or_run('sampled_switching_times_K%d_p%.3f' % (k, p), lambda: sampled_switching_times(net, (top_node_percept, 1), trials=args.samples), force_recompute=args.recompute)
		print '-analytic-'
		analytic  = analytic_switching_times(net, S_start, (top_node_percept, 1), transition=P, max_t=len(empirical))
		
		if args.plot:
			plt.figure()
			plt.plot(analytic)
			plt.plot(empirical)
			plt.legend(['analytic', 'sample-approx'])
			plt.savefig('plots/cmp_empirical_analytic_st_K%d.png' % k)
			plt.close()

	# Make plots of switching times (with percept defined as top node state)
	switching_time_distributions = np.zeros((args.max_t, args.k_max-1))
	actual_max_t = 0
	for k in range(2, args.k_max+1):
		print k
		net = m_deep_bistable(k, marg=args.marg)
		p = net.get_node_by_name('X1').get_table()[0,0]
		print '-transition-'
		P = load_or_run('transition_matrix_K%d_p%.3f_noev' % (k, p), lambda: construct_markov_transition_matrix(net), force_recompute=args.recompute)
		print '-init-'
		S_init = analytic_recently_switched_states(net, top_node_percept, 0, P)
		print '-analytic st-'
		SW_distrib = analytic_switching_times(net, S_init, (top_node_percept, 1), transition=P, max_t=args.max_t)
		actual_max_t = max(actual_max_t, len(SW_distrib))
		switching_time_distributions[:len(SW_distrib),k-2] = SW_distrib
	if args.plot:
		plt.figure()
		bar_width = 1. / (args.k_max-1)
		colors = 'brgcmyk'
		for i in range(2,args.k_max+1):
			plt.bar(i*bar_width+np.arange(1,actual_max_t+1), switching_time_distributions[:actual_max_t,i-2],width=bar_width,color=colors[i-2])
		plt.title('Analytic ST for various M')
		plt.legend(['M = %d' % k for k in range(2, args.k_max+1)])
		plt.xlabel('samples')
		plt.ylabel('P(switch at t)')
		plt.xlim([0,20])
		plt.savefig('plots/analytic_ST_top.png')
		plt.close()

	# Make plots of switching times (with percept defined as majority state)
	switching_time_distributions = np.zeros((args.max_t, args.k_max-1))
	actual_max_t = 0
	for k in range(2, args.k_max+1):
		print k
		net = m_deep_bistable(k, marg=args.marg)
		p = net.get_node_by_name('X1').get_table()[0,0]
		print '-transition-'
		P = load_or_run('transition_matrix_K%d_p%.3f_noev' % (k, p), lambda: construct_markov_transition_matrix(net), force_recompute=args.recompute)
		print '-init-'
		S_init = analytic_recently_switched_states(net, plurality_state, 0, P)
		print '-analytic st-'
		SW_distrib = analytic_switching_times(net, S_init, (plurality_state, 1), transition=P, max_t=args.max_t)
		actual_max_t = max(actual_max_t, len(SW_distrib))
		switching_time_distributions[:len(SW_distrib),k-2] = SW_distrib
	if args.plot:
		plt.figure()
		bar_width = 1. / (args.k_max-1)
		colors = 'brgcmyk'
		for i in range(2,args.k_max+1):
			plt.bar(i*bar_width+np.arange(1,actual_max_t+1), switching_time_distributions[:actual_max_t,i-2],width=bar_width,color=colors[i-2])
		plt.title('Analytic ST for various M (majority percept)')
		plt.legend(['M = %d' % k for k in range(2, args.k_max+1)])
		plt.xlabel('samples')
		plt.ylabel('P(switch at t)')
		plt.xlim([0,20])
		plt.savefig('plots/analytic_ST_majority.png')
		plt.close()
	
