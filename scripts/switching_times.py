if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
from graphical_models import DiscreteVariable, BayesNet
from collections import defaultdict
from models import k_deep_bistable, k_wide_bistable
from sampling import *
from counting import *

def analytic_switching_times(net, init_distribution, target, max_t=None, eps=1e-4):
	"""analytically compute switching time distribution for the given net to transition from
	the distribution of initial states 'init_distribution' to the target state.

	The algorithm terminates when P(T > t) < eps (that is, when the fraction
	(1-eps) of times have been accounted for) OR at max_t, if it is defined. Whichever comes
	first.

	Warning: this algorithm computes the transition table for the full joint distribution, and
	should not be used for large networks

	Disclaimer: although 'analytic', this algorithm is only as accurate as init_distribution
	"""

	n_states = count_states(net)

	P = construct_markov_transition_matrix(net)

	S = init_distribution

	# iteratively transition S by P, counting at each step how much
	# probability mass moved into target and clearing those
	# values (so flipping back and forth across threshold isn't counted
	# multiple times)

	target_ids = id_subset(net, target)

	switching_time_distribution = []

	while S.sum() > eps:
		transitioned_mass = S[target_ids].sum()
		switching_time_distribution.append(transitioned_mass)
		S[target_ids] = 0
		S = np.dot(P, S)

		if max_t is not None and len(switching_time_distribution) >= max_t:
			break

	full_distribution = np.array(switching_time_distribution)
	return full_distribution / full_distribution.sum()

def sampled_switching_times(net, target, max_t=10000, burnin=50, trials=10000):
	# map from time to count (will be converted to an array later)
	times = defaultdict(lambda: 0)

	# gibbs sampler callback (counts switch if it happened and breaks from sampling loop)
	def do_check_switch(i, net):
		switched = True
		for n,v in target.iteritems():
			if n.get_value() != v:
				switched = False
				break
		if switched:
			times[i] = times[i] + 1
		# returning True halts the remaining samples
		return switched

	def is_init_state(i,net):
		for n,v in target.iteritems():
			if n.get_value() != v:
				return True

	# initialize net to a reasonable state
	gibbs_sample(net, {}, None, 0, burnin)

	for t in range(trials):
		# run until in an initialization state (target not true)
		# NOTE by doing this, we get a better comparison with 'analytic' using 'sample_recently_switched_states' for initialization
		gibbs_sample(net, {}, is_init_state, max_t, 0)

		# run sampler until switched (no evidence)
		slow_gibbs_sample(net, {}, do_check_switch, max_t, 0)

	# convert times to a distribution
	max_t = max(times.keys())

	times_array = np.zeros(max_t+1)
	for k,v in times.iteritems():
		times_array[k] = v
	return times_array / times_array.sum()

def sampled_switching_times_plurality(net, plurality_target, max_t=10000, burnin=500, trials=10000):
	# map from time to count (will be converted to an array later)
	times = defaultdict(lambda: 0)

	# gibbs sampler callback (counts switch if it happened and breaks from sampling loop)
	def do_check_switch(i, net):
		switched = plurality_state(net) == plurality_target
		if switched:
			times[i] = times[i] + 1
		# returning True halts the remaining samples
		return switched

	def is_init_state(i,net):
		return plurality_state(net) != plurality_target

	# initialize net to a reasonable state
	gibbs_sample(net, {}, None, 0, burnin)

	for t in range(trials):
		# run until in an initialization state (target not true)
		# NOTE by doing this, we get a better comparison with 'analytic' using 'sample_recently_switched_states' for initialization
		gibbs_sample(net, {}, is_init_state, max_t, 0)

		# run sampler until switched (no evidence)
		slow_gibbs_sample(net, {}, do_check_switch, max_t, 0)

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

if __name__ == '__main__':
	
	def do_analytic_switching_time_plots(fig, p=0.9, k_min=2, k_max=7, model_builder=k_deep_bistable, ls='-', legend=True):
		distributions = [None] * (k_max-k_min+1)
		ax = fig.add_subplot(111)
		ax.set_color_cycle(None) # reset colors so dashed lines are same

		for k in range(k_min, k_max+1):
			print k
			net = model_builder(k, p)
			nodes = net._nodes
			S_start = sample_marginal_states(net, {nodes[0]:0}, 10000)
			distributions[k-k_min] = analytic_switching_times(net, S_start, {nodes[0]: 1})

		for d in distributions:
			ax.plot(d[1:], linestyle=ls)
		if legend:
			ax.legend(['k = %d' % k for k in range(k_min, k_max+1)])

	def compare_empirical_analytic_switching_times(p=0.96, k_min=2, k_max=7, model_builder=k_deep_bistable):
		for k in range(k_min, k_max+1):
			print k
			net = model_builder(k, p)
			nodes = net._nodes
			print '-init-'
			S_start = sample_marginal_states(net, {nodes[0]:0}, 10000)
			print '-sample-'
			empirical = sampled_switching_times(net,  S_start, {nodes[0]: 1}, trials=5000)
			print '-analytic-'
			analytic  = analytic_switching_times(net, S_start, {nodes[0]: 1}, max_t=len(empirical))
			
			plt.figure()
			plt.plot(analytic)
			plt.plot(empirical)
			plt.legend(['analytic', 'sample-approx'])
			plt.savefig('test_st_%d.png' % k)
			plt.close()

	# fig = plt.figure()
	# print '-deep-'
	# do_analytic_switching_time_plots(fig, k_min=3, k_max=6, p=0.96)
	# scale = plt.axis()
	# print '-wide-'
	# do_analytic_switching_time_plots(fig, k_min=1, k_max=4, model_builder=k_wide_bistable, p=0.96, ls='--', legend=False)
	# plt.axis(scale)
	# plt.savefig('deep_vs_wide_analytic_switching_times.png')

	# compare_empirical_analytic_switching_times(k_min=2, k_max=5)

	# Tests with PLURALITY definition of switching
	# net = k_deep_bistable(5, 0.96)
	# S_start = sample_marginal_states(net, {}, 10000, conditional_fn=lambda net: plurality_state(net) == 0)
	# import pdb; pdb.set_trace()
	# distrib = sampled_switching_times_plurality(net, S_start, 1, trials=2000)
	
	# plt.figure()
	# plt.plot(distrib)
	# plt.savefig('plurality1.png')
	# plt.close()
	pass