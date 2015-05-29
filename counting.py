import numpy as np
import itertools, operator
from sampling import gibbs_sample
from collections import defaultdict
from scipy.misc import factorial as fact

def count_states(net):
	return np.prod([n.size() for n in net._nodes])

def state_to_id(net, state_vector=None):
	"""returns an integer id [0..nstates-1] that uniquely identifies the given state in the given net
	"""
	# use net's current state if none given
	if state_vector is None:
		state_vector = net.state_vector()
	_id = 0
	last_size = 0
	for i,n in enumerate(net.iter_nodes()):
		_id *= last_size
		_i = n._states.index(state_vector[i])
		_id += _i
		last_size = n.size()
	return _id

def id_to_state(net, _id):
	"""inverse of state_to_id
	"""
	for i,n in enumerate(net.iter_nodes_reversed()):
		n.set_value_by_index(_id % n.size())
		_id = int(_id / n.size())

	return net.state_vector()

def id_subset(net, where, value=True):
	"""returns list of state ids that satisfy the condition `where(net)==value`
	"""
	tmp = net.state_map()

	n_ids = count_states(net)
	ids = [0] * n_ids

	# brute force
	# TODO make more efficient
	m = 0
	for _id in xrange(n_ids):
		id_to_state(net, _id)
		if where(net) == value:
			ids[m] = _id
			m += 1

	net.evidence(tmp)
	return ids[:m]

def transition_probability(net, state1, state2, sample_order, conditioned_on={}):
	tmp = net.state_map()
	p = 1.0

	net.evidence(dict(zip(net._nodes, state1)))
	net.evidence(conditioned_on)

	for i,n in sample_order:
		if n in conditioned_on:
			if conditioned_on[n] != state2[i]:
				return 0. # impossible to transition to a state inconsistent with evidence
			else:
				pass # implicitly multiplying p by 1
		else:
			node_marginal = net.markov_blanket_marginal(n)
			p *= node_marginal[n._states.index(state2[i])]
			n.set_value(state2[i])
	net.evidence(tmp)
	return p

def construct_markov_transition_matrix(net, conditioned_on={}):
	tmp = net.state_map()
	n_states = count_states(net)

	if n_states > 128:
		ans = raw_input("really compute full %dx%d transition matrix over %d permutations? [y/N]  " % (n_states, n_states, fact(len(net._nodes))))
		if ans[0] not in "yY":
			return

	A = np.zeros((n_states, n_states))

	orderings = itertools.permutations(enumerate(net.iter_nodes()))
	u = 1.0 / fact(len(net._nodes))
	for ns in orderings:
		for i,j in itertools.product(range(n_states), range(n_states)):
			A[i][j] += u * transition_probability(net, id_to_state(net, j), id_to_state(net, i), ns, conditioned_on)

	net.evidence(tmp)
	return A

def steady_state(net, evidence, nodes, eps=0, M=10000, burnin=100):
	"""computes steady state distribution for each node
	"""
	# eps allows for some small count at each state (to avoid zero-probability states)
	counts = {node: eps * np.ones(node.size()) for node in nodes}
	
	def do_count(i, net):
		for n in nodes:
			counts[n][n.state_index()] += 1

	gibbs_sample(net, evidence, do_count, M, burnin)

	for _, c in counts.iteritems():
		c /= c.sum()

	return counts

def analytic_marginal_states(net, conditioned_on={}):
	N = count_states(net)
	S = np.zeros(N)

	for i in range(N):
		id_to_state(net, i)
		S[i]  = net.probability(conditioned_on)
	S = S / S.sum()
	return S

def sample_marginal_states(net, evidence, samples, when=None):
	"""Computes S[i] = vector of marginal probabilities that net is in state id i.

	If given, when(net) is evaluated to decide whether each sample is included
	"""
	n_states = count_states(net)

	# estimate starting distribution over states by sampling
	S = np.zeros(n_states)

	def do_count_state(i, net):
		if when is None or when(net):
			S[state_to_id(net, net.state_vector())] += 1

	gibbs_sample(net, evidence, do_count_state, samples, 1)

	return S / S.sum()

def plurality_state(net):
	counts = defaultdict(lambda: 0)
	for n in net.iter_nodes():
		counts[n.get_value()] += 1
	return max(counts.iteritems(), key=operator.itemgetter(1))[0]

def mean_state(net, S):
	"""given a network and distribution over states, return a map of node:mean state (index)
	"""
	means = [0.0 for n in net.iter_nodes()]
	for i,p in enumerate(S):
		id_to_state(net, i)
		for i,n in enumerate(net.iter_nodes()):
			means[i] += p * n.state_index()
	return means

def node_marginal(net, S, node):
	"""Return the marginal distribution for the given node (lenth node.size()) according to the state distribution S
	"""
	tmp = net.state_map()
	marg = np.zeros(node.size())
	for i in xrange(len(S)):
		id_to_state(net, i)
		marg[node.state_index()] += S[i]
	net.evidence(tmp)
	return marg

def variational_distance(P1, P2):
	return 0.5 * np.abs(P1 - P2).sum()

def mixing_time(start, target, transition, eps=0.05, max_t=1000, converging_to=None):
	# Using markov transition matrix A, loop until `start` converges to within `eps` of `target`
	if converging_to is not None:
		if variational_distance(target, converging_to) > eps:
			return max_t, [] # it will never get there
	S = start
	i = 0
	vds = np.zeros(max_t+1)
	d = variational_distance(target, S)
	vds[0] = d
	while d >= eps:
		i = i+1
		S = np.dot(transition, S)
		d = variational_distance(target, S)
		vds[i] = d
		if i == max_t:
			break
	return i, vds
