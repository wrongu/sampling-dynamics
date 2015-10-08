import numpy as np
import itertools, operator
from sampling import gibbs_sample
from collections import defaultdict
from scipy.misc import factorial as fact
from util import normalized

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

def id_to_state(net, _id, conditioned_on={}):
	"""inverse of state_to_id
	"""
	for i,n in enumerate(net.iter_nodes_reversed()):
		n.set_value_by_index(_id % n.size())
		_id = int(_id / n.size())
	
	net.evidence(conditioned_on)

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

def transition_probability(net, state1, state2, sample_order, conditioned_on={}, fatigue_tau=None, feedforward_boost=None):
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
			node_marginal = net.markov_blanket_marginal(n, fatigue_tau=fatigue_tau, feedforward=feedforward_boost)
			p *= node_marginal[n._states.index(state2[i])]
			n.set_value(state2[i])
	net.evidence(tmp)
	return p

def construct_markov_transition_matrix(net, conditioned_on={}, method='permutations', fatigue_tau=None, feedforward_boost=None):
	tmp = net.state_map()
	n_states = count_states(net)

	if n_states > 512:
		ans = raw_input("really compute full %dx%d transition matrix over %d permutations? [y/N]  " % (n_states, n_states, fact(len(net._nodes))))
		if ans[0] not in "yY":
			return

	sampleable_nodes = [n for n in net.iter_nodes() if n not in conditioned_on]

	A = np.zeros((n_states, n_states))

	if method == 'permutations':
		orderings = itertools.permutations(enumerate(sampleable_nodes))
		u = 1.0 / fact(len(net._nodes))
		for ns in orderings:
			for i,j in itertools.product(range(n_states), range(n_states)):
				A[j][i] += u * transition_probability(net, id_to_state(net, i), id_to_state(net, j), ns, conditioned_on, fatigue_tau, feedforward_boost)

	elif method == 'glauber':
		# enumerate starting states
		for i in xrange(n_states):
			s_start = id_to_state(net, i, conditioned_on) # puts net in state i *then sets evidence to conditioned_on*
			# enumerate single nodes, compute p(change to val) for each value the nodes can take on
			for (node_idx, change_node) in enumerate(sampleable_nodes):
				# s_end begins as a copy of s_start
				s_end = [val for val in s_start]
				# now enumerate values change_node could take on, and alter s_end[node_idx] for each one
				# (note that this will include s_end=s_start and fill out the diagonal of A)
				for val in change_node._states:
					s_end[node_idx] = val
					j = state_to_id(net, s_end)
					# get transition probability *where only change_node is sampled* (i.e. glauber dynamics)
					A[j][i] += transition_probability(net, s_start, s_end, [(node_idx, change_node)], conditioned_on, fatigue_tau, feedforward_boost)

	# normalize columns (every state must go somewhere)
	for i in xrange(n_states):
		if A[:,i].sum() > 0:
			A[:,i] /= A[:,i].sum()

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

def eig_steadystate(A):
	# steady state distribution of A_ff transition matrix is largest eigenvector (eigenvalue=1)
	w,v = np.linalg.eig(A)
	inds = np.argsort(w)
	S_steady_state = np.abs(v[:,inds[-1]])
	return normalized(S_steady_state, order=1)
	
def flip_distribution_binary_nodes(net, S):
	"""if S is a distribution over states {a,b}^net.size(),
	this returns the inverted distribution with all a and b switched
	"""

	n_states = len(S)
	S_flip = np.zeros(S.shape)

	for i in xrange(n_states):
		state_vec = id_to_state(net, i)
		# flip 1->0 and vice versa
		state_vec_inv = [nd._states[1-nd._states.index(st)] for nd,st in zip(net.nodes(), state_vec)]
		j = state_to_id(net, state_vec_inv)

		S_flip[j] = S[i]
	return S_flip

def analytic_marginal_states(net, conditioned_on={}):
	N = count_states(net)
	S = np.zeros(N)

	for i in range(N):
		id_to_state(net, i)
		S[i]  = net.probability(conditioned_on)
	return normalized(S, order=1)

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

	return normalized(S, order=1)

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
		S = S / S.sum()
		d = variational_distance(target, S)
		vds[i] = d
		if i == max_t:
			break
	return i, vds

def coupling_from_past(net, conditioned_on={}, iterations=100):
	"""Perform 'iterations' samples of mixing time using the Coupling From the Past method,
	returning a numpy vector of all the sampled values

	An estimate of mixing time is achieved when a preset sequence of MCMC steps converges
	to a constant function, but we can only "add" to our list of steps backwards in time.
	The sequence of steps can be thought of as the composition of a series of deterministic
	functions, each of which was randomly chosen.

	"update(i,r)" is the deterministic function that udates node i based on the random value r.
	"compose(i_list, r_list)" essentially implements MCMC by applying update() over and over.

	Operations are done in-place on the given net, making the code look strange.
	"""
	tmp = net.state_map()
	net.evidence(conditioned_on)

	sampleable_nodes = [n for n in net.iter_nodes() if n not in conditioned_on]

	def update(i, r):
		node = sampleable_nodes[i]
		marg = net.markov_blanket_marginal(node)
		# new value chosen by inverting the CDF
		# [any distribution can be sampled from by taking CDF^-1(value in [0,1])]
		# r is our (deterministic) choice for that value. this loop finds
		# where the CDF for node i goes above r
		cdf = 0
		for idx,prob in enumerate(marg):
			cdf += prob
			if r < cdf:
				break
		node.set_value_by_index(idx)

	def min_net():
		"""set net to 'minimum' value
		"""
		for n in sampleable_nodes:
			n.set_value_by_index(0)

	def max_net():
		"""set net to 'minimum' value
		"""
		for n in sampleable_nodes:
			n.set_value_by_index(n.size()-1)

	def compose(i_list, r_list):
		for i,r in itertools.izip(i_list, r_list):
			update(i,r)

	def composition_is_constant(i_list, r_list):
		# see where we get to from the 'min' net
		min_net(); compose(i_list, r_list)
		result_from_min = net.state_vector()
		# see where we get to from the 'max' net
		max_net(); compose(i_list, r_list)
		result_from_max = net.state_vector()
		# composition is constant iff these two extreme cases are equal
		# (having assumed that the 'net' has monotone coupling)
		return result_from_min == result_from_max

	T_samples = np.zeros(iterations)
	N = len(sampleable_nodes)
	for itr in xrange(iterations):
		ii = [np.random.randint(N)]
		rr = [np.random.rand()]
		# exponentially increase 'upper bound' until composition 'g' is constant.
		while not composition_is_constant(ii, rr):
			# double length of ii and rr, *prepending*
			ii = [np.random.randint(N) for _ in xrange(len(ii))] + ii
			rr = [np.random.rand() for _ in xrange(len(rr))] + rr
		L = len(ii)

		# narrow with binary search to get T exactly. IE there is a point where
		# things stop being constant (here, T = the number of '_' = 9):
		#
		#     [C C C C C C C _ _ _ _ _ _ _ _ _] 
		#      ^             ^
		#     max           min
		#
		# We also know we're looking in the first half of the array, since on the 2nd to last
		# iteration of the above while loop things *werent* constant but on the last they were.
		#
		# As T increases, we start at smaller and smaller indexes in ii and rr, which is why
		# 	idx = L - test
		# looks backwards.

		min_T, max_T = L/2, L
		while min_T < max_T:
			test = (min_T + max_T) / 2
			# we will use ii[idx:] and rr[idx:] such that len(ii[idx:])==test
			idx = L - test
			# if constant up to 'test', then T must have been even sooner than test.
			if composition_is_constant(ii[idx:], rr[idx:]):
				max_T = test
			# otherwise, still not constant at test, so T must have been bigger
			else:
				min_T = test+1
		# min_T and max_T have converged - we now have a value for T!
		T_samples[itr] = min_T

	net.evidence(tmp)
	return T_samples