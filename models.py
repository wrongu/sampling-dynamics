import numpy as np
import random
import itertools
from graphs import Graph, DiGraph
from graphical_models import BayesNet, DiscreteVariable
from util import normalized

def compute_p_for_given_marginal(m, marg):
	"""In some cases where m_deep_bistable is used, we want P(Xm|X1) to be the same
	across values of m. This function takes in m and P(Xm|X1) (called 'marg') and returns p

	The reason to hold 'marg' rather than 'p' constant is that if p is constant and m grows,
	the "vanishing correlation" effect dominates, and P(Xm|X1) approaches 50/50, making most
	comparisons meaningless
	"""

	# find p such that the recursively defined polynomial f(p,m) = marg
	# f(0) = 1
	# f(p,m) = p*f(p,m-1) + (1-p)*(1-f(p,m-1))
	# (intuitively, marginal that layer M = 0 is P(layer m-1=0)P(0->0) + P(layer m-1=1)P(1->0))

	polynomial = np.zeros(m+1)
	polynomial[-1] = 1

	for i in range(m):
		# rearraning f(p,m) = p*f(p,m-1) + (1-p)*(1-f(p,m-1)),
		# we get f(p,m) = 2*p*f(p,m-1) - f(p,m-1) - p + 1
		new_polynomial = np.zeros(m+1)
		new_polynomial[:-1] = 2 * polynomial[1:] # 2*p*f term
		new_polynomial -= polynomial # - f term
		new_polynomial[-2] -= 1 # -p term
		new_polynomial[-1] += 1 # +1 term
		polynomial = new_polynomial

	polynomial[-1] -= marg # f(p,m) - marg = 0
	roots = np.roots(polynomial)
	return np.real(roots[0])

def compute_marginal_for_given_p(m, p):
	marg = 1.
	for _ in range(m):
		marg = p*marg + (1-p)*(1-marg)
	return marg

def m_deep_with_shortcut(m, p=None, marg=None, fro=None, to=None, cpt='marginal'):
	"""constructs an m_deep_bistable model with a single additional connection from node 'fro' to node 'to'

	(fro must be greater than to to prevent cycles)
	"""
	net = m_deep_bistable(m, p, marg)
	if type(fro) is int:
		fro = 'X%d' % fro
	if type(to) is int:
		to = 'X%d' % to
	fro = net.get_node_by_name(fro)
	to = net.get_node_by_name(to)

	if cpt is 'random':
		cpt = np.random.random((2,2))
		sums = cpt.sum(axis=0)
		cpt[:,0] /= sums[0]
		cpt[:,1] /= sums[1]
	elif cpt is 'marginal':
		if p is None: p = net.get_node_by_name('X1').get_table()[0,0]
		dists = {}
		def store_dists(n,d): dists[n] = d
		net.bfs_traverse([fro], store_dists)
		marg = compute_marginal_for_given_p(dists[to], p)
		cpt = np.array([[marg, 1-marg], [1-marg, marg]])

	prev_table = to.get_table()
	prev_parents = net.parents(to)
	# making room for a new parent
	shape = tuple([fro.size()] + list(prev_table.shape))
	table = np.zeros(shape)
	# populate new table such that P(to|parents,fro) = P(to|fro)P(to|parents)
	for i in range(fro.size()):
		for j in range(to.size()):
			table[i,...,j] = cpt[i,j] * prev_table[...,j];

	# renormalize such that P(to|a particular configuration of parents) = 1
	table = normalized(table, axis=-1, order=1)
	
	net.cpt([fro] + prev_parents + [to], table)
	return net

def m_deep_bistable(m, p=None, marg=None):
	"""constructs a simple bayes net with binary variables in m layers,
	where each is a likely cause of the earlier (0 causes 0) with probability p

	if p is not given (but marg must be), then p is chosen such that the marginal P(Xm|X1) is marg
	"""

	if p is None:
		if marg is None:
			raise AssertionError("Either p OR marg must be specified")
		else:
			p = compute_p_for_given_marginal(m, marg)

	net = BayesNet()
	top_layer = DiscreteVariable([0,1], tbl=np.array([0.5, 0.5]), name="X%d"%m)
	net.add_node(top_layer)
	prev_layer = top_layer
	for i in range(m-1):
		layer = DiscreteVariable([0,1], name="X%d" % (m-i-1))
		table = np.array([[p, 1-p], [1-p, p]])

		net.add_node(layer)
		net.cpt([prev_layer, layer], table)
		prev_layer = layer

	net.verify()
	return net

def m_wide_bistable(m, p):
	"""constructs simple bayes net with binary variables and 3 layers, where
	the middle layer has m variables

	m-width is complementary to (m+1)-depth (counting the root node of this
		model separately)
	"""
	net = BayesNet()
	root = DiscreteVariable([0,1], tbl=np.array([0.5, 0.5]), name="root")
	net.add_node(root)
	middle_nodes = [None]*m
	for i in range(m):
		n = DiscreteVariable([0,1], name="X%d" % (i+1))
		table = np.array([[p, 1-p], [1-p, p]])

		middle_nodes[i] = n
		net.add_node(n)
		net.cpt([root, n], table)

	leaf = DiscreteVariable([0,1], name="Z")
	table = np.zeros(shape=[2]*(m+1))
	for idx in itertools.product(*[[0,1]]*m):
		# independent influences on leaf:
		# p(leaf=0) = p^(#X=0) (1-p)^(#X=1)
		ones = sum(idx)
		q = p**(m-ones) * (1-p)**(ones)
		table[idx] = [q, 1-q]
	net.add_node(leaf)
	net.cpt(middle_nodes + [leaf], table)

	net.verify()
	return net

def erdos_renyi(n, p):
	g = Graph()

	for i in xrange(n):
		g.add_node(DiscreteVariable([0,1], name=str(i)))

	for i,j in itertools.product(xrange(n), xrange(n)):
		if random.random() < p:
			g.add_edge(g.get_node_by_name(str(i)),g.get_node_by_name(str(j)))
	return g

if __name__ == '__main__':
	# test compute_p_for_given_marginal
	for marg in [.7,.8,.9]:
		print marg
		for m in range(2,8):
			print '  ', m, compute_p_for_given_marginal(m, marg)
