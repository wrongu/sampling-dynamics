import numpy as np
import random
import itertools
from graphs import Graph, DiGraph
from graphical_models import BayesNet, DiscreteVariable

def m_deep_bistable(m, p=None, marg=None):
	"""constructs a simple bayes net with binary variables in m layers,
	where each is a likely cause of the earlier (0 causes 0) with probability p

	if p is not given (but marg must be), then p is chosen such that the marginal P(Xm|X1) is marg
	"""

	if p is None:
		if marg is None:
			print "Either p OR marg must be specified"
			import sys; sys.exit(1)
		else:
			# TODO compute rahter than lookup
			lookup = {
				.7 : {
					2: 0.816228,
					3: 0.868403,
					4: 0.897635,
					5: 0.916277,
					6: 0.929187,
					7: 0.938653
				}, .8 : {
					2: 0.887298,
					3: 0.921716,
					4: 0.940056,
					5: 0.951440,
					6: 0.959193,
					7: 0.964812
				}, .9 : {
					2: 0.947214,
					3: 0.964159,
					4: 0.972871,
					5: 0.978176,
					6: 0.981746,
					7: 0.984313
				}
			}
			if marg in lookup and m in lookup[marg]:
				p = lookup[marg][m]

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
