import numpy as np
from itertools import izip
from generators import *
from graphs import Graph, DiGraph

class Variable(object):
	"""A Variable is a node in a belief net"""
	
	_name = ""
	_state = None # current value

	def __init__(self, name=""):
		self._name = name

	def __str__(self):
		return self.__unicode__()

	def __unicode__(self):
		nm = repr(self._name)
		return u"Variable[%s]" % (nm)

	def get_value(self):
		return self._state

	def set_value(self, v):
		self._state = v

	def get_name(self):
		return self._name

class DiscreteVariable(Variable):

	def __init__(self, states, tbl=None, **kwargs):
		super(DiscreteVariable, self).__init__(**kwargs)
		n = len(states)

		# ordered states such that cpt[i] corresponds to the slice for states[i]
		self._states = states
		self._state_index = -1

		# Distribution as Conditional Probability Table; dims = #parents + 1
		self._table = tbl

		# default to flat prior if not given
		if self._table is None:
			self._table = np.ones(n) / n

		# initialize to no state
		self._state = None

	def __str__(self):
		return self.__unicode__()

	def __unicode__(self):
		return "Discrete%s[%s]" % (super(DiscreteVariable,self).__unicode__(), str(self._state))

	def size(self):
		return len(self._states)

	def set_value(self, v):
		"""overrides Variable.set_value; adds additional check that v is valid
		"""
		if v in self._states:
			super(DiscreteVariable, self).set_value(v)
			self._state_index = self._states.index(self._state)
		elif v is None:
			self._state = None
			self._state_index = -1
		else:
			raise AssertionError("%s is not a valid state for %s" % (str(v), str(self)))

	def set_value_by_index(self, i):
		self._state = self._states[i]
		self._state_index = i

	def state_index(self):
		return self._state_index

	def get_table(self):
		return self._table

	def set_table(self, tbl):
		self._table = tbl

class ContinuousVariable(Variable):
	# not implemented yet
	pass

class BayesNet(DiGraph):
	"""Implementation of Bayesian Belief Nets.

	A BayesNet object is a DiGraph where edges represent (probabilistic) causation.

	The model is arranged as follows:
	- by extending the DiGraph class, organization of named nodes and edges is automatic
	- each node in the graph is a Variable from above. currently only DiscreteVariables are supported.
	- the conditional probability table for each variable must be added with a call to cpt().
		This enables unambiguous verification of which parent corresponds to which dimension of the table.
		The table itself is stored in the '_table' field of the variable
	- each edge from parent to child is tagged with an integer: the dimension of the table corresponding
		to that parent (see cpt())
	- if a variable is observed, its '_state' is non-None
	"""

	def __init__(self, *args, **kwargs):
		super(BayesNet, self).__init__(*args, **kwargs)
		
		# _verified is true iff this net has been validated 1) as a DAG and 2) with all CPTs present
		self._verified = True

		# TODO actually use verbosity
		self._verbose = kwargs.get("verbose", False)

	def add_node(self, n, tbl=None):
		"""calls DiGraph.add_node and marks the network as invalid since ordering
		may need to be updated
		"""
		super(BayesNet, self).add_node(n)

		self._verified = False

	def cpt(self, variables, table):
		"""Set a Conditional Probability Table.

		variables: an array of variable names (or ids) for which the cpt is being set. They should
		 be given in the same order they are accessed in the given table. For example, if 'X' 'Y'
		 and 'Z' are nodes with 2, 3, and 4 discrete states, then variables=['X' 'Y' 'Z'] expresses
		 P(Z | X,Y) and table=[a 2x3x4 numpy array]
		"""
		nodes = [(self.get_node_by_name(v) if type(v) is str else v) for v in variables]

		# check table shape and size
		table_dims = len(table.shape)
		if len(nodes) != table_dims:
			raise AssertionError("BayesNet.cpt() table size does not match number of variables")
		for d in xrange(table_dims):
			if nodes[d].size() != table.shape[d]:
				raise AssertionError("BayesNet.cpt() table dimension mismatch on variable '%s'" % (variables[d]))

		# table verified. update net topology
		parents = nodes[:-1]
		child = nodes[-1]
		for d in xrange(table_dims-1):
			# add (or update) edge from parent to child
			# the object associated with the edge is the dimension on the table
			#  for the associated parent (so that table lookup is still possible
			#  even if edges are reordered)
			self.add_edge(parents[d], child, d)

		# ...however, adding this CPT may have altered the net so as to make it invalid
		self._verified = False
		
		# set object of this node to be the cpt
		child.set_table(table)

	def evidence(self, values, learn=False):
		for (var,value) in self.map_convert_strings_to_nodes(values).iteritems():
			var.set_value(value)

		# note: no learning is done by the network, otherwise it would be here

	def evidence_from_vector(self, values_vec, learn=False):
		for (n,v) in zip(self._nodes, values_vec):
			n.set_value(v)

	def is_consistent_with_evidence(self, evidence):
		for n,v in evidence.iteritems():
			if n.get_value() != v: return False
		return True

	def probability(self, conditioned_on={}):
		"""compute probability of current state"""
		# TODO normalize over conditioned_on=all other states??
		tmp = self.state_map()
		self.evidence(conditioned_on)
		p = 1.
		for n in self._nodes:
			if n in conditioned_on:
				if conditioned_on[n] != tmp[n]:
					self.evidence(tmp)
					return 0.
			cpt = n.get_table()
			slc = [Ellipsis] * (len(self._reverse_edges[n])+1)
			for par,dim in self._reverse_edges[n].iteritems():
				slc[dim] = par.state_index()
			slc[-1] = n.state_index()
			# print "\t", cpt[tuple(slc)]
			p = p * cpt[tuple(slc)]
		self.evidence(tmp)
		return p

	def state_vector(self):
		return [n.get_value() for n in self._nodes]

	def state_map(self):
		return {n: n.get_value() for n in self._nodes}

	def __sorting_visit(self, n, _sorted, _visited):
		if _visited[n] is Graph.WHITE:
			_visited[n] = Graph.GRAY
			# visit parents first
			pe = self._reverse_edges[n]
			for p in pe.keys():
				self.__sorting_visit(p, _sorted, _visited)

			# parents done. add this node to the list and mark done
			_visited[n] = Graph.BLACK
			_sorted[self._sort_next_idx] = n
			self._sort_next_idx += 1
		elif _visited[n] is Graph.GRAY:
			raise AssertionError("cycle found - not a DAG")

	def __sort_by_ancestry(self):
		"""reorders self._nodes from ancestors to descendants.

		More formally, sorts such that:
			i > j => self._nodes[i] is not an ancestor of self._nodes[j]
		"""
		_sorted = [None] * len(self._nodes)
		_visited = dict(zip(self._nodes, [Graph.WHITE] * len(self._nodes)))

		# visit each node at least once. Note that if DAG is connected,
		# visit[_nodes[0]] would suffice. visiting each one does not add
		# overhead since visit(n) is fast when n has already been visited
		self._sort_next_idx = 0;			
		for n in self._nodes:
			self.__sorting_visit(n, _sorted, _visited)
		self._nodes = _sorted
		for i,n in enumerate(self._nodes):
			self._index_lookup[n] = i

	def check_cpts(self):
		for n in self._nodes:
			cpt = n.get_table()
			if cpt is None:
				raise AssertionError("No CPT for node %s" % str(n))
			# number of cpt dimensions = # parents + 1
			if len(cpt.shape) != len(self._reverse_edges[n]) + 1:
				raise AssertionError("CPT wrong number of dimensions")
			# last dim of cpt is self: check size
			if cpt.shape[-1] != n.size():
				raise AssertionError("CPT dimension mismatch")
			# check size on each parent's dimensions
			sz  = cpt.shape
			for (p, d) in self._reverse_edges[n].iteritems():
				if sz[d] != p.size():
					raise AssertionError("CPT dimension mismatch")
		return True

	def verify(self):
		if not self._verified:
			self.__sort_by_ancestry()
			self._verified = self.check_cpts()
		return self._verified

	def markov_blanket_marginal(self, node, fatigue_tau=None):
		"""compute the marginal probability of the requested node conditioned on its (observed)
		Markov Blanket. The distribution is returned as a vector parallel to node._state
		"""
		# product of probabilities is done as sums in log space (to preserve precision);
		# the returned vector is not in log space, however
		cpt = np.log(node.get_table())
		
		# slice into CPT according to parent states
		causes_slice = [Ellipsis] * cpt.ndim
		for par, table_dim in self._reverse_edges[node].iteritems():
			causes_slice[table_dim] = par.state_index()
			# TODO: a version allowing for unobserved parents
			# (would require implementing message passing algorithm to get marginal for parent)
		cpt = cpt[causes_slice]

		# must update CPT according to complete markov blanket.
		# So here, need to consider to what extent the current node
		# can serve as the explanation for each child's state (which
		# also depends on the other possible explanations of the child's
		# states, i.e. this node's 'spouses')
		for (child, node_idx) in self._edges[node].iteritems():
			if child.get_value() is None:
				# unobserved children have no effect on P(self)
				continue
			
			table = child.get_table()
			# get table slice corresponding to current child state
			table = table[..., child.state_index()]
			explanations_slice = [Ellipsis] * table.ndim
			for (childs_parent, cp_idx) in self._reverse_edges[child].iteritems():
				if cp_idx == node_idx:
					# don't collapse dimension relating to the node we're sampling
					continue
				# slice out with respect to current value of childs-parent
				explanations_slice[cp_idx] = childs_parent.state_index()

			# do slicing
			table = table[explanations_slice]

			# here, table is reduced to possible explanations of child with respect to node
			cpt = cpt + np.log(table.reshape(*cpt.shape))

		distrib = np.exp(cpt.reshape(node.size()))

		if fatigue_tau is not None:
			# TODO generalize beyond binary?
			distrib[node.state_index()] *= fatigue_tau
			distrib[1-node.state_index()] *= (1-fatigue_tau)

		return distrib / float(distrib.sum())

	def sample_node(self, node):
		"""Sample the requested Node according to the state of its Markov Blanket
		"""
		# take random sample
		distrib = self.markov_blanket_marginal(node)
		value = np.random.choice(node._states, p=distrib)
		node.set_value(value)
		return value

	# Overriding DirectedGraph.parents to ensure order is same as in CPT indices
	def parents(self, n):
		rev = self._reverse_edges[n]
		rents = super(BayesNet, self).parents(n)
		idxs = [rev[p] for p in rents]
		return [rents[i] for i in idxs] # radix sort