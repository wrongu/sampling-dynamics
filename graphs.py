import Queue

class Graph(object):
	# colors used for marking nodes during traversal algorithms
	WHITE = 0 # untouched 
	GRAY  = 1 # in process
	BLACK = 2 # finished

	_nodes_by_name = {} # map from name (string) to node. unnamed nodes not represented here
	_nodes = []         # all nodes, in proper sampling order
	_edges = {}         # map from source nodes to {map from destination node to edge objects}

	def __init__(self):
		self._nodes_by_name = {}
		self._index_lookup = {}
		self._nodes = []
		self._edges = {}

	def add_node(self, n):
		"""add a new node with some associated object
		"""
		self._nodes.append(n)
		if n._name is not "":
			if n._name in self._nodes_by_name:
				raise AssertionError("Node with name '%s' already exists" % n._name)
			self._nodes_by_name[n._name] = n
			self._index_lookup[n] = len(self._nodes)-1
		self._edges[n] = {}

	def add_edge(self, n1, n2, obj=None):
		"""add a new edge with some associated object
		"""
		e = self._edges[n1]
		e[n2] = obj
		self._edges[n1] = e
		# add reverse edge
		re = self._edges[n2]
		re[n1] = obj
		self._edges[n2] = re

	def get_node_by_name(self, name):
		n = self._nodes_by_name.get(name)
		if n is None:
			raise AssertionError("No node found with name '%s'" % name)
		return n

	def map_convert_strings_to_nodes(self, nmap):
		"""take a dict {node:val, ...} where 'node' may be a node object or
		a string, and return a map where keys are exclusively objects"""
		for k,v in nmap.iteritems():
			if type(k) is str:
				nmap[self.get_node_by_name(k)] = v
				del nmap[k]
		return nmap

	def size(self):
		return len(self._nodes)

	def iter_nodes(self):
		for n in self._nodes:
			yield n

	def iter_nodes_reversed(self):
		for n in reversed(self._nodes):
			yield n

	def iter_edges(self):
		for u in self.iter_nodes():
			for v in self._edges[u]:
				yield (u,v)

	def nodes(self):
		return self._nodes[:]

	def node_index(self, n):
		return self._index_lookup[n]

	def bfs_traverse(self, roots, traversal_fn):
		markers = {}
		q = Queue.Queue()

		for n in self.iter_nodes():
			markers[n] = Graph.WHITE
		for n in roots:
			markers[n] = Graph.BLACK
			q.put((n,0))

		while not q.empty():
			u,d = q.get()
			traversal_fn(u,d)
			neighbors = self._edges[u]
			for n in neighbors:
				if markers[n] == Graph.WHITE:
					q.put((n,d+1))
					markers[n] = Graph.BLACK

class DiGraph(Graph):
	_reverse_edges = {} # reverse direction (parent pointers rather than children)

	def __init__(self):
		super(DiGraph, self).__init__()
		self._reverse_edges = {}

	def add_node(self, n):
		super(DiGraph,self).add_node(n)
		self._reverse_edges[n] = {}

	def add_edge(self, n1, n2, obj=None):
		"""add a new edge with some associated object
		"""
		# add foward edge
		e = self._edges[n1]
		e[n2] = obj
		self._edges[n1] = e

		# add reverse edge
		# *NOT* to _edges but to _reverse_edges
		re = self._reverse_edges[n2]
		re[n1] = obj
		self._reverse_edges[n2] = re

	def get_roots(self):
		return [n for n in self.iter_nodes() if len(self._reverse_edges[n]) == 0]
