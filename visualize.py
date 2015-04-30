import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from collections import defaultdict
from graphical_models import *
from networkx_interface import net_to_digraph

def plot_net_layerwise(net, x_spacing=5, y_spacing=10, colors={}, use_labels=True, ax=None, cmap='gist_heat', cbar=False):
	args = {
		'ax' : ax,
		'node_color' : colors,
		'nodelist' : net.nodes(), # ensure that same order is used throughout for parallel data like colors
		'vmin' : 0,
		'vmax' : 1,
		'cmap' : cmap
	}

	# compute layer-wise positions of nodes (distance from roots)
	nodes_by_layer = defaultdict(lambda: [])
	def add_to_layer(n,l):
		nodes_by_layer[l].append(n)
	net.bfs_traverse(net.get_roots(), add_to_layer)


	positions = {}
	for l, nodes in nodes_by_layer.iteritems():
		y = -l*y_spacing
		# reorder layer lexicographically
		nodes.sort(key=lambda n: n.get_name())
		width = (len(nodes)-1) * x_spacing
		for i,n in enumerate(nodes):
			x = x_spacing*i - width/2
			positions[n] = (x,y)
	args['pos'] = positions

	if use_labels:
		labels = {n:n.get_name() for n in net.iter_nodes()}
		args['labels'] = labels

	if ax is None:
		ax = plt.figure().add_subplot(1,1,1)
	nxg = net_to_digraph(net)
	nx.draw_networkx(nxg, **args)
	ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
	ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

	if cbar:
		color_map = ScalarMappable(cmap=cmap)
		color_map.set_clim(vmin=0, vmax=1)
		color_map.set_array(np.array([0,1]))
		plt.colorbar(color_map, ax=ax)

if __name__ == '__main__':
	from models import *
	fig = plt.figure()
	
	net1 = m_deep_bistable(5, 0.9)
	net2 = m_wide_bistable(5, 0.9)

	fig.add_subplot(121)
	plot_net_layerwise(net1)

	fig.add_subplot(122)
	plot_net_layerwise(net2)

	plt.show()