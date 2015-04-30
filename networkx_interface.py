import networkx as nx
from counting import *
from graphical_models import *

def net_to_digraph(net):
	g = nx.DiGraph()
	g.add_nodes_from(net.iter_nodes())
	g.add_edges_from(net.iter_edges())

	return g

if __name__ == '__main__':
	from models import erdos_renyi
	import matplotlib.pyplot as plt

	G = erdos_renyi(100,0.01)
	DG = net_to_digraph(G)

	fig = plt.figure()
	fig.add_subplot(121)
	nx.draw_random(DG)
	fig.add_subplot(122)
	nx.draw_circular(DG)	
	plt.show()
	plt.close()

	from models import m_deep_bistable
	net = m_deep_bistable(7, 0.96)
	g = net_to_digraph(net)
	nx.draw(g)
	plt.show()
	plt.close()