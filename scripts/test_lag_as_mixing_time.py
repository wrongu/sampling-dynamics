if __name__ == '__main__' and __package__ is None:
	__package__ = 'sampling.scripts'

import numpy as np
import matplotlib.pyplot as plt
import argparse
from models import k_deep_bistable
from visualize import plot_net_layerwise
from util import load_or_run
from counting import *

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', dest='recompute', action='store_true', default=False)
parser.add_argument('--prob', dest='p', type=float, default=0.96)
parser.add_argument('--eps', dest='eps', type=float, default=0.05)
parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
parser.add_argument('--k-max', dest='k_max', type=int, default=7)
parser.add_argument('--init-samples', dest='samples', type=int, default=10000)

args = parser.parse_args()

def variational_distance(P1, P2):
	return 0.5 * np.abs(P1 - P2).sum()

max_t = 1000
mixing_times = np.zeros(args.k_max-1)
vds = np.zeros((max_t, args.k_max-1))
for K in range(2, args.k_max+1):
	print K
	net = k_deep_bistable(K, args.p)
	ev = net.get_node_by_name('X1')
	P = load_or_run('transition_matrix_K%d_p%.3f' % (K, args.p), lambda: construct_markov_transition_matrix(net, conditioned_on={ev: 1}))
	S_start  = np.zeros(count_states(net))
	S_target = np.zeros(count_states(net))

	for i in range(2**K):
		id_to_state(net, i)
		S_start[i]  = net.probability(conditioned_on={ev: 0})
		S_target[i] = net.probability(conditioned_on={ev: 1})
	S_start = S_start / S_start.sum()
	S_target = S_target / S_target.sum()

	S = S_start
	i = 0
	d = variational_distance(S_target, S)
	while d >= args.eps:
		vds[i,K-2] = d
		i = i+1
		S = np.dot(P,S)
		d = variational_distance(S_target, S)
		print i,d
		if i == max_t-1:
			break
	mixing_times[K-2] = i

if args.plot:
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.plot(range(2,args.k_max+1), mixing_times, '-k*')
	ax.set_xlim([0,args.k_max+1])
	plt.xlabel('number of layers')
	plt.ylabel('mixing time (samples to variational distance < eps)')
	plt.title('Mixing Time as a funciton of depth')
	plt.savefig('plots/mixing_time.png')
	plt.close()