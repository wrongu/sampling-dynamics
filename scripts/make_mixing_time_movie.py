if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import argparse
from counting import *
from util import load_or_run
from models import k_deep_bistable
from visualize import plot_net_layerwise

parser = argparse.ArgumentParser()
parser.add_argument('--prob', dest='p', type=float, default=0.96)
parser.add_argument('--eps', dest='eps', type=float, default=0.05)
parser.add_argument('--k', dest='K', type=int, default=6)

args = parser.parse_args()

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Mixing Time Animation', artist='Matplotlib', comment='')
writer = FFMpegWriter(fps=15, metadata=metadata)

# see test_lag_as_mixing_time:
K = args.K
net = k_deep_bistable(K, args.p)
ev = net.get_node_by_name('X1')
P = load_or_run('transition_matrix_K%d_p%.3f' % (K, args.p), lambda: construct_markov_transition_matrix(net, conditioned_on={ev: 1}))

S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
S_target = analytic_marginal_states(net, conditioned_on={ev: 1})

max_t = 100

S = np.zeros((2**K, max_t))
S[:,0] = S_start
i = 0
d = variational_distance(S_target, S[:,0])
while d >= args.eps:
	i = i+1
	S[:,i] = np.dot(P,S[:,i-1])
	d = variational_distance(S_target, S[:,i])
	print i,d
	if i == max_t-1:
		break
t_end = i

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plot_net_layerwise(net, colors=mean_state(net, S[:,0]), ax=ax)

with writer.saving(fig, "mixing_time_animation.mp4", 100):
	for i in range(t_end+1):
		print i
		plot_net_layerwise(net, colors=mean_state(net, S[:,i]), ax=ax)
		writer.grab_frame()

print '-done-'