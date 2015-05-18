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
from models import m_deep_bistable
from visualize import plot_net_layerwise

parser = argparse.ArgumentParser()
parser.add_argument('--prob', dest='p', type=float, default=0.96)
parser.add_argument('--eps', dest='eps', type=float, default=0.05)
parser.add_argument('--m', dest='M', type=int, default=6)
parser.add_argument('--res', type=int, default=240)

args = parser.parse_args()

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Mixing Time Animation', artist='Matplotlib', comment='')
writer = FFMpegWriter(fps=15, metadata=metadata)

# see test_lag_as_mixing_time:
M = args.M
net = m_deep_bistable(M, args.p)
ev = net.get_node_by_name('X1')
A = load_or_run('transition_matrix_M%d_p%.3f' % (M, args.p), lambda: construct_markov_transition_matrix(net, conditioned_on={ev: 1}))

S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
S_target = analytic_marginal_states(net, conditioned_on={ev: 1})

max_t = 100

S = np.zeros((2**M, max_t))
vds = np.zeros(max_t)
S[:,0] = S_start
i = 0
d = variational_distance(S_target, S[:,0])
vds[0] = d
while d >= args.eps:
	i = i+1
	S[:,i] = np.dot(A,S[:,i-1])
	d = variational_distance(S_target, S[:,i])
	vds[i] = d
	if i == max_t-1:
		break
t_end = i

fig = plt.figure()
net_ax = fig.add_subplot(2,1,1)
vd_ax  = fig.add_subplot(2,1,2)
vd_ax.set_title('Total Variational Distance')

plot_net_layerwise(net, colors=mean_state(net, S[:,0]), ax=net_ax, cbar=True)
vd_line, = vd_ax.plot([],[],'-m')
vd_ax.set_xlim([0, t_end])
vd_ax.set_ylim([0, 1.1])

text_obj = net_ax.text(.005, 0, "sample 0", fontsize=24, verticalalignment='top')
with writer.saving(fig, "plots/mixing_time_animation.mp4", args.res):
	for i in range(t_end+1):
		print "writing frame", i
		plot_net_layerwise(net, colors=mean_state(net, S[:,i]), ax=net_ax)
		text_obj.set_text("sample %d" % i)

		vd_line.set_xdata(np.arange(i+1))
		vd_line.set_ydata(vds[:i+1])
		writer.grab_frame()

print '-done-'