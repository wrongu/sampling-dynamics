if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
from models import m_deep_with_shortcut, m_deep_bistable, compute_marginal_for_given_p
from util import load_or_run
from counting import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', action='store_true', default=False)
parser.add_argument('--target-baseline', dest='baseline', action='store_true', default=False)
parser.add_argument('--steps', dest='steps', type=int, default=12)
parser.add_argument('--q-min', dest='q_min', type=float, default=0.5)
parser.add_argument('--q-max', dest='q_max', type=float, default=1.0)
parser.add_argument('--eps', dest='eps', type=float, default=.05)
parser.add_argument('--depth', dest='m', type=int, default=6)
parser.add_argument('--marg', dest='marg', type=float, default=0.9)
parser.add_argument('--fro', dest='fro', type=int, default=5)
parser.add_argument('--to', dest='to', type=int, default=2)
parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
args = parser.parse_args()

# first data point: baseline mixing time (no shortcut)
net_baseline = m_deep_bistable(args.m, marg=args.marg)
ev = net_baseline.get_node_by_name('X1')
p = ev.get_table()[0,0]
A = load_or_run('transition_matrix_M%d_p%.3f_ev1' % (args.m, p),
	lambda: construct_markov_transition_matrix(net_baseline, conditioned_on={ev: 1}),
	force_recompute=args.recompute)
S_start_baseline  = analytic_marginal_states(net_baseline, conditioned_on={ev: 0})
S_target_baseline = analytic_marginal_states(net_baseline, conditioned_on={ev: 1})
mixing_time_baseline, _ = mixing_time(S_start_baseline, S_target_baseline, A, eps=args.eps)
print 'baseline', mixing_time_baseline

def get_mixing_time_base_self(net, identifier):
	ev = net.get_node_by_name('X1')
	A_sc = load_or_run('transition_matrix_shortcuts_m%d_f%d_t%d_%s_ev1' % (args.m, args.fro, args.to, identifier),
		lambda: construct_markov_transition_matrix(net, conditioned_on={ev: 1}),
		force_recompute=args.recompute)
	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_steady_state = analytic_marginal_states(net, conditioned_on={ev: 1})

	mt_base,_ = mixing_time(S_start, S_target_baseline, A_sc, eps=args.eps, converging_to=S_steady_state)
	mt_self,_ = mixing_time(S_start, S_steady_state, A_sc, eps=args.eps, converging_to=S_steady_state)
	return (mt_base, mt_self)

# get results for varying fro-to dependencies
dependencies = np.linspace(args.q_min, args.q_max, args.steps) # AKA 'q'
mixing_times_base = np.zeros(args.steps)
mixing_times_self = np.zeros(args.steps)
for i,dep in enumerate(dependencies):
	cpt = np.array([[dep, 1-dep],[1-dep, dep]])
	net = m_deep_with_shortcut(args.m, marg=args.marg, fro=args.fro, to=args.to, cpt=cpt)
	mixing_times_base[i], mixing_times_self[i] = get_mixing_time_base_self(net, 'dep%.3f' % dep)
	print 'dep', dep, mixing_times_base[i]

if args.plot:
	# from visualize import plot_net_layerwise
	# # plot model
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# node_positions = {
	# 	net._nodes[0] : (0,5),
	# 	net._nodes[1] : (0,4),
	# 	net._nodes[2] : (.5,3),
	# 	net._nodes[3] : (.5,2),
	# 	net._nodes[4] : (0,1),
	# 	net._nodes[5] : (0,0),
	# }
	# plot_net_layerwise(net, ax=ax, positions=node_positions, x_spacing=.5, y_spacing=1)
	# plt.savefig("plots/model_f%d_t%d_shortcut.png" % (args.fro, args.to))
	# plt.close()

	# plot mixing times
	fig = plt.figure()
	ax = fig.add_subplot(2,1,1)

	# dashed 'baseline' line
	ax.plot([args.q_min, args.q_max], [mixing_time_baseline]*2, '--k')

	# plot MT_baseline
	reasonable_times = mixing_times_base < 1000
	ax.plot(dependencies[reasonable_times], mixing_times_base[reasonable_times], '-bo')
	# plot MT_self
	reasonable_times = mixing_times_self < 1000
	ax.plot(dependencies[reasonable_times], mixing_times_self[reasonable_times], '--go')

	plt.title('Effect of X%d-X%d Shortcut on Mixing Times' % (args.to, args.fro))
	plt.ylabel('Mixing Time')

	yl = ax.get_ylim()
	ax.set_ylim([0,yl[1]+5])

	# plot TVD as function of q
	ax = fig.add_subplot(2,1,2)
	tvds = np.zeros(len(dependencies))
	for i,dep in enumerate(dependencies):
		cpt = np.array([[dep, 1-dep],[1-dep, dep]])
		net = m_deep_with_shortcut(args.m, marg=args.marg, fro=args.fro, to=args.to, cpt=cpt)
		ev = net.get_node_by_name('X1')
		tvds[i] = variational_distance(S_target_baseline, analytic_marginal_states(net, conditioned_on={ev: 1}))
	plt.plot(dependencies, tvds, '-ok')
	plt.title('Change in Posterior')
	plt.xlabel('q')
	plt.ylabel('TVD from baseline')

	plt.savefig('plots/shortcut_f%d_t%d_allplots.png' % (args.fro, args.to))
	plt.close()

