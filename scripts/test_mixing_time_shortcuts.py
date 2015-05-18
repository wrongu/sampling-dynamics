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

def get_mixing_time(net, identifier):
	ev = net.get_node_by_name('X1')
	P = load_or_run('transition_matrix_shortcuts_m%d_f%d_t%d_%s' % (args.m, args.fro, args.to, identifier),
		lambda: construct_markov_transition_matrix(net),
		force_recompute=args.recompute)
	P = set_transition_matrix_evidence(net, P, {ev: 1})
	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_target = analytic_marginal_states(net, conditioned_on={ev: 1})
	return mixing_time(S_start, S_target, P, eps=args.eps)[0]

# first data point: baseline mixing time (no shortcut)
net_baseline = m_deep_bistable(args.m, marg=args.marg)
ev = net_baseline.get_node_by_name('X1')
p = ev.get_table()[0,0]
P = load_or_run('transition_matrix_K%d_p%.3f' % (args.m, p),
	lambda: construct_markov_transition_matrix(net_baseline),
	force_recompute=args.recompute)
S_start_baseline  = analytic_marginal_states(net_baseline, conditioned_on={ev: 0})
S_target_baseline = analytic_marginal_states(net_baseline, conditioned_on={ev: 1})
mixing_time_baseline, _ = mixing_time(S_start_baseline, S_target_baseline, P, eps=args.eps)
print 'baseline', mixing_time_baseline

# second data point: shortcut uses marginal distribution
net_marginal = m_deep_with_shortcut(args.m, marg=args.marg, fro=args.fro, to=args.to, cpt='marginal')
mixing_time_marginal = get_mixing_time(net_marginal, 'marginal')
print 'marginal', mixing_time_marginal

# third data point: q = 0.5
net_half = m_deep_with_shortcut(args.m, marg=args.marg, fro=args.fro, to=args.to, cpt=np.array([[.5, .5],[.5, .5]]))
mixing_time_half = get_mixing_time(net_half, 'dep0.500')
print 'q = .5', mixing_time_half

# get results for varying fro-to dependencies
dependencies = np.linspace(args.q_min, args.q_max, args.steps) # AKA 'q'
mixing_times = np.zeros(args.steps)
for i,dep in enumerate(dependencies):
	cpt = np.array([[dep, 1-dep],[1-dep, dep]])
	net = m_deep_with_shortcut(args.m, marg=args.marg, fro=args.fro, to=args.to, cpt=cpt)
	mixing_times[i] = get_mixing_time(net, 'dep%.3f' % dep)
	print 'dep', dep, mixing_times[i]

if args.plot:
	from visualize import plot_net_layerwise
	# plot model
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	node_positions = {
		net._nodes[0] : (0,5),
		net._nodes[1] : (0,4),
		net._nodes[2] : (.5,3),
		net._nodes[3] : (.5,2),
		net._nodes[4] : (0,1),
		net._nodes[5] : (0,0),
	}
	plot_net_layerwise(net, ax=ax, positions=node_positions, x_spacing=.5, y_spacing=1)
	plt.savefig("plots/model_shortcut.png")
	plt.close()

	fig = plt.figure()

	# plot mixing times
	ax = fig.add_subplot(2,1,1)

	marg_x = compute_marginal_for_given_p(args.fro-args.to, p)
	idx = np.searchsorted(dependencies, marg_x)
	dependencies = np.insert(dependencies, idx, marg_x)
	mixing_times = np.insert(mixing_times, idx, mixing_time_marginal)

	idx = np.searchsorted(dependencies, 0.5)
	dependencies = np.insert(dependencies, idx, 0.5)
	mixing_times = np.insert(mixing_times, idx, mixing_time_half)

	ax.plot([args.q_min, args.q_max], [mixing_time_baseline]*2, '--k')

	ax.plot(dependencies, mixing_times, '-bo')
	ax.plot(marg_x, mixing_time_marginal, 'ro')

	plt.title('Effect of X2-X5 Shortcut on Mixing Times')
	plt.ylabel('Mixing Time')
	plt.legend(['no shortcut', 'shortcut'], loc='upper right')

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
	plt.title('Change in Posterior Varying q')
	plt.xlabel('q')
	plt.ylabel('TVD(S_q, S_baseline)')

	plt.savefig('plots/shortcut_allplots.png')
	plt.close()

