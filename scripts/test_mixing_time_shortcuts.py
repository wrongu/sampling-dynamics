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

def get_mixing_time(net, identifier, S_target=None):
	ev = net.get_node_by_name('X1')
	A = load_or_run('transition_matrix_shortcuts_m%d_f%d_t%d_%s' % (args.m, args.fro, args.to, identifier),
		lambda: construct_markov_transition_matrix(net),
		force_recompute=args.recompute)
	A = set_transition_matrix_evidence(net, A, {ev: 1})
	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_steady_state = analytic_marginal_states(net, conditioned_on={ev: 1})
	return mixing_time(S_start, S_target, A, eps=args.eps, converging_to=S_steady_state)[0]

# first data point: baseline mixing time (no shortcut)
net_baseline = m_deep_bistable(args.m, marg=args.marg)
ev = net_baseline.get_node_by_name('X1')
p = ev.get_table()[0,0]
A = load_or_run('transition_matrix_M%d_p%.3f' % (args.m, p),
	lambda: construct_markov_transition_matrix(net_baseline),
	force_recompute=args.recompute)
A = set_transition_matrix_evidence(net_baseline, A, {ev:1})
S_start_baseline  = analytic_marginal_states(net_baseline, conditioned_on={ev: 0})
S_target_baseline = analytic_marginal_states(net_baseline, conditioned_on={ev: 1})
mixing_time_baseline, _ = mixing_time(S_start_baseline, S_target_baseline, A, eps=args.eps)
print 'baseline', mixing_time_baseline

mixing_time_target = S_target_baseline if args.baseline else None

# second data point: shortcut uses marginal distribution
net_marginal = m_deep_with_shortcut(args.m, marg=args.marg, fro=args.fro, to=args.to, cpt='marginal')
mixing_time_marginal = get_mixing_time(net_marginal, 'marginal', mixing_time_target)
print 'marginal', mixing_time_marginal

# third data point: q = 0.5
net_half = m_deep_with_shortcut(args.m, marg=args.marg, fro=args.fro, to=args.to, cpt=np.array([[.5, .5],[.5, .5]]))
mixing_time_half = get_mixing_time(net_half, 'dep0.500', mixing_time_target)
print 'q = .5', mixing_time_half

# get results for varying fro-to dependencies
dependencies = np.linspace(args.q_min, args.q_max, args.steps) # AKA 'q'
mixing_times = np.zeros(args.steps)
for i,dep in enumerate(dependencies):
	cpt = np.array([[dep, 1-dep],[1-dep, dep]])
	net = m_deep_with_shortcut(args.m, marg=args.marg, fro=args.fro, to=args.to, cpt=cpt)
	mixing_times[i] = get_mixing_time(net, 'dep%.3f' % dep, mixing_time_target)
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

	# plot mixing times
	fig = plt.figure()
	ax = fig.add_subplot(2,1,1)

	idx = np.searchsorted(dependencies, 0.5)
	dependencies = np.insert(dependencies, idx, 0.5)
	mixing_times = np.insert(mixing_times, idx, mixing_time_half)

	reasonable_times = mixing_times < 1000

	ax.plot([args.q_min, args.q_max], [mixing_time_baseline]*2, '--k')

	ax.plot(dependencies[reasonable_times], mixing_times[reasonable_times], '-bo')

	plt.title('Effect of X2-X5 Shortcut on Mixing Times')
	plt.ylabel('Mixing Time')
	plt.legend(['no shortcut', 'shortcut'], loc='lower left')

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

	plt.savefig('plots/shortcut_allplots.png')
	plt.close()

