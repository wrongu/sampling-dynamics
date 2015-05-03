if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
from models import m_deep_with_shortcut, m_deep_bistable
from util import load_or_run
from counting import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', action='store_true', default=False)
parser.add_argument('--steps', dest='steps', type=int, default=12)
parser.add_argument('--eps', dest='eps', type=float, default=.05)
parser.add_argument(['--depth', '-M'], dest='m', type=int, default=6)
parser.add_argument('--marg', dest='marg', type=float, default=0.9)
parser.add_argument(['--fro', '--from'], dest='fro', type=int, default=5)
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
P = load_or_run('transition_matrix_K%d_p%.3f' % (K, p),
	lambda: construct_markov_transition_matrix(net_baseline),
	force_recompute=args.recompute)
S_start  = analytic_marginal_states(net_baseline, conditioned_on={ev: 0})
S_target = analytic_marginal_states(net_baseline, conditioned_on={ev: 1})
mixing_time_baseline, _ = mixing_time(S_start, S_target, P, eps=args.eps)

# second data point: shortcut uses marginal distribution
net_marginal = m_deep_with_shortcut(args.m, marg=args.marg, fro=args.fro, to=args.to, cpt='marginal')
mixing_time_marginal = get_mixing_time(net_marginal, 'marginal')

# get results for varying fro-to dependencies
dependencies = np.linspace(0.5, 1.0, args.steps)
mixing_times = np.zeros(args.steps)
for i,dep in enumerate(dependencies):
	cpt = np.array([[dep, 1-dep],[1-dep, dep]])
	net = m_deep_with_shortcut(args.m, marg=args.marg, fro=args.fro, to=args.to, cpt=cpt)
	mixing_times[i] = get_mixing_time(net, 'dep%.3f' % dep)

if args.plot:
	fig = plt.figure()
	ax = plt.add_subplot(1,1,1)
	plt.plot(dependencies, mixing_times, '-bo')
	plt.line([0.5,1.0], mixing_time_baseline, '--k')
	plt.plot([0.5,1.0], mixing_time_marginal, '--r')