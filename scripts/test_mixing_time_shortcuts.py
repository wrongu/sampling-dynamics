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
parser.add_argument('--plot-self', dest='plot_self', action='store_true', default=False)
parser.add_argument('--steps', dest='steps', type=int, default=12)
parser.add_argument('--q-min', dest='q_min', type=float, default=0.5)
parser.add_argument('--q-max', dest='q_max', type=float, default=1.0)
parser.add_argument('--eps', dest='eps', type=float, default=.05)
parser.add_argument('--depth', dest='m', type=int, default=6)
parser.add_argument('--marg', dest='marg', type=float, default=0.9)
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

def get_mixing_time_base_self(net, fro, to, identifier):
	ev = net.get_node_by_name('X1')
	A_sc = load_or_run('transition_matrix_shortcuts_m%d_f%d_t%d_%s_ev1' % (args.m, fro, to, identifier),
		lambda: construct_markov_transition_matrix(net, conditioned_on={ev: 1}),
		force_recompute=args.recompute)
	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_steady_state = analytic_marginal_states(net, conditioned_on={ev: 1})

	mt_base,_ = mixing_time(S_start, S_target_baseline, A_sc, eps=args.eps, converging_to=S_steady_state)
	mt_self,_ = mixing_time(S_start, S_steady_state, A_sc, eps=args.eps, converging_to=S_steady_state)
	return (mt_base, mt_self)

qs = np.linspace(args.q_min, args.q_max, args.steps)

nodes = range(2,args.m+1)
fro_to_pairs = [(fro,to) for fro,to in itertools.product(nodes, nodes) if fro-to>1]

mixing_times_base = np.zeros((args.steps, len(fro_to_pairs)))
mixing_times_self = np.zeros((args.steps, len(fro_to_pairs)))

# shortcuts: all valid combinations of fro->to (where fro-to > 1)
for pi, (fro,to) in enumerate(fro_to_pairs):
	for qi,q in enumerate(qs):
		cpt = np.array([[q, 1-q],[1-q, q]])
		net = m_deep_with_shortcut(args.m, marg=args.marg, fro=fro, to=to, cpt=cpt)
		mixing_times_base[qi,pi], mixing_times_self[qi,pi] = get_mixing_time_base_self(net, fro, to, 'dep%.3f' % q)

if args.plot:

	# plot mixing times
	fig = plt.figure()
	ax = fig.add_subplot(2,1,1)

	markers = 'o^+D'

	# plot MT_baseline
	for pi,(fro,to) in enumerate(fro_to_pairs):
		marker = markers[fro-to-2]
		mts = mixing_times_base[:,pi]
		reasonable_times = mts < 1000
		reasonable_times[-1] = False # q=1.0 is distracting and not actually useful
		ax.plot(qs[reasonable_times], mts[reasonable_times], '-%c' % marker, label='$x_%d-x_%d$' % (fro,to))
	ax.legend(['%d->%d' % (fro,to) for fro,to in fro_to_pairs], loc='lower left', ncol=2)
	# plot MT_self
	if args.plot_self:
		ax.set_color_cycle(None)
		for pi,(fro,to) in enumerate(fro_to_pairs):
			marker = markers[fro-to-2]
			mts = mixing_times_self[:,pi]
			reasonable_times = mts < 1000
			reasonable_times[-1] = False # q=1.0 is distracting and not actually useful
			ax.plot(qs[reasonable_times], mts[reasonable_times], '--%c' % marker)

	# dashed 'baseline' line
	ax.plot([args.q_min, args.q_max], [mixing_time_baseline]*2, '--k')
	ax.set_xlim([args.q_min, args.q_max])
	ax.set_ylim([0,50])

	plt.ylabel('mixing time')

	yl = ax.get_ylim()
	ax.set_ylim([0,yl[1]+5])

	# plot TVD as function of q
	ax = fig.add_subplot(2,1,2)
	tvds = np.zeros((len(qs), len(fro_to_pairs)))
	for pi,(fro,to) in enumerate(fro_to_pairs):
		for qi,q in enumerate(qs):
			cpt = np.array([[q, 1-q],[1-q, q]])
			net = m_deep_with_shortcut(args.m, marg=args.marg, fro=fro, to=to, cpt=cpt)
			ev = net.get_node_by_name('X1')
			tvds[qi,pi] = variational_distance(S_target_baseline, analytic_marginal_states(net, conditioned_on={ev: 1}))
	for pi,(fro,to) in enumerate(fro_to_pairs):
		marker = markers[fro-to-2]
		plt.plot(qs[:-1], tvds[:-1,pi],'-%c' % marker)
	ax.set_xlim([args.q_min, args.q_max])
	plt.xlabel('q')
	plt.ylabel('TVD from baseline')

	plt.savefig('plots/shortcut_allplots.png')
	plt.close()

