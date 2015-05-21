if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
import argparse
from scripts.switching_times import top_node_percept, analytic_switching_times
from models import m_deep_bistable
from util import load_or_run
from counting import *

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', dest='recompute', action='store_true', default=False)
parser.add_argument('--marg', dest='marg', type=float, default=0.9)
parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
parser.add_argument('--m-max', dest='m_max', type=int, default=6)
parser.add_argument('--switch-eps', dest='s_eps', type=float, default=0.001) # maximum amount of un-accounted-for probability mass at the end of computing switching time distribution
parser.add_argument('--mix-eps', dest='m_eps', type=float, default=0.05) # 'mixing' finishes when TVD gets below this threshold
args = parser.parse_args()

Ms = range(2, args.m_max+1)

switching_times = np.zeros(len(Ms))
mixing_times = np.zeros(len(Ms))

for i,m in enumerate(Ms):
	net = m_deep_bistable(m, marg=args.marg)
	ev = net.get_node_by_name('X1')
	p = ev.get_table()[0,0]
	A = load_or_run('transition_matrix_M%d_p%.3f_noev' % (m, p), lambda: construct_markov_transition_matrix(net), force_recompute=args.recompute)

	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_target = analytic_marginal_states(net, conditioned_on={ev: 1})
	
	switching_time_histogram = analytic_switching_times(net, S_start, (top_node_percept, 1), transition=A, eps=args.s_eps)
	# compute mean switching time = sum_{t} t*p(t) = dot(histogram, 0...max_t)
	switching_times[i] = np.dot(switching_time_histogram, np.arange(len(switching_time_histogram)))

	A = set_transition_matrix_evidence(net, A, {ev: 1})
	mixing_times[i], _ = mixing_time(S_start, S_target, transition=A, eps=args.m_eps)

if args.plot:
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(switching_times, mixing_times, s=15)
	for m,x,y in zip(Ms, switching_times, mixing_times):
		ax.text(x+1,y,'M=%d' % m)
	ax.set_xlabel('mean dominance time')
	ax.set_ylabel('mixing time')
	xl, yl = ax.get_xlim(), ax.get_ylim()
	ax.set_ylim([0,yl[1]])
	plt.title('Relation between Mixing Time and Dominance Time')
	plt.savefig('plots/switch_vs_mix.png')
	plt.close()