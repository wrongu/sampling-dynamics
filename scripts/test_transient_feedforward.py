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
parser.add_argument('--m-min', dest='m_min', type=int, default=2)
parser.add_argument('--eps', dest='eps', type=float, default=0.05)
parser.add_argument('--transience-max', dest='t_max', type=int, default=6) # max number of samples to try where boost has an effect
parser.add_argument('--boost-max', dest='boost_max', type=float, default=2.0) # exponent in feedforward term
parser.add_argument('--boost-steps', dest='steps', type=int, default=11) # exponent in feedforward term
args = parser.parse_args()

Ms = range(args.m_min, args.m_max+1)

mixing_times = np.zeros((args.t_max+1, args.steps, len(Ms)))

for mi,m in enumerate(Ms):
	net = m_deep_bistable(m, marg=args.marg)
	ev = net.get_node_by_name('X1')
	p = ev.get_table()[0,0]

	A = load_or_run('transition_matrix_M%d_p%.3f_noev' % (m, p), lambda: construct_markov_transition_matrix(net), force_recompute=args.recompute)
	A = set_transition_matrix_evidence(net, A, {ev: 1})
	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_target = analytic_marginal_states(net, conditioned_on={ev: 1})

	for bi, b in enumerate(np.linspace(1.0, args.boost_max, args.steps)):
		A_ff = load_or_run('transition_matrix_transient_ff_M%d_p%.3f_b%.3f' % (m, p, b),
			lambda: construct_markov_transition_matrix(net, feedforward_boost=b), force_recompute=args.recompute)
		A_ff = set_transition_matrix_evidence(net, A_ff, {ev: 1})

		for transient_steps in range(args.t_max+1):
			# first 'transience' samples are with A_ff
			S = S_start.copy()
			tvd = variational_distance(S, S_target)
			for t in range(transient_steps):
				if tvd < args.eps:
					mixing_times[transient_steps, bi, mi] = t
					break
				S = np.dot(A_ff, S)
				tvd = variational_distance(S, S_target)
			else: # aka 'nobreak'
				mixing_times[transient_steps, bi, mi], _ = mixing_time(S, S_target, transition=A, eps=args.eps)
				mixing_times[transient_steps, bi, mi] += transient_steps
# if args.plot:
# 	fig = plt.figure()
# 	ax = fig.add_subplot(1,1,1)
# 	ax.plot(Ms, mixing_times_transient, '-o')
# 	ax.plot(Ms, mixing_times, '-o')
# 	ax.legend(['boosted', 'baseline'])
# 	ax.set_xlabel('Model Depth (M)')
# 	ax.set_ylabel('mixing time')
# 	plt.title('Change in Mixing Time with %d-Step %.1f-strength Transient feedforward boost' % (args.transience, args.boost))
# 	plt.savefig('plots/transient_t%d_b%.3f.png' % (args.transience, args.boost))
# 	plt.close()
