if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
from models import m_deep_bistable
from util import load_or_run
from counting import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', action='store_true', default=False)
parser.add_argument('--eps', dest='eps', type=float, default=.05)
parser.add_argument('--m-max', dest='m', type=int, default=6)
parser.add_argument('--marg', dest='marg', type=float, default=0.9)
parser.add_argument('--tau-steps', dest='tau_steps', type=int, default=11)
parser.add_argument('--tau-min', dest='tau_min', type=float, default=0.1)
parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
args = parser.parse_args()

# for each M [2..m-max], get baseline mixing time, then try the
# 'adaptation' model for <tau_steps> values of tau=0.5...1.0
Ms = range(2,args.m+1)
taus = np.linspace(args.tau_min, 0.5, args.tau_steps)
mixing_times = np.zeros((args.tau_steps, len(Ms))) # one data point per model per tau

for mi,m in enumerate(Ms):
	net = m_deep_bistable(m, marg=args.marg)
	ev = net.get_node_by_name('X1')
	p = ev.get_table()[0,0]

	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_target = analytic_marginal_states(net, conditioned_on={ev: 1})

	for ti, tau in enumerate(taus):
		A_adapt = load_or_run('transition_matrix_adapt_M%d_p%.3f_tau%.3f_ev1' % (m, p, tau),
			lambda: construct_markov_transition_matrix(net, fatigue_tau=tau, conditioned_on={ev: 1}),
			force_recompute=args.recompute)

		mixing_times[ti,mi] = mixing_time(S_start, S_target, A_adapt, eps=args.eps,
			converging_to=eig_steadystate(A_adapt))[0]

if args.plot:
	# mixing time plots
	for mi,m in enumerate(Ms):

		net = m_deep_bistable(m, marg=args.marg)
		ev = net.get_node_by_name('X1')
		p = ev.get_table()[0,0]

		A = load_or_run('transition_matrix_M%d_p%.3f_ev1' % (m, p),
			lambda: construct_markov_transition_matrix(net, conditioned_on={ev: 1}))

		# Mixing Time Plot
		fig = plt.figure()
		ax = fig.add_subplot(2,1,1)

		mts = mixing_times[:,mi]
		reasonable_times = mts < 1000
		ax.plot(taus[reasonable_times], mts[reasonable_times], '-o')
		yl = ax.get_ylim()
		ax.set_ylim([0, yl[1]+2])

		plt.title('Mixing Times with Adaptation, M=%d' % m)
		ax.set_ylabel('mixing time')

		# TVD Plot
		ax = fig.add_subplot(2,1,2)
		tvds = np.zeros(taus.shape)
		for ti,tau in enumerate(taus):
			A_adapt = load_or_run('transition_matrix_adapt_M%d_p%.3f_tau%.3f_ev1' % (m, p, tau),
				lambda: construct_markov_transition_matrix(net, fatigue_tau=tau, conditioned_on={ev: 1}))

			tvds[ti] = variational_distance(
				eig_steadystate(A),
				eig_steadystate(A_adapt))

		ax.plot(taus, tvds, '-ok')
		plt.title('Change in Posterior with Adaptation')
		ax.set_xlabel('tau')
		ax.set_ylabel('TVD from baseline')

		plt.savefig('plots/adapt_allplots_m%d.png' % m)
		plt.close()
