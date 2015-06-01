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

# _true and _self are MT with respect to unmodified transition matrix and modified one, respectively.
# When TVD(S_true_ss, S_self_ss)>eps, mixing_time_true is infinite.
mixing_times_true = np.zeros((args.tau_steps, len(Ms))) # one data point per model per tau
mixing_times_self = np.zeros((args.tau_steps, len(Ms)))

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

		converging_to = eig_steadystate(A_adapt)

		# compute mixing time to 'true' posterior
		mixing_times_true[ti,mi] = mixing_time(S_start, S_target, A_adapt, eps=args.eps,
			converging_to=converging_to)[0]
		# compute mixing time to 'modified' posterior
		mixing_times_self[ti,mi] = mixing_time(S_start, converging_to, A_adapt, eps=args.eps,
			converging_to=converging_to)[0]

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

		# 'true' mixing time plot
		reasonable_times = mixing_times_true[:,mi] < 1000
		ax.plot(taus[reasonable_times], mixing_times_true[reasonable_times,mi], '-o')
		# 'self' mixing time plot
		reasonable_times = mixing_times_self[:,mi] < 1000
		ax.plot(taus[reasonable_times], mixing_times_self[reasonable_times,mi], '--o')

		yl = ax.get_ylim()
		ax.set_ylim([0, yl[1]+2])
		xl = [max(args.tau_min-.05, 0), 0.55]
		ax.set_xlim(xl)

		ax.plot(xl,[mixing_times_true[-1,mi]]*2, '--k')

		plt.title('Mixing Times with Adaptation')
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
