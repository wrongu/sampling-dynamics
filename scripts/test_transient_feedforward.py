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
parser.add_argument('--boost-steps', dest='boost_steps', type=int, default=11) # exponent in feedforward term
args = parser.parse_args()

Ms = range(args.m_min, args.m_max+1)
Ts = range(args.t_max + 1)
alphas = np.linspace(1.0, args.boost_max, args.boost_steps)

def eig_steadystate(A):
	# steady state distribution of A_ff transition matrix is largest eigenvector (eigenvalue=1)
	w,v = np.linalg.eig(A)
	inds = np.argsort(w)
	S_steady_state = np.abs(v[:,inds[-1]])
	S_steady_state /= S_steady_state.sum()
	return S_steady_state

mixing_times = np.zeros((args.boost_steps, args.t_max+1, len(Ms)))

for mi,m in enumerate(Ms):
	net = m_deep_bistable(m, marg=args.marg)
	ev = net.get_node_by_name('X1')
	p = ev.get_table()[0,0]

	A = load_or_run('transition_matrix_M%d_p%.3f_noev' % (m, p), lambda: construct_markov_transition_matrix(net), force_recompute=args.recompute)
	A = set_transition_matrix_evidence(net, A, {ev: 1})
	S_start  = analytic_marginal_states(net, conditioned_on={ev: 0})
	S_target = analytic_marginal_states(net, conditioned_on={ev: 1})

	for ai, a in enumerate(alphas):
		A_ff = load_or_run('transition_matrix_transient_ff_M%d_p%.3f_b%.3f' % (m, p, a),
			lambda: construct_markov_transition_matrix(net, feedforward_boost=a), force_recompute=args.recompute)
		A_ff = set_transition_matrix_evidence(net, A_ff, {ev: 1})

		for transient_steps in Ts:
			# avoid possibility that S-->S_ff_steadystate 'passes through' S_target, apparently
			# looking like mixing time is very small
			ff_steady_state_in_range = variational_distance(eig_steadystate(A_ff), S_target) < args.eps
			# first 'transience' samples are with A_ff
			S = S_start.copy()
			tvd = variational_distance(S, S_target)
			for t in range(transient_steps):
				if ff_steady_state_in_range and tvd < args.eps:
					mixing_times[ai, transient_steps, mi] = t
					break
				S = np.dot(A_ff, S)
				tvd = variational_distance(S, S_target)
			else: # aka 'nobreak'
				mixing_times[ai, transient_steps, mi], _ = mixing_time(S, S_target, transition=A, eps=args.eps)
				mixing_times[ai, transient_steps, mi] += transient_steps
if args.plot:
	# Mixing Time Plots
	for mi,m in enumerate(Ms):
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		for t in Ts:
			ax.plot(alphas, mixing_times[:,t,mi], '-o')
		ax.set_ylim([0,mixing_times.max()+5])
		ax.legend(['baseline'] + ['%d steps' % t for t in Ts[1:]], loc='upper right')
		ax.set_xlabel('alpha')
		ax.set_ylabel('mixing time')
		plt.title('Mixing Time reductions as function of alpha, m = %d' % m)
		plt.savefig('plots/transient_m%d.png' % m)
		plt.close()

	# TVD Plots
	for mi,m in enumerate(Ms):
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		tvds = np.zeros(alphas.shape)
		
		net = m_deep_bistable(m, marg=args.marg)
		ev = net.get_node_by_name('X1')
		p = ev.get_table()[0,0]
		
		for ai,a in enumerate(alphas):
			A = load_or_run('transition_matrix_M%d_p%.3f_noev' % (m, p), lambda: construct_markov_transition_matrix(net))
			A = set_transition_matrix_evidence(net, A, {ev: 1})
	
			A_ff = load_or_run('transition_matrix_transient_ff_M%d_p%.3f_b%.3f' % (m, p, a),
				lambda: construct_markov_transition_matrix(net, feedforward_boost=a))
			A_ff = set_transition_matrix_evidence(net, A_ff, {ev: 1})
			S_ff_steady_state = eig_steadystate(A_ff)
			S_baseline = analytic_marginal_states(net, conditioned_on={ev: 1}) # same as S_target above
			tvds[ai] = variational_distance(S_baseline, S_ff_steady_state)
		ax.plot(alphas, tvds, '-o')
		ax.set_xlabel('alpha')
		ax.set_ylabel('total variational distance')
		plt.title('Total Variational Distance of steady states A vs A_ff')
		plt.savefig('plots/transient_tvd_m%d.png' % m)
		plt.close()