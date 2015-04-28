if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from util import load_or_run
from counting import *
from models import k_deep_bistable
from sampling import gibbs_sample
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', dest='recompute', action='store_true', default=False)
parser.add_argument('--prob', dest='p', type=float, default=0.96)
parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
parser.add_argument('--k-max', dest='k_max', type=int, default=7)
parser.add_argument('--samples', dest='samples', type=int, default=100000)
parser.add_argument('--burnin', dest='burnin', type=int, default=20)
args = parser.parse_args()


class SwitchedFunction(object):

	def __init__(self):
		self.last_state = None
		self.t0 = 0
		self.switching_times = defaultdict(lambda: 0)

	def __call__(self, i, net):
		current_state = plurality_state(net)
		if self.last_state is not None and self.last_state != current_state:
			self.switching_times[i-self.t0] += 1
			self.t0 = i
		self.last_state = current_state

	def distribution(self):
		d = np.zeros(max(self.switching_times.keys())+1)
		for k,v in self.switching_times.iteritems():
			d[k] = v
		return d

for K in range(2, args.k_max+1):
	print K
	net = k_deep_bistable(K, args.p)
	N = count_states(net)
	S = analytic_marginal_states(net)

	p_sum_state = np.zeros(len(net._nodes)+1)

	for i in range(N):
		sum_state = sum(id_to_state(net, i))
		p_sum_state[sum_state] += S[i]

	if args.plot:
		# plot distribution over sum of states (show it's bimodal)
		plt.figure()
		plt.plot(p_sum_state)
		plt.title('P(sum of states) for K = %d' % K)
		plt.xlabel('sum of states')
		plt.savefig('plots/p_sum_state_K%d.png' % K)
		plt.close()

	# run sampler to get switching times histogram
	def compute_distribution():
		counter = SwitchedFunction()
		gibbs_sample(net, {}, counter, args.samples, args.burnin)
		return counter.distribution()

	d = load_or_run('sampled_switching_time_distribution_majority_K%d' % K, compute_distribution, force_recompute=args.recompute)

	if args.plot:
		plt.figure()
		plt.bar(np.arange(len(d)), d)
		plt.title('Sampled switching time distributions (majority percept) K=%d' % K)
		plt.savefig('plots/sampled_switching_time_majority_K%d.png' % K)
		plt.close()