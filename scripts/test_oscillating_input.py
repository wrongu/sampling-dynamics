if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from visualize import plot_net_layerwise
import argparse
from models import k_deep_bistable
from generators import alternator
from sampling import gibbs_sample_dynamic_evidence
from util import load_or_run
from counting import *

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', dest='recompute', action='store_true', default=False)
parser.add_argument('--prob', dest='p', type=float, default=0.96)
parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
parser.add_argument('--k-max', dest='k_max', type=int, default=7)
parser.add_argument('--period', dest='T', type=int, default=5)
parser.add_argument('--samples', dest='samples', type=int, default='10000')
parser.add_argument('--burnin', dest='burnin', type=int, default='20')
parser.add_argument('--frames', dest='frames', type=int, default=60)
parser.add_argument('--no-movie', dest='movie', action='store_false', default=True)
parser.add_argument('--res', type=int, default=240)
args = parser.parse_args()

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Mixing Time Animation', artist='Matplotlib', comment='')
writer = FFMpegWriter(fps=15, metadata=metadata)

k_min = 2
layers = range(k_min, args.k_max+1)

for K in layers:
	print K
	net = k_deep_bistable(K, args.p)
	ev_node = net.get_node_by_name('X1')

	def sample_net_response():
		# keep track of state of every node at every sample
		states = np.zeros((args.samples, K))

		def record_sample(i, net):
			states[i,:] = net.state_vector()

		gibbs_sample_dynamic_evidence(
			net,
			alternator([{ev_node: 0}, {ev_node: 1}], period=args.T),
			record_sample,
			args.samples,
			args.burnin)
		return states

	states = load_or_run('sampled_osc_K%d_T%d_S%d' % (K, args.T, args.samples), sample_net_response, force_recompute=args.recompute)

	# compute frequency response for each node
	freq_response = np.fft.fft(states*2-1, axis=0)

	if args.plot:
		plt.figure()
		plt.loglog(np.fliplr(np.abs(freq_response)**2))
		plt.title('Frequency response of nodes to input with T=%d' % args.T)
		plt.xlabel('frequency')
		plt.legend(['X%d' % (K-l+k_min) for l in reversed(layers)], loc='lower left')
		plt.savefig('plots/frequency_response_K%d_T%d.png' % (K, args.T))
		plt.close()

		if args.movie:
			fig = plt.figure()
			ax = fig.add_subplot(1,1,1)
			text_obj = ax.text(.005, 0, "sample 0", fontsize=24, verticalalignment='top')
			with writer.saving(fig, "plots/oscillating_input_animation_K%d_T%d.mp4" % (K, args.T), args.res):
				for i in range(args.frames):
					print "writing frame", i+1
					state_vector = states[i,:]
					net.evidence_from_vector(state_vector)
					plot_net_layerwise(net, colors=state_vector, ax=ax)
					text_obj.set_text("sample %d" % i)
					writer.grab_frame()
