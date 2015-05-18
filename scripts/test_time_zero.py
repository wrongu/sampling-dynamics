if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import matplotlib.pyplot as plt
from sys import argv
from visualize import plot_net_layerwise
from models import m_deep_bistable, m_wide_bistable
from switching_times import sample_recently_switched_states, analytic_switching_times, sampled_switching_times
from util import load_or_run
from counting import *

model = m_deep_bistable
model_nm = 'deep'
if '--model' in argv:
	if len(argv) > argv.index('--model'):
		m = argv[argv.index('--model')+1].lower() 
		if m == 'wide':
			model = m_wide_bistable
			model_nm = 'wide'

for M in range(2,7):
	print M

	net = model(M, 0.96)
	S1 = load_or_run('compare_init_S1_%s_%02d.npy' % (model_nm, M), lambda: sample_marginal_states(net, {}, 10000, conditional_fn=lambda net: plurality_state(net) == 0))
	S2 = load_or_run('compare_init_S2_%s_%02d.npy' % (model_nm, M), lambda: sample_marginal_states(net, {net._nodes[0]: 0}, 10000))
	S3 = load_or_run('compare_init_S3_%s_%02d.npy' % (model_nm, M), lambda: sample_recently_switched_states(net, lambda n: n._nodes[0].state_index(), max_iterations=100000))

	colors1 = mean_state(net, S1)
	colors2 = mean_state(net, S2)
	colors3 = mean_state(net, S3)

	colormap = u'afmhot'

	fig = plt.figure()
	fig.add_subplot(131)
	plot_net_layerwise(net, colors=colors1)
	plt.title('plurality')
	fig.add_subplot(132)
	plot_net_layerwise(net, colors=colors2)
	plt.title('XM clamped')
	fig.add_subplot(133)
	plot_net_layerwise(net, colors=colors3)
	plt.title('Just-Switched')

	plt.savefig('plots/init_colors_%s_%02d.png' % (model_nm, M))
	plt.close()

###########################################################################################################################
# try switching time (empirical and predicted) with different initializations for the predicted ones

for M in range(2,7):
	print M
	net = model(M, 0.96)
	S_xm = load_or_run('compare_init_S2_%s_%02d.npy' % (model_nm, M), lambda: sample_marginal_states(net, {net._nodes[0]: 0}, 10000))
	S_sw = load_or_run('compare_init_S3_%s_%02d.npy' % (model_nm, M), lambda: sample_recently_switched_states(net, lambda n: n._nodes[0].state_index(), max_iterations=100000))

	analytic_xm = analytic_switching_times(net, S_xm, {net._nodes[0]: 1}, max_t=None, eps=1e-4)
	analytic_sw = analytic_switching_times(net, S_sw, {net._nodes[0]: 1}, max_t=None, eps=1e-4)
	empirical   = load_or_run('sampled_switching_times_M%02d.npy' % M, lambda: sampled_switching_times(net, {net._nodes[0]: 1}, trials=5000))

	fig = plt.figure()
	plt.title('Compare Switching Time Models, %d-%s net' % (M, model_nm))
	plt.plot(analytic_xm)
	plt.plot(analytic_sw)
	plt.plot(empirical)
	plt.legend(['An_In_Xm', 'An_In_Sw', 'Emp_Sw'])
	plt.savefig('plots/cmp_st_models_%s_%02d.png' % (model_nm, M))
	plt.close()