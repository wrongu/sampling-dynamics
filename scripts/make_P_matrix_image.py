if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
from counting import construct_markov_transition_matrix
from util import load_or_run

m = 3
p = .964

P = load_or_run('transition_matrix_M%d_p%.3f_noev' % (m, p), lambda: construct_markov_transition_matrix(net))

plt.figure()
plt.imshow(P, interpolation='nearest')
plt.colorbar()
plt.savefig('plots/P_matrix_m%d_p%.3f.png' % (m,p))
plt.close()