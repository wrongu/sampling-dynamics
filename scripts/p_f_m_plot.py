if __name__ == '__main__' and __package__ is None:
	__package__ = 'scripts'

import numpy as np
import matplotlib.pyplot as plt
from models import compute_p_for_given_marginal

# plot M vs p vor varying P

Ps = [0.9, 0.8, 0.7]
Ms = range(2,7)

plt.figure()
for P in Ps:
	ps = np.array([compute_p_for_given_marginal(m,P) for m in Ms])
	plt.plot(Ms,ps,'-o')
plt.xlim([0,7])
plt.legend(['P=%.1f' % P for P in Ps], loc='upper left')
plt.xlabel('Model depth')
plt.ylabel('rho')
plt.savefig('plots/p_f_m.png')
plt.close()