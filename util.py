import numpy as np
import os

def load_or_run(filename, function, subdir='saved_data', force_recompute=False):
	if filename[-4:] != '.npy':
		filename += '.npy'
	filename = os.path.join(subdir, filename)
	if force_recompute or not os.path.isfile(filename):
		data = function()
		np.save(filename, data)
	else:
		data = np.load(filename)
	return data