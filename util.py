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

# normalize an ndarray on the specified axis
# thanks to http://stackoverflow.com/a/21032099/1935085
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
