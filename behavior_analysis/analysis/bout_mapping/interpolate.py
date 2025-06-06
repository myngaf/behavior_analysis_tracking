import numpy as np
from scipy.interpolate import splprep, splev


def interpolate_nd(x, fs, fs_new):
    t = np.arange(len(x)) / float(fs)
    t_new = np.arange(0, t[-1] * fs_new) / float(fs_new)
    tck, u = splprep(x.T, u=t, s=0)
    x_new = splev(t_new, tck, der=0)
    return np.array(x_new).T
