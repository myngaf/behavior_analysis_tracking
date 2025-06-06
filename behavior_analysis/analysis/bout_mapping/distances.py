import numpy as np
from joblib import Parallel, delayed

from .dynamic_time_warping import DynamicTimeWarping


def fill_row(*bouts, **kwargs):
    s = bouts[0]
    bw = kwargs.get('bw', 0.01)
    fs = kwargs.get('fs', 500.)
    dtw = DynamicTimeWarping(s, bw, fs)
    if kwargs.get('flip', False):
        row = np.array([dtw.align(-t) for t in bouts[1:]])
    else:
        row = np.array([dtw.align(t) for t in bouts[1:]])
    return row


def fill_row_min(*bouts, **kwargs):
    s = bouts[0]
    bw = kwargs.get('bw', 0.01)
    fs = kwargs.get('fs', 500.)
    dtw = DynamicTimeWarping(s, bw, fs)
    row = np.array([min(dtw.align(t), dtw.align(-t)) for t in bouts[1:]])
    return row


def calculate_distance_matrix_templates(bouts, templates, fs=500., bw=0.01, parallel_processing=True, n_processors=-1):
    bouts = list(bouts)
    templates = list(templates)
    bouts_by_row = [[bout] + templates for bout in bouts]
    if parallel_processing:
        distances = Parallel(n_processors)(delayed(fill_row_min)(*row, fs=fs, bw=bw) for row in bouts_by_row)
    else:
        distances = [fill_row_min(*row, fs=fs, bw=bw) for row in bouts_by_row]
    D = np.array(distances)
    return D


def calculate_distance_matrix(bouts, fs=500., bw=0.01, flip=False, parallel_processing=True, n_processors=-1):
    bouts = list(bouts)
    bouts_by_row = [bouts[i:] for i in range(len(bouts) - 1)]
    if parallel_processing:
        distances = Parallel(n_processors)(delayed(fill_row)(*row, fs=fs, bw=bw, flip=flip) for row in bouts_by_row)
    else:
        distances = [fill_row(*row, fs=fs, bw=bw, flip=flip) for row in bouts_by_row]
    D = np.array([d for row in distances for d in row])
    return D
