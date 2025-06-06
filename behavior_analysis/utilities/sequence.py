import numpy as np


def find_contiguous(data, stepsize=1, minsize=1):
    """Finds continuous sequences in a list or array

    Parameters
    ----------
    data : list like
        A sequence of numbers
    stepsize : int, optional (default = 1)
        The maximum allowed step between numbers in the continuous sequences
    minsize : int, optional (default = 1)
        The minimum length of a contiguous sequence within the data

    Returns
    -------
    output : list
        List of arrays of continuous sequences within the data
    """
    runs = np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)
    output = [run for run in runs if len(run) >= minsize]
    return output


def find_subsequence(a, seq):
    """Finds all the occurrences of a sub-sequence (seq) within a longer array (a)

    Parameters
    ----------
    a : array-like
        The array to be searched
    seq : array-like
        A sequence of numbers

    Returns
    -------
    idxs : numpy array
        The indices of all occurrences of target sub-sequence (seq) within the array (a)
    """
    n = len(seq)
    idxs = np.array([np.arange(i, i + n) for i in np.arange(1 + len(a) - n) if np.all(a[i:i + n] == seq)])
    return idxs
