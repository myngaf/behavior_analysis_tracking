import numpy as np
import pandas as pd
import time
import cv2


def print_heading(s):
    print '=' * len(s)
    print s
    print '=' * len(s)
    print ''


def print_subheading(s):
    print s
    print '-' * len(s)


class Timer(object):

    def __init__(self):
        self.times = []

    def start(self):
        self.times = [time.time()]
        return self.times[0]

    def stop(self):
        self.times.append(time.time())
        return self.times[-1] - self.times[0]

    def lap(self):
        self.times.append(time.time())
        return self.lap_times[-1]

    @property
    def lap_times(self):
        return np.diff(self.times)

    @property
    def lap_time(self):
        return time.time() - self.times[-1]

    @property
    def time(self):
        return time.time() - self.times[0]

    @property
    def average(self):
        return np.mean(self.lap_times)

    @staticmethod
    def convert_time(t):
        if t < 60:
            return '{} seconds'.format(t)
        elif t < 3600:
            return '{} minutes'.format(t / 60.)
        else:
            return '{} hours'.format(t / 3600.)


def yes_no_question(q):
    """Asks the user a yes/no question

    Valid affirmative answers (case insensitive): 'y', 'yes', '1', 't', 'true'
    Valid negative answers (case insensitive): 'n', 'no', '0', 'f', 'false'

    Parameters
    ----------
    q : str
        A question that the user can answer in the command line

    Returns
    -------
    bool : True for affirmative answers, False for negative answers

    Raises
    ------
    ValueError
        If unrecognised answer given
    """
    affirmative_answers = ['y', 'yes', '1', 't', 'true']
    negative_answers = ['n', 'no', '0', 'f', 'false']
    answer = raw_input(q + ' [y/n] ')
    if answer.lower() in affirmative_answers:
        return True
    elif answer.lower() in negative_answers:
        return False
    else:
        raise ValueError('Invalid answer! Recognised responses: {}'.format(affirmative_answers + negative_answers))


class KeyboardInteraction(object):
    """Class for handling keyboard interaction

    Class Attributes
    ----------------
    enter : 13
        ASCII code for enter/carriage return key
    esc : 27
        ASCII code for escape key
    space : 32
        ASCII code for escape key"""

    enter_key = 13
    esc_key = 27
    space_key = 32

    def __init__(self):
        self.k = None

    def wait(self, t=0):
        self.k = cv2.waitKey(t)

    def enter(self):
        return self.k == self.enter_key

    def space(self):
        return self.k == self.space_key

    def esc(self):
        return self.k == self.esc_key

    def valid(self):
        """Checks whether the enter, escape or space keys were pressed

        Parameters
        ----------
        k : int
            ASCII key code

        Returns
        -------
        bool
        """
        if self.k in (self.enter_key, self.esc_key, self.space_key):
            return True
        else:
            return False


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
    idxs = np.array([np.arange(i, i + n) for i in range(1 + len(a) - n) if np.all(a[i:i + n] == seq)])
    return idxs


def read_csv(path, **kwargs):
    """Opens a csv file as a DataFrame

    Parameters
    ----------
    path : str
        The path to a csv file.

    kwargs : dict, optional
        (Column, callable) pairs for reading the csv file.

    Returns
    -------
    df : pd.DataFrame
        A correctly formatted DataFrame.
    """
    df = pd.read_csv(path)
    for col, read_as in kwargs.iteritems():
        existing_data = df[col][~df[col].isnull()]
        df.loc[existing_data.index, col] = existing_data.apply(read_as)
    return df


def array2point(p):
    """Converts an (x, y) coordinate pair into an integer tuple for openCV draw functions

    Parameters
    ----------
    p : tuple, list or array-like
        An point's (x, y) coordinates

    Returns
    -------
    tuple
        The point reformatted as an tuple of integers compatible with drawing in openCV

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([1.2, 3.7])
    >>> array2point(p)
    (1, 3)
    >>> q = (4.5, 8.1)
    >>> array2point(q)
    (4, 8)
    >>> r = [0.1, 0.7]
    >>> array2point(r)
    (0, 0)
    """
    return tuple(np.int32(p))
