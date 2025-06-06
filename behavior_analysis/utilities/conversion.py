import numpy as np


def array_to_point(p):
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
    >>> array_to_point(p)
    (1, 3)
    >>> q = (4.5, 8.1)
    >>> array_to_point(q)
    (4, 8)
    >>> r = [0.1, 0.7]
    >>> array_to_point(r)
    (0, 0)
    """
    return tuple(np.int32(p))


def angle_to_vector(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def euclidean_to_polar(xy):
    """Converts xy coordinates to polar coordinates"""
    r = np.linalg.norm(xy, axis=1)
    th = np.arctan2(xy[:, 1], xy[:, 0])
    return np.vstack([r, th]).T
