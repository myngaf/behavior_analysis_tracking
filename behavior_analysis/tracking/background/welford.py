import numpy as np
import cv2
import time


class Welford:
    """Adapted from: https://gist.github.com/alexalemi/2151722

    Implements Welford's algorithm for computing a running mean
    and standard deviation as described at:
        http://www.johndcook.com/standard_deviation.html

    can take single values or iterables

    Properties:
        mean    - returns the mean
        std     - returns the std
        meanfull- returns the mean and std of the mean

    Usage:
        >>> foo = Welford()
        >>> foo(range(100))
        >>> foo
        <Welford: 49.5 +- 29.0114919759>
        >>> foo([1]*1000)
        >>> foo
        <Welford: 5.40909090909 +- 16.4437417146>
        >>> foo.mean
        5.409090909090906
        >>> foo.std
        16.44374171455467
        >>> foo.meanfull
        (5.409090909090906, 0.4957974674244838)
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M) * 1. / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

    def __call__(self, x):
        self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def meanfull(self):
        return self.mean, self.std / np.sqrt(self.k)

    @property
    def std(self):
        if self.k == 1:
            return 0
        return np.sqrt(self.S / (self.k - 1))

    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)


def background_statistics_welford(*args):
    """This is approx. three times slower!"""
    accumulator = Welford()
    for arg in args:
        now = time.time()
        print(arg.name, end=' ')
        cap = cv2.VideoCapture(str(arg))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in range(n_frames):
            ret, frame = cap.read()
            if ret:
                accumulator(frame[..., 0])
        cap.release()
        print(time.time() - now)
    return accumulator.mean, accumulator.std
