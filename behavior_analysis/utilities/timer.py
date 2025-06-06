import numpy as np
import time


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
            return f'{t} seconds'
        elif t < 3600:
            return f'{t / 60.} minutes'
        else:
            return f'{t / 3600.} hours'


def timedmethod(method):
    """Decorator for timing method calls."""
    def timed_wrapper(self, *args, **kwargs):
        timer = Timer()
        timer.start()
        result = method(self, *args, **kwargs)
        run_time = timer.stop()
        return result, run_time
    return timed_wrapper


def timed(func):
    """Decorator for timing function calls."""
    def timed_wrapper(*args, **kwargs):
        timer = Timer()
        timer.start()
        result = func(*args, **kwargs)
        run_time = timer.stop()
        return result, run_time
    return timed_wrapper
