import numpy as np
import pandas as pd
from scipy.signal.windows import gaussian
from behavior_analysis.utilities import find_contiguous
from typing import Union


class BoutDetector:

    def __init__(self, threshold, winsize, frame_rate):
        self.threshold = threshold
        self.winsize = winsize
        self.frame_rate = frame_rate

    def __call__(self, s: Union[pd.Series, np.ndarray], threshold=None, winsize=None, frame_rate=None):
        # Reset parameters
        if threshold:
            self.threshold = threshold
        if winsize:
            self.winsize = winsize
        if frame_rate:
            self.frame_rate = frame_rate
        # Convert to series and save copy of input data
        if isinstance(s, np.ndarray):
            self._s = pd.Series(s)
        else:
            self._s = s.copy()
        # ----------
        # Find bouts
        # ----------
        t = self._s[~pd.isnull(self._s)]  # remove nans
        # Generate kernel
        window = int(self.winsize * self.frame_rate)
        kernel = gaussian(window * 2, (window * 2) / 5.)
        kernel /= np.sum(kernel)
        self._kernel = kernel
        # Find absolute derivative of tail angle trace
        diffed = t.diff().shift(-1)
        diffed.iloc[-1] = 0
        mod_derivative = diffed.abs()
        # Convolve with kernel
        filtered = np.convolve(mod_derivative, self._kernel, mode='same')
        # Find contiguous frames above threshold
        above_threshold = t.index[np.where(filtered > self.threshold)[0]]
        movement_frames = find_contiguous(above_threshold, minsize=window)
        # Remove bouts that span nan data
        contiguous_segments = find_contiguous(t.index)
        first_frames, last_frames = list(zip(*[(frames[0], frames[-1]) for frames in contiguous_segments]))
        self._bouts = [(frames[0], frames[-1]) for frames in movement_frames if
                       (frames[0] not in first_frames) and (frames[-1] not in last_frames)]
        # Save copy of convolved derivative trace
        self._filtered = self._s.copy()
        self._filtered.loc[:] = np.nan
        self._filtered.loc[~pd.isnull(self._s)] = filtered
        # Return first and last frames of each bout
        return self._bouts
