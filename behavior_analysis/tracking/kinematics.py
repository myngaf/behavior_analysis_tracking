from ..multiprocess import MultiProcessing, trackingmethod
from behavior_analysis.utilities import find_contiguous

from pathlib import Path
import numpy as np
from scipy.signal.windows import gaussian
import pandas as pd
import warnings


class VelocityConverter:

    columns = ['speed', 'angular_velocity']

    def __init__(self, fs):
        self.fs = fs

    @staticmethod
    def calculate_speed(xs, ys, fs, scale):
        """Calculates the instantaneous speed in each frame

        Parameters
        ----------
        centres : pd.Series
            The centre of the fish (x, y) coordinate in each frame
        fs : float
            The sampling frequency of the data (frames per second)
        scale : float
            The image scale (the size of one pixel)

        Returns
        -------
        speeds : np.array
            The instantaneous speed in each frame (note: length wil be one less than the input)
        """
        positions = np.array([xs, ys]).T
        vectors = np.diff(positions, axis=0)
        distances = np.linalg.norm(vectors, axis=1)
        speeds = distances * float(scale) * float(fs)
        return speeds

    @staticmethod
    def calculate_angular_velocity(headings, fs):
        """Calculate the instantaneous angular velocity in each frame

        Parameters
        ----------
        headings : pd.Series or np.array
            The heading in each frame (radians)
        fs : float
            The sampling frequency of the data (frames per second)

        Returns
        -------
        angular_velocity : np.array
            The instantaneous angular velocity in each frame (note: length will be one less than the input)
        """
        heading_vectors = np.array([np.cos(headings), np.sin(headings)]).T
        sin_angular_change = np.cross(heading_vectors[:-1], heading_vectors[1:])
        angular_velocity = np.arcsin(sin_angular_change) * float(fs)
        return angular_velocity

    def analyse(self, tracking):
        output = pd.DataFrame(index=tracking.index, columns=self.columns)
        speed = self.calculate_speed(tracking['x'], tracking['y'], self.fs, 1)
        angular_velocity = self.calculate_angular_velocity(tracking['heading'], self.fs)
        output.loc[output.index[:-1], 'speed'] = speed
        output.loc[output.index[:-1], 'angular_velocity'] = angular_velocity
        return output


class EyeTrackingConverter:

    columns = ['left_angle', 'right_angle']

    def __init__(self, fs, smooth_eye_angles=True):
        self.smooth_eye_angles = smooth_eye_angles
        self.fs = fs

    @staticmethod
    def calculate_eye_angles(eye_angles, heading_angles):
        """Calculate the angles of an eye relative to the heading

        First, the angles are converted into unit vectors. For each frame the cross product is calculated between the eye
        vector and the heading vector. The signed angle between the heading vector and eye vector is the arcsin of their
        cross product: a x b = |a||b|sin(theta) -> theta = arcsin(a x b)

        Parameters
        ----------
        eye_angles, heading_angles : pd.Series
            The angle of an eye and the heading over many frames (angles should be in radians)

        Returns
        -------
        corrected_eye_angles : np.array (ndim = 1)
            The corrected angles of the eye for all frames
        """
        eye_vectors = np.array([np.cos(eye_angles), np.sin(eye_angles)])
        heading_vectors = np.array([np.cos(heading_angles), np.sin(heading_angles)])
        corrected_eye_angles = np.arcsin(np.cross(eye_vectors.T, heading_vectors.T))
        return corrected_eye_angles

    def analyse(self, tracking):
        output = pd.DataFrame(index=tracking.index, columns=self.columns)
        smoothed_tracking = tracking.copy()
        for col in self.columns:
            smoothed_tracking.loc[:, col] = smoothed_tracking.loc[:, col].rolling(window=3,
                                                                                  min_periods=0,
                                                                                  center=True).median()
            angles = self.calculate_eye_angles(smoothed_tracking.loc[:, col], smoothed_tracking.loc[:, 'heading'])
            output.loc[:, col] = angles
            if self.smooth_eye_angles:
                window = int(0.1 * self.fs)  # 100ms rolling median
                output.loc[:, col] = output.loc[:, col].rolling(window=window,
                                                                min_periods=0,
                                                                center=True).median()
        return output


class TailTrackingConverter:

    def __init__(self, tail_points, smooth_tail=True, use_headings=True):
        super().__init__()
        self.tail_points = tail_points
        self.smooth_tail = smooth_tail
        self.use_headings = use_headings
        self.k_cols = ['k{}'.format(i) for i in range(self.tail_points.shape[1] - 1)]
        self.tip_columns = self.k_cols[-int(len(self.k_cols) / 5):]
        self.columns = self.k_cols + ['tip', 'length']

    @staticmethod
    def smooth_tail_points(points, size=7, kind='boxcar'):
        """Smooths the position of points along the tail

        Parameters
        ----------
        points : array-like
            An array representing the tail points in each frame, shape (n_frames, n_points, 2)
        size : int, optional
            The width of the filter (number of points to average)
        kind : str {'boxcar', 'gaussian'}, optional
            The shape of the kernel for smoothing points (i.e. how points should be weighted)

        Returns
        -------
        smoothed_points : ndarray
            The smoothed positions of tail points (same shape as input array)

        Notes
        -----
        This function returns smooths points spatially, not temporally
        """
        if kind == 'boxcar':
            kernel = np.ones((size,))
        elif kind == 'gaussian':
            kernel = gaussian(size, 1)
        else:
            raise ValueError()
        kernel /= np.sum(kernel)
        n = int((size - 1) / 2)
        padded_points = np.pad(points, ((0, 0), (n, n), (0, 0)), 'edge')
        smoothed_points = np.apply_along_axis(np.convolve, 1, padded_points, kernel, mode='valid')
        return smoothed_points

    def calculate_tail_curvature(self, points, headings=None, **kwargs):
        """Converts tail points to a 1D vector of angles for each frame after smoothing with gaussian filter

        In each frame, the tail is essentially rotated and centred such that first point is at the origin and the fish
        is facing right along the x-axis. Then, tangents are approximated between pairs of adjacent points. The shape of the
        tail is defined by the angles at which each of these tangents intersect the x-axis.

        Parameters
        ----------
        points : np.array
            Array representing the position of points along the tail within an image, shape = (n_frames, n_points, 2)
        headings : np.array
            Array representing the heading of the fish in each frame (radians), shape = (n_frames,)

        Returns
        -------
        ks, tail_lengths : np.array
            Tail angles and length of the tail in each frame

        Notes
        -----
        Since tail angles are calculated from tangents drawn between successive pairs of points in each frame, the length of
        the vector that defines the shape of the tail will be one less than the number of points fitted to the tail (i.e.
        fitting 51 points to the tail yields a 50 dimensional vector that describes its shape).

        Representing the tail this way requires the heading to the accurately known and points to be equally spaced along
        the tail.
        """
        # Smooth tail points
        smoothed_points = self.smooth_tail_points(points, **kwargs)
        # Compute the vectors for each tail segment
        vs = np.empty(smoothed_points.shape)
        vs[:, 1:] = np.diff(smoothed_points, axis=1)
        if headings is not None:  # compute curvature relative to fish heading
            headings_r = headings + np.pi
            vs[:, 0] = np.array([np.cos(headings_r), np.sin(headings_r)]).T
        else:  # compute curvature relative to average vector
            vs[:, 0] = np.mean(vs[:, 1:], axis=1)
        # Tail segment lengths
        ls = np.linalg.norm(vs, axis=2)
        # Compute angles between successive tail segments from the arcsin of cross products
        crosses = np.cross(vs[:, :-1], vs[:, 1:])
        crosses /= (ls[:, :-1] * ls[:, 1:])
        dks = np.arcsin(crosses)
        # Cumulative sum angle differences between segments
        ks = np.cumsum(dks, axis=1)
        # Sum tail segments lengths
        tail_lengths = np.sum(ls[:, 1:], axis=1)
        return ks, tail_lengths

    def analyse(self, tracking):
        output = pd.DataFrame(index=tracking.index, columns=self.columns)
        points = self.tail_points[tracking.index]
        # CALCULATE TAIL ANGLES
        heading = tracking['heading']
        if self.use_headings:
            ks, tail_lengths = self.calculate_tail_curvature(points, heading)
        else:
            ks, tail_lengths = self.calculate_tail_curvature(points)
        output.loc[:, self.k_cols] = np.array(ks)
        if self.smooth_tail:
            output.loc[:, self.k_cols] = output.loc[:, self.k_cols].rolling(window=3, min_periods=0,
                                                                            center=True).median()
        output.loc[:, 'tip'] = output.loc[:, self.tip_columns].apply(np.mean, axis=1)
        output.loc[:, 'length'] = tail_lengths
        return output


class Kinematics(MultiProcessing):

    def __init__(self, n_processes):
        super().__init__(n_processes=n_processes)

    @trackingmethod()
    def run(self,
            tracking_path,
            output_path,
            tail_tracking=True,
            fs=500.,
            min_tracked_length=0.1,
            smooth_eye_angles=True,
            smooth_tail=True,
            use_headings=True):
        """Generate kinematic data for a tracked video.

        Parameters
        ----------
        tracking_path : Path
            Path to a .csv file containing tracking data

        points_path : str
            Path to a .npy file containing tail points

        fs : float
            Sampling frequency (frames per second)

        min_tracked_length : float, optional (default = 0.1)
            The minimum length of time for which continuously tracked data are available to calculate kinematics (seconds)

        smooth_eye_angles : bool, optional (default = True)
            Whether to apply a 100 ms sliding median to the eye angle data (edge-preserving smoothing of eye angles)

        smooth_tail : bool, optional (default = True)
            Whether to apply a 3 frame sliding median to the tail tracking data (removes single frame noise from tracking)

        save_output : bool, optional (default = False)
            Whether to save the output

        output_path : str or None, optional (default = None)
            The output path (.csv) for saving kinematics if save_output = True

        Returns
        -------
        if save_output = False:
            kinematics : pd.DataFrame
                DataFrame containing kinematic data. Columns:
                    'k0' - 'k(n-1)' : angle of tangents between n successive tail points
                    'tip' : the average curvature (tail angle) over the last 20% of the tail
                    'length' : the length of the tail (useful for finding tracking errors)
                    'left' : the angle of the left eye relative to the heading
                    'right' : the angle of the right eye relative to the heading
                    'speed' : the instantaneous speed in each frame
                    'angular_velocity' : the instantaneous angular velocity in each frame
                    'tracked' : whether kinematic data exists from the frame
        if save_output = True:
            analysis_time : float
                The time it took to perform the analysis (seconds).

        """

        # create analysis components
        analysis_components = [VelocityConverter(fs)]

        # open tracking data
        tracking_df = pd.read_csv(tracking_path, dtype=dict(tracked=bool))

        # add eye tracking
        if 'left_x' in tracking_df.columns:
            analysis_components.append(EyeTrackingConverter(fs, smooth_eye_angles=smooth_eye_angles))

        # add tail tracking
        if tail_tracking:
            tracking_path = Path(tracking_path)
            points_path = tracking_path.parent.joinpath(tracking_path.stem + '.npy')
            if points_path.exists():
                tail_points = np.load(points_path)
                analysis_components.append(TailTrackingConverter(tail_points,
                                                                 smooth_tail=smooth_tail,
                                                                 use_headings=use_headings))
            else:
                warnings.warn(f'Points path {points_path} does not exist.')

        # create output data frame
        columns = []
        for component in analysis_components:
            columns.extend(component.columns)
        columns.append('tracked')
        kinematics = pd.DataFrame(index=tracking_df.index, columns=columns)
        kinematics['tracked'] = False

        # find tracked segments
        tracked_frames = tracking_df[tracking_df['tracked']]
        tracked_segments = find_contiguous(tracked_frames.index, 1, int(min_tracked_length * fs))

        for segment_frames in tracked_segments:
            first, last = segment_frames[0], segment_frames[-1]
            segment = tracking_df.loc[first:last, :].copy()  # make a copy of the tracked segment
            segment.loc[:, 'heading'] = segment.loc[:, 'heading'].rolling(window=3, min_periods=0, center=True).median()
            for component in analysis_components:
                kinematics.loc[first:last, component.columns] = component.analyse(segment)
            kinematics.loc[first:last, 'tracked'] = True

        # save
        output_path = Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        kinematics.to_csv(output_path, index=False)
