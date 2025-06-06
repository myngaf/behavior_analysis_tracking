from behavior_analysis.tracking.background_subtraction import BackgroundSubtractor
from .detectors import FishDetector, FeatureDetector, TrackingError
from .tail_tracker import TailTracker
from ..multiprocess import MultiProcessing, trackingmethod
from video_analysis_toolbox.video import Video
from behavior_analysis.utilities import array_to_point
from behavior_analysis.tracking.kalman import FishKalmanFilter

import cv2
import numpy as np
import pandas as pd
from pathlib import Path


class Tracker(MultiProcessing):
    """Class for handling fish tracking.

    Parameters
    ----------
    background : np.ndarray
        Background image
    threshold_1 : int
        Threshold for finding fish contour
    threshold_2 : int
        Threshold for finding eyes and swim bladder
    track_tail : bool
        Whether or not to perform tail tracking
    n_points : int
        Number of points to fit to tail
    normalize : bool
        Whether or not to perform normalization of the image after background subtraction
    foreground : str {'dark' | 'light'}
        Whether foreground objects are darker or lighter than the background
    n_processes : int
        Number of processes to run in parallel

    Attributes
    ----------
    background_subtractor : BackgroundSubtractor
        Holds the background subtractor object
    fish_detector : FishDetector
        Holds the fish detector object
    feature_detector : FeatureDetector
        Holds the feature detector object
    tail_tracker : TailTracker
        Holds the tail tracker object (None if no tail tracking)
    """

    columns = ('x', 'y', 'heading', 'left_x', 'left_y', 'left_angle', 'right_x', 'right_y', 'right_angle', 'tracked')

    def __init__(self,
                 background: np.ndarray,
                 bg_mask: np.ndarray,
                 threshold_1: int = 10,
                 threshold_2: int = 200,
                 track_tail: bool = True,
                 n_points: int = 51,
                 normalize: bool = True,
                 foreground: str = 'dark',
                 n_processes: int = 4):
        super().__init__(n_processes=n_processes)
        self.background_subtractor = BackgroundSubtractor(background, bg_mask, normalize=normalize, foreground=foreground)
        self.fish_detector = FishDetector(threshold_1)
        self.feature_detector = FeatureDetector(threshold_2)
        if track_tail:
            self.tail_tracker = TailTracker(n_points)
        else:
            self.tail_tracker = None

    @classmethod
    def from_background_path(cls, path, path_mask, threshold_1=10, threshold_2=200, **kwargs):
        background = cv2.imread(str(path), 0)
        bg_mask = cv2.imread(str(path_mask), 0)
        return cls(background, bg_mask, threshold_1, threshold_2, **kwargs)

    def find_contours(self, image):
        sub = self.background_subtractor.process(image)
        fish_contour, fish_mask, fish_masked = self.fish_detector.find_masked(sub, equalize=True)
        feature_contours = self.feature_detector.find_contours(fish_masked)
        return fish_masked, fish_contour, feature_contours

    def track_points(self, contours):
        contour_info = [self.feature_detector.contour_info(c) for c in contours]
        features = self.feature_detector.assign_features(*contour_info)
        return features

    @staticmethod
    def draw_mask(image, bg_mask):
        masked = image.copy()
        mask = bg_mask.copy()
        mask = cv2.multiply(mask, 255, dtype=cv2.CV_8U)
        masked = cv2.bitwise_and(masked, masked, mask=mask)
        return masked

    @staticmethod
    def draw_contours(image, fish_contour, feature_contours):
        contoured = image.copy()
        cv2.drawContours(contoured, [fish_contour], 0, 0, 1)
        cv2.drawContours(contoured, feature_contours, -1, 255, 1)
        return contoured

    @staticmethod
    def draw_tracking(image, heading, sb, left, right, **kwargs):
        img = image.copy()
        if image.ndim < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        colors = dict(k=0, w=(255, 255, 255), r=(255, 0, 0), g=(0, 255, 0), b=(0, 0, 255), y=(255, 255, 0))
        # plot heading
        heading_vector = np.array(heading[:2])
        centre = sb[:2]
        cv2.circle(img, array_to_point(centre), 3, colors['y'], -1)
        cv2.line(img, array_to_point(centre), array_to_point(centre + (80 * heading_vector)), colors['y'], 2)
        # plot eyes
        left_v = np.array([np.cos(left.angle), np.sin(left.angle)])
        right_v = np.array([np.cos(right.angle), np.sin(right.angle)])
        for p, v, c in zip((left[:2], right[:2]), (left_v, right_v), ('g', 'r')):
            cv2.line(img, array_to_point(p - (20 * v)), array_to_point(p + (20 * v)), colors[c], 2)
        # plot tail points
        for p in kwargs['tail_points']:
            cv2.circle(img, array_to_point(p), 1, colors['b'], -1)
        return img

    @staticmethod
    def convert_to_tracking_output(sb, heading, left, right, **kwargs):
        output = [sb.x, sb.y, heading.angle, left.x, left.y, left.angle, right.x, right.y, right.angle, 1]
        return output

    @property
    def empty_tracking_output(self):
        output = np.empty(len(self.columns)) + np.nan
        output[-1] = 0
        return output

    @trackingmethod()
    def run(self, video_path, output_path):

        # Open the video
        video = Video.open(video_path)
        # Assign memory for storing tracking output
        tracking_output = np.zeros((video.frame_count, 10))
        if self.tail_tracker:
            tail_tracking = np.zeros((video.frame_count, self.tail_tracker.n_points, 2))

        # Iterate through frames
        for f in range(video.frame_count):
            frame = video.advance_frame()  # grab frame
            if np.any(frame):  # frame exists
                try:
                    sub = self.background_subtractor.process(frame)  # subtract background
                    fish_contour, fish_mask, fish_masked = self.fish_detector.find_masked(sub, equalize=True)
                    feature_contours = self.feature_detector.find_contours(fish_masked)
                    tracked_features = self.track_points(feature_contours)  # track features
                    tracking = self.convert_to_tracking_output(**tracked_features)  # convert to correct output
                except TrackingError:  # could not find contours
                    tracking = self.empty_tracking_output
            else:  # frame does not exist
                tracking = self.empty_tracking_output
            if self.tail_tracker:
                if tracking[-1]:  # tracking data exists for frame
                    centre = (tracked_features['sb'].x, tracked_features['sb'].y)
                    tail_points = self.tail_tracker.track_tail(fish_mask, centre)
                    tail_tracking[f] = tail_points
                else:  # tracking failed
                    tail_tracking[f] = np.nan
            tracking_output[f] = tracking

        # Save
        output = Path(output_path)
        parent, name = output.parent, output.stem
        if not parent.exists():
            parent.mkdir(parents=True)
        # save tracking
        tracking = pd.DataFrame(tracking_output, columns=self.columns)
        tracking.to_csv(parent.joinpath(name + '.csv'), index=False)
        if self.tail_tracker:
            np.save(parent.joinpath(name + '.npy'), tail_tracking)


class FrameAligner(Tracker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def rotate_bound(image, centre, angle):
        """Adapted from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/"""
        # grab the dimensions of the image
        (h, w) = image.shape[:2]
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D(tuple(centre), np.degrees(angle), 1)
        cos, cs = np.abs(M[0, 0]), np.sign(M[0, 0])
        sin, ss = np.abs(M[0, 1]), np.sign(M[0, 1])
        # compute the new bounding dimensions of the image
        fval = int(((cs * ss) + 1) // 2)
        width = lambda p: ([w - p[0], p[0]][fval] * cos) + (p[1] * sin)
        height = lambda p: ([p[1], h - p[1]][fval] * cos) + (p[0] * sin)
        w2 = max(width(centre), width((w - centre[0], h - centre[1])))
        h2 = max(height(centre), height((w - centre[0], h - centre[1])))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += w2 - centre[0]
        M[1, 2] += h2 - centre[1]
        # perform the actual rotation and return the image
        try:
            return cv2.warpAffine(image, M, (int(2 * w2), int(2 * h2)))
        except cv2.error:
            print(image.shape, M, w2, h2)
            import sys
            sys.exit()

    @trackingmethod()
    def run(self, video_path, output_path):

        # Open the video
        video = Video.open(video_path)

        kalman_filter = None
        frames = []
        tracking = np.zeros((video.frame_count, 7))

        # Iterate through frames
        for f in range(video.frame_count):
            frame = video.advance_frame()  # grab frame
            if np.any(frame):  # frame exists

                # Subtract background
                sub = self.background_subtractor.process(frame)
                # Find fish
                contour = self.fish_detector.find_contour(sub)
                # Crop to fish
                cropped, p1, p2 = self.fish_detector.crop_to_contour(sub, contour, (10, 10))

                # Calculate position and heading
                x, y, th = self.fish_detector.contour_info(contour)
                z = np.array([x, y, th])
                tracking[f, :3] = z

                # Apply kalman filter
                if f:
                    kalman_filter.filter(z)
                    zk = kalman_filter.xFiltered[-1]
                else:
                    kalman_filter = FishKalmanFilter(z)
                    zk = z.copy()
                tracking[f, 3:6] = zk
                tracking[f, 6] = 1

                # Rotate and center fish in image
                c = zk[:2] - p1
                aligned = self.rotate_bound(cropped, c, zk[2])
                frames.append(aligned)

            else:  # frame does not exist
                tracking[f, :6] = np.nan
                frames.append(np.empty((0, 0), dtype='uint8'))

        # Pad frames to largest size
        shape = np.max([a.shape for a in frames], axis=0)
        aligned_frames = np.zeros((len(frames), shape[0], shape[1]), dtype='uint8')
        for f, frame in enumerate(frames):
            pad = shape - frame.shape
            pad_width = pad // 2
            pad_width = np.array([pad_width, pad - pad_width]).T
            padded = np.pad(frame, pad_width, 'constant', constant_values=0)
            aligned_frames[f] = padded.astype('uint8')

        # Save
        output = Path(output_path)
        parent, name = output.parent, output.stem
        if not parent.exists():
            parent.mkdir(parents=True)
        # save frames
        np.save(parent.joinpath(name + '.npy'), aligned_frames)
        # save tracking
        tracking = pd.DataFrame(tracking, columns=('x_tracked', 'y_tracked', 'heading_tracked',
                                                   'x_kalman', 'y_kalman', 'heading_kalman', 'tracked'))
        tracking.to_csv(parent.joinpath(name + '.csv'), index=False)
