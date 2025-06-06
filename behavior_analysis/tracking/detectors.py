from video_analysis_toolbox.image_processing import contours, cropping
from video_analysis_toolbox.utilities import TrackingError, feature_vector
from behavior_analysis.utilities import array_to_point

import numpy as np
import cv2
from scipy.spatial.distance import pdist


class FishDetector(contours.ContourDetector, cropping.Cropper):
    """Finding fish in an image"""

    def __init__(self, threshold):
        super().__init__(threshold, 1)  # n=1 -> largest contour only

    def find_contour(self, image):
        """Finds the largest contour in the image

        Parameters
        ----------
        image : array like
            Unsigned 8-bit integer array

        Returns
        -------
        contour : np.array
            The largest contour in the image
        """
        contours = super().find_contours(image)
        try:
            return contours[0]
        except IndexError:
            raise TrackingError("Cannot detect fish with given threshold.")

    def find_masked(self, image, equalize=False):
        fish = self.find_contour(image)
        mask, masked = self.mask(image, [fish], equalize=equalize)
        return fish, mask, masked

    def crop_to_fish(self, image, pad=(0, 0)):
        """Crops an image to the bounding box of its largest contour

        Parameters
        ----------
        image : array like
            Unsigned 8-bit integer array
        pad : tuple (default = (0, 0))
            The number of additional pixels around the bounding box of the contour to include (x_pad, y_pad)

        Returns
        -------
        cropped : np.array
            Cropped image
        fish : np.array
            Array representing contour of the fish (in coordinates of original image)
        """
        fish = self.find_contour(image)
        cropped, p1, p2 = self.crop_to_contour(image, fish, pad)
        return cropped, fish


class FeatureDetector(contours.ContourDetector):
    """Finds fish features (eyes and swim bladder) within an image"""

    def __init__(self, threshold):
        super().__init__(threshold, 3)  # n=3 -> three largest contours (corresponding to eyes and swim bladder)

    def watershed(self, image, contours):
        """Watershed algorithm for finding the centres and angles of the eyes and swim bladder if simple thresholding fails

        The algorithm works by first fitting a triangle that encloses all the points in the internal contours of the fish.
        Using this triangle, the approximate locations of the eyes and swim bladder are calculated. These approximate
        locations are used as seeds for a watershed on the original background-subtracted image. The watershed marks
        contiguous areas of the image belonging to the same feature, from which a contour is calculated and the its centre
        and angle.

        The function is considerably slower at finding the internal features than straightforward thresholding. However, it
        is useful for when two contours fuse for a couple of frames in a recording, as occasionally happens during tracking.
        The function can still work when the fish rolls, however in these cases the eye tracking tends to be very inaccurate.
        Nonetheless, it is still useful for approximating the heading of the fish in such cases.

        Parameters
        ----------
        image : array-like
            Unsigned 8-bit integer array representing a background-subtracted image
        contours : list
            The contours that were found after applying a threshold and finding contours

        Returns
        -------
        centres, angles : np.array
            Arrays representing the centres, shape (3, 2), and angles, shape (3,), of internal features

        Raises
        ------
        TrackingError
            If any error is encountered during the watershed process. Errors tend to occur if contours is an empty list, or
            if a cv2 error is encountered when trying to calculate the minEnclosingTriangle.

        References
        ----------
        Uses slightly modified version of the watershed algorithm here:
        http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
        """
        try:
            # find the minimum enclosing triangle for the contours
            internal_points = np.concatenate(contours, axis=0)
            ret, triangle_points = cv2.minEnclosingTriangle(internal_points)
            triangle_points = np.squeeze(triangle_points)
            # find approximate locations of the features
            triangle_centre = np.mean(triangle_points, axis=0)
            estimated_feature_centres = (triangle_points + triangle_centre) / 2
            sure_fg = np.zeros(image.shape, np.uint8)
            for c in estimated_feature_centres:
                contour_check = np.array([cv2.pointPolygonTest(cntr, array_to_point(c), False) for cntr in contours])
                if np.all(contour_check == -1):
                    internal_points = np.squeeze(internal_points)
                    distances = np.linalg.norm(internal_points - c, axis=1)
                    c = internal_points[np.argmin(distances)]
                cv2.circle(sure_fg, array_to_point(c), 1, 255, -1)
            # watershed
            unknown = np.zeros(image.shape, np.uint8)
            cv2.drawContours(unknown, contours, -1, 255, -1)
            unknown = cv2.morphologyEx(unknown, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)
            unknown[sure_fg == 255] = 0
            ret, markers = cv2.connectedComponents(sure_fg, connectivity=4)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
            # calculate contour features
            new_contours = []
            for i in range(2, 5):
                contour_mask = np.zeros(image.shape, np.uint8)
                contour_mask[markers == i] = 255
                img, cntrs, hierarchy = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                new_contours.append(cntrs[0])
            return new_contours
        except Exception:
            raise TrackingError("Unable to track frame using watershed method.")

    def find_contours(self, image):
        """Finds the three largest contours in the image at the specified threshold

        Parameters
        ----------
        image : array like

        Returns
        -------
        contours : list
            List of contours
        """
        contours = super().find_contours(image)
        if 0 < len(contours) < 3:  # if three contours cannot be identified at specified threshold, try watershed
            contours = self.watershed(image, contours)
        elif len(contours) == 0:  # raise a tracking error if no contours can be identified at the given threshold
            raise TrackingError("Cannot find any contours with the given threshold.")
        return contours

    def assign_features(self, features1, features2, features3) -> dict:
        """Assigns points and angles to the left eye, right eye and swim bladder

        The order of the centres and angles should be the same (centres[0] and angles[0] should correspond to the same contour)

        Parameters
        ----------
        features1, features2, features3 : feature_vector
            Vectors containing the x_position, y_position and orientation of each contours

        Returns
        -------
        features : dict
            Dictionary of values for the centres and angles of the eyes and swim bladder
                sb : the x_position, y_position and angle of the swim bladder
                left : the x_position, y_position and angle of the left eye
                right : the x_position, y_position and angle of the right eye
                midpoint : the x_position and y_position of the midpoint between the eyes and the inter-eye angle
        """
        x = np.array((features1, features2, features3))
        assert x.shape == (3, 3), 'feature vectors must contain three values'
        # calculate pairwise distances between centroids
        distances = pdist(x[:, :2])
        # assign the swim bladder index (i.e. the point that is furthest away from the other two)
        sb_index = 2 - distances.argmin()
        # assign the eye indices
        eye_indices = [i for i in range(3) if i != sb_index]
        # assign centres and angles to either the swim bladder or eyes
        sb = x[sb_index]
        eyes = x[eye_indices]
        # calculate vectors from the swim bladder to each eye
        eye_vectors = eyes[:, :2] - sb[:2]
        # assign the left and right eye based on the cross product of the vectors
        cross_product = np.cross(*eye_vectors)
        cross_sign = int(np.sign(cross_product))
        try:
            eyes = eyes[::cross_sign]  # left, right
        except ValueError:  # this happens if the eyes are co-linear with the swim bladder
            raise TrackingError("Eyes are co-linear with swim bladder.")
        # midpoint between eyes
        eye_midpoint = np.mean(eyes[:, :2], axis=0)
        # calculate the angle in between the vectors
        inter_eye_angle = np.abs(cross_product / np.product(np.linalg.norm(eye_vectors, axis=1)))
        # return labelled features
        features = {'sb': feature_vector(*sb),
                    'left': feature_vector(*eyes[0]),
                    'right': feature_vector(*eyes[1]),
                    'midpoint': feature_vector(eye_midpoint[0], eye_midpoint[1], inter_eye_angle)}
        # compute heading
        features = self.heading(features)
        # correct eye angles
        features = self.correct_eye_angles(features)
        return features

    @staticmethod
    def heading(features):
        """Calculate the heading of the fish from identified features

        Parameters
        ----------
        features : dict
            Dictionary of features returned from assign_features

        Returns
        -------
        heading : feature_vector
            Normalized vector (x, y) representing the heading of the fish, and the angle this vector represents
        """
        sb_c = np.array(features['sb'][:2])
        eye_midpoint = np.array(features['midpoint'][:2])
        heading_vector = eye_midpoint - sb_c
        heading_vector /= np.linalg.norm(heading_vector)
        heading = np.arctan2(*heading_vector[::-1])
        features['heading'] = feature_vector(heading_vector[0], heading_vector[1], heading)
        return features

    @staticmethod
    def correct_eye_angles(features):
        """Computes the corrected eye angles using the heading"""
        left_v = np.array([np.cos(features['left'].angle), np.sin(features['left'].angle)])
        right_v = np.array([np.cos(features['right'].angle), np.sin(features['right'].angle)])
        left_dot, right_dot = np.dot(left_v, features['heading'][:2]), np.dot(right_v, features['heading'][:2])
        if left_dot == 0 or right_dot == 0:
            raise TrackingError("Eye is orthogonal to heading - cannot compute accurate angle.")
        if left_dot < 0:
            left_corr = feature_vector(features['left'].x, features['left'].y, features['left'].angle + np.pi)
            features['left'] = left_corr
        if right_dot < 0:
            right_corr = feature_vector(features['right'].x, features['right'].y, features['right'].angle + np.pi)
            features['right'] = right_corr
        return features
