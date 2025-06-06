from video_analysis_toolbox.video import Video
from ...utilities import array_to_point
import cv2
import numpy as np


class TrackedVideo:

    colors = dict(k=0, w=(255, 255, 255), b=(255, 0, 0), g=(0, 255, 0), r=(0, 0, 255), y=(0, 255, 255))

    def __init__(self, video_path, tracking, points, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video = Video.open(video_path, convert_to_grayscale=False, *args, **kwargs)
        self.tracking = tracking
        self.points = points

    def show_tracking(self, frame, **kwargs):
        img = frame.copy()
        tracking = self.tracking.loc[self.video.frame_number]
        points = self.points[self.video.frame_number]
        if tracking.tracked:
            # Centre and heading
            centre = array_to_point([tracking.x, tracking.y])
            heading = np.array([np.cos(tracking.heading), np.sin(tracking.heading)])
            cv2.circle(img, array_to_point(centre), 3, self.colors['y'], -1)
            cv2.line(img, array_to_point(centre), array_to_point(centre + (80 * heading)), self.colors['y'], 2)
            # Eyes
            left_c = np.array([tracking.left_x, tracking.left_y])
            left_v = np.array([np.cos(tracking.left_angle), np.sin(tracking.left_angle)])
            right_c = np.array([tracking.right_x, tracking.right_y])
            right_v = np.array([np.cos(tracking.right_angle), np.sin(tracking.right_angle)])
            cv2.line(img, array_to_point(left_c - (20 * left_v)), array_to_point(left_c + (20 * left_v)),
                     self.colors['g'], 2)
            cv2.line(img, array_to_point(right_c - (20 * right_v)), array_to_point(right_c + (20 * right_v)),
                     self.colors['r'], 2)
            # Tail points
            for p in points:
                cv2.circle(img, array_to_point(p), 1, self.colors['b'], -1)
        return img

    def scroll(self, **kwargs):
        kwargs['display_function'] = self.show_tracking
        self.video.scroll(**kwargs)


# def draw_tracking(image, heading, sb, left, right, **kwargs):
#     img = image.copy()
#     if image.ndim < 3:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     colors = dict(k=0, w=(255, 255, 255), r=(255, 0, 0), g=(0, 255, 0), b=(0, 0, 255), y=(255, 255, 0))
#     # plot heading
#     heading_vector = np.array(heading[:2])
#     centre = sb[:2]
#     cv2.circle(img, array_to_point(centre), 3, colors['y'], -1)
#     cv2.line(img, array_to_point(centre), array_to_point(centre + (80 * heading_vector)), colors['y'], 2)
#     # plot eyes
#     left_v = np.array([np.cos(left.angle), np.sin(left.angle)])
#     right_v = np.array([np.cos(right.angle), np.sin(right.angle)])
#     for p, v, c in zip((left[:2], right[:2]), (left_v, right_v), ('g', 'r')):
#         cv2.line(img, array_to_point(p - (20 * v)), array_to_point(p + (20 * v)), colors[c], 2)
#     # plot tail points
#     for p in kwargs['tail_points']:
#         cv2.circle(img, array_to_point(p), 1, colors['b'], -1)
#     return img
