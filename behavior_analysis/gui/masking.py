import cv2
import numpy as np
from behavior_analysis.tracking import MaskGenerator


class SetCircularMask:
    def __init__(self,
                 masker,
                 window_name='Set Thresholds for Circular Mask',):
        self.masker = masker
        self.window_name = window_name
        self.frame = None
        self.masks = None
        self.initialize()

    def initialize(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # make a window with name 'image'
        # cv2.createTrackbar('#image', 'image', n, max_images - 1, callback)
        cv2.createTrackbar('blur', self.window_name, self.masker.blur, 200, self.callback)
        cv2.createTrackbar('dp', self.window_name, self.masker.dp, 20, self.callback)
        cv2.createTrackbar('minDist', self.window_name, self.masker.mindist, 1024, self.callback)
        cv2.createTrackbar('param1', self.window_name, self.masker.param1, 512, self.callback)  # lower threshold trackbar for window 'image
        cv2.createTrackbar('param2', self.window_name, self.masker.param2, 512, self.callback)  # upper threshold trackbar for window 'image
        # cv2.setTrackbarMin('#image', self.window_name, 1)
        cv2.setTrackbarMin('blur', self.window_name, 1)
        cv2.setTrackbarMin('dp', self.window_name, 1)
        cv2.setTrackbarMin('minDist', self.window_name, 1)
        cv2.setTrackbarMin('param1', self.window_name, 1)
        cv2.setTrackbarMin('param2', self.window_name, 1)

    def get_parameters(self):
        self.masker.blur = cv2.getTrackbarPos('blur', self.window_name)
        # Requirement for median blur
        if (self.masker.blur % 2) == 0:
            self.masker.blur += 1
        self.masker.dp = cv2.getTrackbarPos('dp', self.window_name)
        self.masker.mindist = cv2.getTrackbarPos('minDist', self.window_name)
        self.masker.param1 = cv2.getTrackbarPos('param1', self.window_name)
        self.masker.param2 = cv2.getTrackbarPos('param2', self.window_name)

    def update_frame(self):
        # Update the frame content here
        blurred, canny, circles = self.masker.generate_masks()
        self.frame = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
        self.masks = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0]:
                x, y, r = circle
                cv2.circle(self.masks, (x, y), r, (0, 0, 255), thickness=1)

    def show(self):
        while True:
            self.get_parameters()
            self.update_frame()
            display = np.concatenate((self.frame, self.masks), axis=1)  # to display image side by side
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n') or key == 27:  # Press 'n' to close window and 'ESC' to quit assignment for experiment
                break
        cv2.destroyAllWindows()
        return self.masker.blur, self.masker.dp, self.masker.mindist, self.masker.param1, self.masker.param2, key

    @staticmethod
    def callback(value):
        pass


# Example usage:
if __name__ == "__main__":
    bg_path = r'D:\Experiments\PC_induction_\backgrounds\2024032901.tiff'
    masker = MaskGenerator()
    masker.read_img(bg_path)
    widget = SetCircularMask(masker)
    parameters = widget.show()
    print(parameters)
