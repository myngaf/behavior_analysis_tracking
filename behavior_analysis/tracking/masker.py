import cv2
import numpy as np
from pathlib import Path


class MaskGenerator:
    """Generating mask for an individual fish."""

    def __init__(self,
                 img: np.ndarray = None,
                 blur: int = 20,
                 dp: int = 1,
                 mindist: int = 512,
                 param1: int = 60,
                 param2: int = 50):
        self.img = img
        self.blur = blur
        self.dp = dp
        self.mindist = mindist
        self.param1 = param1
        self.param2 = param2
        self.circles = None
        self.mask_parameter = None
        self.mask = None

    def read_img(self, img_path):
        # Read image as grayscale
        image = cv2.imread(str(img_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.img = gray

    def generate_masks(self):
        # Get image parameters
        h, w = self.img.shape
        # Blur image
        # blurred = cv2.GaussianBlur(self.img, (self.blur, self.blur), 0)
        # blurred = cv2.bilateralFilter(self.img, self.blur, 20, 20)
        blurred = cv2.medianBlur(self.img, self.blur)
        # Detect edges
        canny = cv2.Canny(blurred, self.param1 / 2, self.param1)
        # Generate circle
        self.circles = cv2.HoughCircles(blurred,
                                        cv2.HOUGH_GRADIENT,
                                        self.dp,
                                        self.mindist,
                                        param1=self.param1,
                                        param2=self.param2,
                                        minRadius=int(np.round(h / 4 * 1.8)),
                                        maxRadius=int(np.round(h / 2 * 1.1)))
        return blurred, canny, self.circles

    def extract_mask(self):
        if self.circles is not None:
            self.circles = np.uint16(np.around(self.circles))
            self.mask_parameter = self.circles[0, 0]
        return self.mask_parameter

    def save_mask(self, output_path):
        h, w = self.img.shape
        self.mask = np.zeros((h, w), dtype=np.uint8)
        center = (self.mask_parameter[0], self.mask_parameter[1])
        cv2.circle(self.mask, center, self.mask_parameter[2], 255, thickness=-1)
        cv2.imwrite(str(output_path), self.mask.astype('uint8'))
        return self.mask

    def run(self, img_path, output_path, **kwargs):
        try:
            assert img_path.exists()
        except AssertionError:
            print(f'{img_path} does not exist. Continue...')
        self.read_img(img_path)
        self.generate_masks()
        result = self.extract_mask()
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        if result is not None:
            self.save_mask(output_path)
        else:
            print('Mask could not be generated!')
        return result


if __name__=='__main__':
    bg_path = r'D:\Experiments\PC_induction_\backgrounds\2024032901.tiff'
    output_path = r'D:\Experiments\PC_induction_\circles\2024032901.tiff'
    masker = MaskGenerator(blur=25, dp=1, mindist=512, param1=2, param2=1)
    masker.run(Path(bg_path), Path(output_path))
