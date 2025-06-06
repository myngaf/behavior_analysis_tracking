import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_image(image):
    plt.imshow(image)
    plt.show()
    # Wait for 5 seconds and then close the plot window
    plt.pause(5)
    plt.close()


def _background_subtraction(image, background, foreground):
    if foreground == 'dark':
        new = background - image
    else:
        new = image - background
    clipped = np.clip(new, 0, 255).astype('uint8')
    return clipped


def _background_subtraction_normalized(image, background, foreground):
    img = image.astype('float64')
    bg = background.astype('float64')
    if foreground == 'dark':
        div = (bg - img + 1) / (bg + 1)
    else:
        div = (img - bg + 1) / (bg + 1)
    div *= 255
    clipped = np.clip(div, 0, 255).astype('uint8')
    return clipped


def background_subtraction(image: np.ndarray, background: np.ndarray, normalize=True, foreground='dark') -> np.ndarray:
    """Subtracts the background from an image or series of image.

    Parameters
    ----------
    image : array like
        The image to perform background subtraction on
    background : array like
        The background image
    normalize : bool
        If True, performs pixel-wise division of the subtracted image by the background image
    foreground : str {'dark', 'light'}
        Whether foreground objects appear dark against a light background, or light against a dark background

    Returns
    -------
    bg : array like
        The result of the background subtraction as an unsigned 8-bit array
    """
    if foreground not in ['dark', 'light']:
        raise ValueError('Invalid foreground. Expected one of: [dark, light].')
    if normalize:
        bg = _background_subtraction_normalized(image, background, foreground)
    else:
        bg = _background_subtraction(image, background, foreground)
    return bg


class BackgroundSubtractor:

    def __init__(self, background, bg_mask, normalize=True, foreground='dark'):
        self.background = background.astype('uint8')
        self.bg_mask = bg_mask.astype('uint8')
        self.normalize = normalize
        if foreground == 'dark':
            self.order = -1
        else:
            self.order = 1

    def _normalize(self, image):
        img = image.astype('float64')
        bg = self.background.astype('float64')
        normed = (img + 1) / (bg + 1)
        normed *= 255
        return normed

    def process(self, image):
        # image_c = cv2.bitwise_not(image.astype('i4'))
        background = self.background
        # background_c = cv2.bitwise_not(background)
        mask = self.bg_mask.astype('u1')
        # masked_image = cv2.bitwise_and(image_c, mask)
        # masked_background = cv2.bitwise_and(background_c, mask)
        # new_image = cv2.bitwise_not(masked_image)
        # new_background = cv2.bitwise_not(masked_background)
        new = background_subtraction(image, background, True, foreground='dark')
        # cv2.imshow('name', new)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # if self.normalize:
        #     new = self._normalize(new)
        # new_c = cv2.bitwise_not(new).astype('u1')
        new = cv2.bitwise_and(new, new, mask=mask)
        # new = cv2.bitwise_not(masked_new).astype('u1')
        # new = np.clip(new, 0, 255).astype('uint8')
        return new
