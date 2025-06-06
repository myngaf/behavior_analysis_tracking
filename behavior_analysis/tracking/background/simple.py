import cv2
import numpy as np
import time


def calculate_background(*args, **kwargs):
    video_backgrounds = []
    video_lengths = []
    for arg in args:
        now = time.time()
        print(arg.name, end=' ')
        cap = cv2.VideoCapture(str(arg))
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        n_frames = int(n_frames)
        h, w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h, w = int(h), int(w)
        frames = np.empty((n_frames, h, w), dtype='uint8')
        for f in range(n_frames):
            ret, frame = cap.read()
            if ret:
                frames[f] = frame[..., 0]
            else:
                print(f'No frames {f}')
        background = np.mean(frames, axis=0)
        video_backgrounds.append(background)
        video_lengths.append(n_frames)
        print(time.time() - now)
    video_backgrounds = np.array(video_backgrounds, dtype='float64')
    video_lengths = np.array(video_lengths, dtype='float64')
    video_lengths /= video_lengths.sum()
    background = np.einsum('i,ijk->jk', video_lengths, video_backgrounds)
    return background
