import cv2
import numpy as np
import time


def chunked_average(video_file, chunk_size=1000):
    # Open video file
    cap = cv2.VideoCapture(str(video_file))
    # Video properties
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    n_frames = int(n_frames)
    h, w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h, w = int(h), int(w)
    # Assign space in RAM for chunks
    n_chunks, remainder = np.divmod(n_frames, chunk_size)
    chunk_sizes = [chunk_size] * n_chunks
    if remainder > 0:
        chunk_sizes += [remainder]
    chunk_sizes = np.array(chunk_sizes)
    chunks = np.empty((len(chunk_sizes), h, w))
    # Compute the average of each chunk
    for i, size in enumerate(chunk_sizes):
        chunk = np.empty((size, h, w), dtype='uint8')
        for f in range(size):
            ret, frame = cap.read()
            if ret:
                chunk[f] = frame[..., 0]
            # else:
            #     print(f'No frame: {i * chunk_size + f}')
        chunks[i] = np.mean(chunk, axis=0)
    # Compute average from chunks
    chunk_sizes = chunk_sizes.astype('float64') / float(chunk_sizes.sum())
    average = np.einsum('i,ijk->jk', chunk_sizes, chunks)
    cap.release()
    return average, float(n_frames)


def calculate_background_chunked(*args, chunk_size=1000):
    video_backgrounds, video_weights = [], []
    for arg in args:
        average, n_frames = chunked_average(arg, chunk_size=chunk_size)
        video_backgrounds.append(average)
        video_weights.append(n_frames)
    video_backgrounds = np.array(video_backgrounds, dtype='float64')
    video_weights = np.array(video_weights, dtype='float64')
    video_weights /= video_weights.sum()
    background = np.einsum('i,ijk->jk', video_weights, video_backgrounds)
    return background


def chunked_statistics(video_file, chunk_size=1000):
    # Open video file
    cap = cv2.VideoCapture(str(video_file))
    # Video properties
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    n_frames = int(n_frames)
    h, w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h, w = int(h), int(w)
    # Assign space in RAM for chunks
    n_chunks, remainder = np.divmod(n_frames, chunk_size)
    chunk_sizes = [chunk_size] * n_chunks
    if remainder > 0:
        chunk_sizes += [remainder]
    chunk_sizes = np.array(chunk_sizes)
    chunk_averages = np.empty((len(chunk_sizes), h, w))
    chunk_variances = np.empty((len(chunk_sizes), h, w))
    # Compute the average of each chunk
    for i, size in enumerate(chunk_sizes):
        chunk = np.empty((size, h, w), dtype='uint8')
        for f in range(size):
            ret, frame = cap.read()
            if ret:
                chunk[f] = frame[..., 0]
            # else:
            #     print(f'No frame: {i * chunk_size + f}')
        chunk_averages[i] = np.mean(chunk, axis=0)
        chunk_variances[i] = np.var(chunk, axis=0)
    # Compute average from chunks
    chunk_sizes = chunk_sizes.astype('float64') / float(chunk_sizes.sum())
    average = np.einsum('i,ijk->jk', chunk_sizes, chunk_averages)
    variance = np.einsum('i,ijk->jk', chunk_sizes ** 2, chunk_variances)
    cap.release()
    return average, variance, float(n_frames)


def calculate_background_statistics(*args, chunk_size=1000):
    video_averages, video_vars, video_weights = [], [], []
    for arg in args:
        now = time.time()
        print(arg.name, end=' ')
        average, variance, n_frames = chunked_statistics(arg, chunk_size=chunk_size)
        video_averages.append(average)
        video_vars.append(variance)
        video_weights.append(n_frames)
        print(time.time() - now)
    video_averages = np.array(video_averages, dtype='float64')
    video_vars = np.array(video_vars, dtype='float64')
    video_weights = np.array(video_weights, dtype='float64')
    video_weights /= video_weights.sum()
    average = np.einsum('i,ijk->jk', video_weights, video_averages)
    variance = np.einsum('i,ijk->jk', video_weights ** 2, video_vars)
    return average, variance
