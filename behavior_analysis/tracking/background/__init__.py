import cv2
from .simple import calculate_background
from .chunked import calculate_background_chunked
from .welford import background_statistics_welford
from ...multiprocess import MultiProcessing, trackingmethod


class BackgroundGenerator(MultiProcessing):

    methods = {'normal': calculate_background,
               'chunked': calculate_background_chunked}

    def __init__(self, method, n_processes=4):
        super().__init__(n_processes=n_processes)
        if isinstance(method, str):
            self.method = self.methods[method]
        elif hasattr(method, "__call__"):
            self.method = method
        else:
            raise ValueError(f'func must be callable or {self.methods.keys()}')

    @trackingmethod()
    def run(self, video_paths, output, **kwargs):
        if not output.parent.exists():
            output.parent.mkdir(parents=True)
        result = self.method(*video_paths, **kwargs)
        cv2.imwrite(str(output), result.astype('uint8'))
