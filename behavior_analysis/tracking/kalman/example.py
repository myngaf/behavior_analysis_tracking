from .kalman_ import KalmanFilter
import numpy as np


class Kalman2DExample(KalmanFilter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Transformation matrix
        self.A = np.array([[1, 1],
                           [0, 1]])
        self.B = np.diag([0.5, 1])

    def update(self, y, u=(2, 2)):
        super().update(y, u=u)


if __name__ == "__main__":
    Y = [[4000, 280], [4260, 282], [4550, 285], [4860, 286], [5110, 290]]
    y_err = (25, 6)
    p_err = (20, 5)

    y0 = Y.pop(0)
    kalman = Kalman2DExample(y0, p_err, y_err)
    for y in Y:
        kalman.update(y)
        print(kalman.predict())
