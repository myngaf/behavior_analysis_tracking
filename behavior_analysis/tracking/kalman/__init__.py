from filterpy.kalman import KalmanFilter, FixedLagSmoother
import numpy as np


class AngleUnwrapper:

    def __init__(self):
        super().__init__()
        self.a = 0

    def __call__(self, angle, *args, **kwargs):
        self.a = np.unwrap(np.array([self.a, angle]))[1]
        return self.a


class _FishFilter:

    def __init__(self, z0):
        self.F = np.array([[1, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0],
                           [0, 0, 1, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]], dtype='float64')
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='float64').T
        self.P *= 0.001
        self.Q *= [0.0001, 0.0001, 0.0001, 1., 1., 1.]
        self.R *= 10.
        self.x = np.array([z0[0], z0[1], z0[2], 0, 0, 0])
        self.unwrap_angle = AngleUnwrapper()


class FishKalmanFilter(KalmanFilter, _FishFilter):

    def __init__(self, z0):
        super().__init__(dim_x=6, dim_z=3)
        super(KalmanFilter, self).__init__(z0)
        self.xFiltered = []

    def filter(self, z):
        zk = z.copy()
        zk[2] = self.unwrap_angle(z[2])
        self.predict()
        self.update(zk)
        self.xFiltered.append(self.x[:3])


class FishKalmanSmoother(FixedLagSmoother, _FishFilter):

    def __init__(self, z0, n=100):
        super().__init__(6, 3, N=n)
        super(FixedLagSmoother, self).__init__(z0)

    def smooth(self, z, u=None):
        zk = z.copy()
        zk[2] = self.unwrap_angle(z[2])
        super().smooth(zk)
