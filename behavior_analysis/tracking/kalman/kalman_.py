import numpy as np
from collections import deque, namedtuple


class KalmanFilter:

    def __init__(self, x0, p_error, y_error, **kwargs):

        # Initial state matrix
        self.X = np.array(x0)
        self.n = len(self.X)
        # Initial process covariance matrix
        self.P = self.covariance(p_error)
        # Measurement error
        self.R = self.covariance(y_error)

        # Transformation matrices (override in subclass)
        self.A = kwargs.get('a', np.eye(self.n))
        self.B = kwargs.get('b', np.zeros((self.n, self.n)))

    @staticmethod
    def covariance(v):
        return np.diag(np.array(v) ** 2)

    def update(self, y, u=(0,)):
        # Predicted state
        X = self.A@self.X
        if np.any(u):
            X = X + self.B@np.array(u)
        P = np.diag(np.diag(self.A@self.P@self.A.T))
        # Kalman gain
        K = P@np.linalg.inv(P + self.R)
        # Update process and state matrices
        self.X = X + K@(np.array(y) - X)
        self.P = (np.eye(self.n) - K)@P

    def predict(self):
        return self.X


class FishTrackerKalman(KalmanFilter):

    vector = namedtuple('vector', ('x_pos', 'y_pos', 'x_dir', 'y_dir'))
    # x_pos, y_pos : position
    # x_dir, y_dir : heading
    error = namedtuple('error', ('x_pos', 'dx_pos', 'y_pos', 'dy_pos',
                                 'x_dir', 'dx_dir', 'y_dir', 'dy_dir'))

    def __init__(self, y0, error, **kwargs):
        v0 = kwargs.get('v0', np.zeros(len(y0)))   # set initial velocity
        y = np.array(list(zip(y0, v0))).flatten()  # combine position and velocity vectors
        super().__init__(y, error, error, **kwargs)
        # Create transformation matrices A and B
        self.A = np.eye(self.n)
        self.A[np.arange(0, self.n, 2), np.arange(1, self.n, 2)] = 1
        self.B = np.diag(np.tile([0.5, 1], self.n // 2))
        # Observed values
        self.Y = self.X.copy()
        # Queues for calculating running averages for velocity and acceleration
        self.xy_acc = deque(np.zeros((20, 2)), maxlen=50)  # position
        self.vxy_acc = deque(np.zeros((10, 2)), maxlen=10)  # heading vector
        self.dy = np.zeros(4)

    def reset(self, y0, **kwargs):
        v0 = kwargs.get('vel_init', np.zeros(len(y0)))
        y = np.array(list(zip(y0, v0))).flatten()
        assert y.shape == self.X.shape
        self.X = y
        self.Y = self.X.copy()

    def update(self, y, u=(0,)):
        # Calculate instantaneous velocity
        dy = y - self.Y[::2]
        self.xy_acc.append(dy[:2])
        self.vxy_acc.append(dy[2:])
        # Calculate running average velocity
        xy = np.mean(np.array(self.xy_acc), axis=0)
        vxy = np.median(np.array(self.vxy_acc), axis=0)
        dy = np.array([xy[0], xy[1], vxy[0], vxy[1]])
        # Calculate new acceleration
        u = dy - self.dy
        # Update model
        self.Y = np.array(list(zip(y, dy))).flatten()
        u = np.repeat(u, 2)
        self.dy = dy
        super().update(self.Y, u)

    def predict(self):
        return self.X[::2]
