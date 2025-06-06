from sklearn.decomposition import PCA as sklearn_pca
import numpy as np


class PCA:

    def __init__(self, data):
        self.data = data

    @property
    def mean(self):
        """Returns the mean of the data."""
        return self.data.mean(axis=0)

    @property
    def std(self):
        """Returns the standard deviation of the data."""
        return self.data.std(axis=0)

    def whiten(self, mean=None, std=None):
        if (mean is None) and (std is None):
            whitened = (self.data - self.mean) / self.std
        elif (mean is not None) and (std is not None):
            whitened = (self.data - mean) / std
        else:
            raise ValueError('both mean and std must be specified!')
        return PCA(whitened)

    def transform(self, whiten=True, **kwargs):
        if whiten:
            data_to_transform = self.whiten(**kwargs).data
        else:
            data_to_transform = self.data
        pca = sklearn_pca()
        transformed = pca.fit_transform(data_to_transform)
        return PCA(transformed), pca

    def map(self, vectors, whiten=True, **kwargs):
        if whiten:
            data_to_map = self.whiten(**kwargs).data
        else:
            data_to_map = self.data
        assert vectors.shape[1] == data_to_map.shape[1], 'pca vector shape does not match data shape'
        mapped = np.dot(data_to_map, vectors.T)
        return PCA(mapped)
