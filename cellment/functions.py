import numpy as np
from scipy import ndimage, stats
from scipy.ndimage._ni_support import _normalize_sequence


def normalized_vector(vector, axis=-1):
    """Normalizes a vector field.

    Parameters
    ----------
    vector : numpy.array
        Vector field.
    axis : int
        Axis of vector coordinates. By default, last axis.

    Returns
    -------
    numpy.array
        Normalized vector field.

    """
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    return np.divide(vector, norm, where=(norm > 0))


def normalized_gradient(input):
    """Calculates the normalized gradient of a scalar field.

    Parameters
    ----------
    input : numpy.array
        Input field.

    Returns
    -------
    numpy.array
        The normalized gradient of the scalar field.
    """
    grad = np.array(np.gradient(input.astype(float, copy=False)), ndmin=2)
    return normalized_vector(grad, axis=0)


def silver_mountain_operator(input, sigma, size):
    """Applies the Silver Mountain Operator (SMO) to a scalar field.

    Parameters
    ----------
    input : numpy.array
        Input field.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel.
    size : int or sequence of int
        Averaging window size.

    Returns
    -------
    numpy.array
    """
    size = _normalize_sequence(size, input.ndim)
    input = ndimage.gaussian_filter(input.astype(float, copy=False), sigma=sigma)
    norm_grad = normalized_gradient(input)
    sliding_mean = ndimage.uniform_filter(norm_grad, size=(1, *size))
    magnitude = np.linalg.norm(sliding_mean, axis=0)
    return magnitude


class HistogramRV(stats.rv_histogram):
    @classmethod
    def from_data(cls, data, bins='fd'):
        """Generates RV from data.

        data : array-like
            Array of observations. If it is multidimensional, it will be flattened.
        """
        if np.issubdtype(data.dtype, np.integer):
            data = np.ma.asarray(data)
            hist = np.bincount(data.compressed())
            bins = np.arange(hist.size + 1)
            # TODO: Keep only non-zero entries.
            return cls((hist, bins))
        else:
            return cls(np.histogram(data, bins=bins))

    def save(self, path):
        """Saves histogram to an npz file."""
        hist, bins = self._histogram
        np.savez(path, hist=hist, bins=bins)

    @classmethod
    def load(cls, path):
        """Loads histogram from npz file."""
        file = np.load(path)
        return cls((file['hist'], file['bins']))
