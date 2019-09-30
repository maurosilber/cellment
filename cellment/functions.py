import numpy as np
from scipy import ndimage, stats


def normalized_vector(vector):
    """Normalizes a vector field.

    Parameters
    ----------
    vector : numpy.array
        The input vector field (ndim, ...).

    Returns
    -------
    numpy.array
        Normalized gradient.

    """
    norm = np.linalg.norm(vector, axis=0)
    cond = norm > 0
    vector[:, cond] = vector[:, cond] / norm[cond]
    return vector


def normalized_gradient(image):
    """Calculates the normalized gradient of an image.

    Parameters
    ----------
    image : numpy.array
        Input image.

    Returns
    -------
    numpy.array
        The normalized gradient of the image.
    """
    grad = np.array(np.gradient(image.astype(float)))
    grad = normalized_vector(grad)
    return grad


def silver_mountain_operator(input, sigma, size):
    """Applies the Silver Mountain Operator (SMO) to the input.

    Parameters
    ----------
    input : numpy.array
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    size : int or tuple of int
        Averaging window size.

    Returns
    -------
    numpy.array
    """
    size = ndimage._ni_support._normalize_sequence(size, input.ndim)
    input = ndimage.gaussian_filter(input.astype(float), sigma=sigma)
    norm_grad = normalized_gradient(input)
    sliding_mean = ndimage.uniform_filter(norm_grad, size=(1, *size))
    mod = np.linalg.norm(sliding_mean, axis=0)
    return mod


class HistogramRV(stats.rv_histogram):
    """Returns a random variable from an array of observations.

    data : numpy.array
        Array of observations. If it is multidimensional, it will be flattened.
    bins : int or str
        See numpy.histogram.
    """

    @classmethod
    def from_data(cls, data, bins='fd'):
        return cls(np.histogram(data, bins=bins))

    def save(self, path):
        hist, bins = self._histogram
        np.savez(path, hist=hist, bins=bins)

    @classmethod
    def load(cls, path):
        file = np.load(path)
        return cls((file['hist'], file['bins']))
