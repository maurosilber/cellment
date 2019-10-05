import numpy as np

from .functions import silver_mountain_operator, HistogramRV


def smo_rv(im_shape, sigma, size):
    """Generates a random variable of the SMO operator for a given sigma and size.

    Parameters
    ----------
    im_shape : tuple
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    size : int or tuple of int
        Averaging window size.

    Returns
    -------
    HistogramRV
        Subclass of scipy.stats.rv_histogram.
    """
    im = np.random.normal(size=im_shape)
    smo = silver_mountain_operator(im, sigma, size)
    return HistogramRV.from_data(smo)


def smo_mask(im, sigma, size, threshold=0.1):
    """Returns the mask of (some) background noise.

    Parameters
    ----------
    im : numpy.array or numpy.ma.MaskedArray
        Image. If there are saturated pixels, they should be masked.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    size : int or tuple of int
        Averaging window parameter.
    threshold : float
        Percentile value [0, 1] for the SMO distribution.

    Returns
    -------
    mask : numpy.array

    Notes
    -----
    Sigma and size are scale parameters, and should be less than the typical cell size.
    """
    im = np.ma.asarray(im)
    smo = silver_mountain_operator(im, sigma, size)
    threshold = smo_rv(im.shape, sigma, size).ppf(threshold)
    return (smo < threshold) & ~im.mask


def bg_rv(im, sigma, size, threshold=0.1):
    """Returns the distribution of background noise.

    Use self.median() to get the median value,
    or self.ppf(percentile) to calculate any other desired value.

    Parameters
    ----------
    im : numpy.array or numpy.ma.MaskedArray
        Image. If there are saturated pixels, they should be masked.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    size : int or tuple of int
        Averaging window parameter.
    threshold : float
        Percentile value [0, 1] for the SMO distribution.

    Returns
    -------
    HistogramRV
        Subclass of scipy.stats.rv_histogram.

    Notes
    -----
    Sigma and size are scale parameters, and should be less than the typical cell size.
    """
    mask = smo_mask(im, sigma, size, threshold=threshold)
    bg = im[mask]
    # Remove outliers
    p25, p50 = np.percentile(bg, (25, 50))
    bg = bg[bg < p50 + 20 * (p50 - p25)]
    return HistogramRV.from_data(bg)
