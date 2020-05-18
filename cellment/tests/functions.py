import unittest

import numpy as np

from ..functions import normalized_vector, normalized_gradient


class NormalizationTest(unittest.TestCase):
    def test_normalized_vector(self):
        n = np.array([[0, 0], [1, 0], [0, 1], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
        self.assertTrue(np.allclose(normalized_vector(n), n))

    def test_non_normalized_vector(self):
        n = 2 * np.array([[0, 0], [1, 0], [0, 1], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
        self.assertTrue(np.allclose(normalized_vector(n), n / 2))

    def test_axis(self):
        n = np.array([[1, 0], [1, 0]])
        self.assertTrue(np.allclose(normalized_vector(n, axis=1), n))
        self.assertTrue(np.allclose(normalized_vector(n, axis=0), n / np.sqrt(2)))


class GradientTest(unittest.TestCase):
    def test_zero_gradient(self):
        x = np.zeros(5)
        self.assertTrue(np.allclose(normalized_gradient(x), x))

    def test_gradient(self):
        self.assertTrue(np.allclose(normalized_gradient(np.arange(5)), np.ones(5)))
        self.assertTrue(np.allclose(normalized_gradient(2 * np.arange(5)), np.ones(5)))


if __name__ == "__main__":
    unittest.main()
