import numpy as np
from dft import hartree_fock

import matplotlib.pyplot as plt


def test_evaluation():
    x = np.linspace(-3, 3)

    gaussian = hartree_fock.Gaussian(1, 0)

    y = gaussian(x)


def test_mul():
    x = np.linspace(-3, 3)
    g1 = hartree_fock.Gaussian(1, 0)
    g2 = g1 * 3

    y1 = g1(x)
    y2 = g2(x)

    np.testing.assert_allclose(y2 / y1, 3, err_msg="Multiplication of Gaussian with number failed")
