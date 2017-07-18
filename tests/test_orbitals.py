import numpy as np
from dft import hartree_fock

import matplotlib.pyplot as plt


def test_evaluation():
    x = np.linspace(-3, 3)

    gaussian = hartree_fock.Gaussian(1, 0)

    y = gaussian(x)


def test_multiplication_w_number():
    x = np.linspace(-3, 3)
    g1 = hartree_fock.Gaussian(1, 0)
    g2 = g1 * 3
    g3 = 3 * g1

    y1 = g1(x)
    y2 = g2(x)
    y3 = g3(x)

    np.testing.assert_allclose(y2 / y1, 3, err_msg="Multiplication of Gaussian with number failed")
    np.testing.assert_allclose(y3 / y1, 3, err_msg="Multiplication of Gaussian with number failed")


def test_multiplication_w_gaussian():
    x = np.linspace(-3, 3)
    g1 = hartree_fock.Gaussian(2, -1)
    g2 = hartree_fock.Gaussian(1, 1)
    g3 = g1 * g2

    y1 = g1(x)
    y2 = g2(x)
    y3 = g3(x)

    numeric_result = y1 * y2

    assert np.allclose(y3, numeric_result), "Gaussian product does not match numerical result"


def test_slater():
    x = np.linspace(-3, 3)
    s1 = hartree_fock.Slater(1, 0)

    plt.plot(x, s1(x))
    plt.show()
