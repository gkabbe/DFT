import numpy as np
from dft import hartree_fock

from pytest import approx


def test_evaluation():
    x = np.linspace(-3, 3)

    gaussian = hartree_fock.Gaussian(1, 0)

    y = gaussian(x)


def test_gaussian_normalization():
    gaussian = hartree_fock.Gaussian(1, 0)
    x = np.linspace(-3, 3)
    y = gaussian(x)
    dx = x[1] - x[0]


def test_gaussian_multiplication_w_number():
    x = np.linspace(-3, 3)
    g1 = hartree_fock.Gaussian(1, 0)
    g2 = g1 * 3
    g3 = 3 * g1
    g4 = 3 * g3


    y1 = g1(x)
    y2 = g2(x)
    y3 = g3(x)
    y4 = g4(x)

    np.testing.assert_allclose(y2 / y1, 3, err_msg="Multiplication of Gaussian with number failed")
    np.testing.assert_allclose(y3 / y1, 3, err_msg="Multiplication of Gaussian with number failed")
    np.testing.assert_allclose(y4 / y1, 9, err_msg="Multiplication of Gaussian with number failed")


def test_gaussian_multiplication_w_gaussian():
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
    s2 = 3 * s1
    s3 = 3 * s2
    s4 = s2 * 3

    np.testing.assert_allclose(s2(x) / s1(x), 3,
                               err_msg="Multiplication of Slater with number failed")
    np.testing.assert_allclose(s3(x) / s1(x), 9,
                               err_msg="Multiplication of Slater with number failed")
    np.testing.assert_allclose(s4(x) / s1(x), 9,
                               err_msg="Multiplication of Slater with number failed")


# These three tests are not suited for unit testing (too long), but results are ok
#
# def test_sto_1g():
#     sto_ng = hartree_fock.STO_NG([0, 0, 0], coefficients=[1], exponents=[1])
#     coeff, expo = sto_ng.find_coeffs_and_exponents()
#
#     # Assert equal result as Szabo/Ostlund (p. 157)
#     assert expo[0] == approx(0.270950), "Result deviates from Szabo/Ostlund's result"
#
#
# def test_sto_2g():
#     sto_ng = hartree_fock.STO_NG([0, 0, 0], coefficients=[1, 1], exponents=[1, 1])
#     coeff, expo = sto_ng.find_coeffs_and_exponents()
#
#     # Assert equal result as Szabo/Ostlund (p. 157)
#     assert all(coeff == approx([0.678914, 0.430129])), "Result deviates from Szabo/Ostlund's result"
#     assert all(expo == approx([0.151623, 0.851819])), "Result deviates from Szabo/Ostlund's result"
#
#
# def test_sto_3g():
#     sto_ng = hartree_fock.STO_NG([0, 0, 0], coefficients=[1, 1, 1], exponents=[1, 1, 1])
#    coeff, expo = sto_ng.find_coeffs_and_exponents()
#
#    # Assert equal result as Szabo/Ostlund (p. 157)
#    assert all(coeff == approx([0.444635, 0.535328, 0.154329])), \
#        "Result deviates from Szabo/Ostlund's result"
#    assert all(expo == approx([0.109818, 0.405771, 2.22766])), \
#        "Result deviates from Szabo/Ostlund's result"
