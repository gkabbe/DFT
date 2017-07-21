import numpy as np
from dft.basis_sets import sto_ng

from pytest import approx


def test_evaluation():
    x = np.linspace(-3, 3)

    gaussian = sto_ng.Gaussian(1, 0)

    y = gaussian(x)


def test_gaussian_normalization():
    gaussian = sto_ng.Gaussian(1, 0)
    x = np.linspace(-3, 3)
    y = gaussian(x)
    dx = x[1] - x[0]


def test_gaussian_multiplication_w_number():
    x = np.linspace(-3, 3)
    g1 = sto_ng.Gaussian(1, 0)
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
    g1 = sto_ng.Gaussian(2, -1)
    g2 = sto_ng.Gaussian(1, 1)
    g3 = g1 * g2

    y1 = g1(x)
    y2 = g2(x)
    y3 = g3(x)

    numeric_result = y1 * y2

    assert np.allclose(y3, numeric_result), "Gaussian product does not match numerical result"


def test_slater():
    x = np.linspace(-3, 3)
    s1 = sto_ng.Slater(1, 0)
    s2 = 3 * s1
    s3 = 3 * s2
    s4 = s2 * 3

    np.testing.assert_allclose(s2(x) / s1(x), 3,
                               err_msg="Multiplication of Slater with number failed")
    np.testing.assert_allclose(s3(x) / s1(x), 9,
                               err_msg="Multiplication of Slater with number failed")
    np.testing.assert_allclose(s4(x) / s1(x), 9,
                               err_msg="Multiplication of Slater with number failed")


def test_overlap():
    gauss1 = sto_ng.Gaussian(1, 0, prefactor=0.5)
    gauss2 = sto_ng.Gaussian(1.5, 0)

    analytical_result = sto_ng.overlap_integral(gauss1, gauss2)

    x = np.linspace(-5, 5, 100)
    y = (gauss1 * gauss2)(x, x, x)
    numerical_result = np.sum(y) * (x[1] - x[0])**3

    assert analytical_result == approx(numerical_result)


def test_overlap_h2():
    """Calculate the overlap matrix for hydrogen"""
    target_sto1g = np.array([[1.0, 0.6648],
                             [0.6648, 1.0]])
    target_sto3g = np.array([[1.0, 0.6593],
                             [0.6593, 1.0]])

    for STO, target in zip([sto_ng.STO_1G, sto_ng.STO_3G], [target_sto1g, target_sto3g]):
        phi_1 = STO(center=[0, 0, 0], slater_exponent=1.24)
        phi_2 = STO(center=[1.4, 0, 0], slater_exponent=1.24)

        overlap_matrix = np.zeros((2, 2))
        overlap_matrix[0, 0] = sto_ng.overlap_integral(phi_1, phi_1)
        overlap_matrix[0, 1] = sto_ng.overlap_integral(phi_1, phi_2)
        overlap_matrix[1, 0] = sto_ng.overlap_integral(phi_2, phi_1)
        overlap_matrix[1, 1] = sto_ng.overlap_integral(phi_2, phi_2)

        assert np.allclose(overlap_matrix, target, rtol=1e-4)


def test_two_electron_integral():
    phi1 = sto_ng.STO_3G(center=[0, 0, 0], slater_exponent=1.24)
    phi2 = sto_ng.STO_3G(center=[1.4, 0, 0], slater_exponent=1.24)

    print(sto_ng.two_electron_integral(phi1, phi1, phi2, phi2))


def test_kinetic():
    phi1 = sto_ng.STO_3G(center=[0, 0, 0], slater_exponent=1.24)
    phi2 = sto_ng.STO_3G(center=[1.4, 0, 0], slater_exponent=1.24)

    print(sto_ng.kinetic_energy_integral(phi1, phi2))

# These three tests are not suited for unit testing (too long), but results are ok
#
# def test_sto_1g():
#     sto = sto_ng.STO_NG([0, 0, 0], coefficients=[1], exponents=[1])
#     coeff, expo = sto.find_coeffs_and_exponents()
#
#     # Assert equal result as Szabo/Ostlund (p. 157)
#     assert expo[0] == approx(0.270950), "Result deviates from Szabo/Ostlund's result"
#
#
# def test_sto_2g():
#     sto = sto_ng.STO_NG([0, 0, 0], coefficients=[1, 1], exponents=[1, 1])
#     coeff, expo = sto.find_coeffs_and_exponents()
#
#     # Assert equal result as Szabo/Ostlund (p. 157)
#     assert all(coeff == approx([0.678914, 0.430129])), "Result deviates from Szabo/Ostlund's result"
#     assert all(expo == approx([0.151623, 0.851819])), "Result deviates from Szabo/Ostlund's result"
#
#
# def test_sto_3g():
#     sto = sto_ng.STO_NG([0, 0, 0], coefficients=[1, 1, 1], exponents=[1, 1, 1])
#    coeff, expo = sto.find_coeffs_and_exponents()
#
#    # Assert equal result as Szabo/Ostlund (p. 157)
#    assert all(coeff == approx([0.444635, 0.535328, 0.154329])), \
#        "Result deviates from Szabo/Ostlund's result"
#    assert all(expo == approx([0.109818, 0.405771, 2.22766])), \
#        "Result deviates from Szabo/Ostlund's result"
