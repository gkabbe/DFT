from abc import ABCMeta, abstractmethod
import logging
from functools import reduce
from operator import mul

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
from multipledispatch import dispatch


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class OrbitalFunction(metaclass=ABCMeta):
    def __init__(self, alpha, center, *, prefactor=1.0, **kwargs):
        """
        Metaclass for all orbital functions.


        Parameters
        ----------
        alpha : float
        center : float
        prefactor : float, optional
            by default 1
        """
        self.alpha = alpha
        self.center = np.asfarray(center)
        self.prefactor = prefactor
        self.normalization_constant = None

    @abstractmethod
    def _distance_func(self, dims, center):
        pass

    def __call__(self, x):
        x = np.atleast_2d(x)
        return (
            self.prefactor
            * self.normalization_constant
            * np.exp(-self.alpha * self._distance_func(x, self.center))
        )

    def on_grid(self, *x):
        """
        Transform given axes into a grid, and evaluate the orbital on the grid
        """
        dims = np.meshgrid(*x, sparse=True)
        center = np.broadcast_to(self.center, len(dims))
        return (
            self.prefactor
            * self.normalization_constant
            * np.exp(-self.alpha * self._distance_func(dims, center))
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        return type(self)(self.alpha, self.center, prefactor=self.prefactor * other)

    def __repr__(self):
        return (
            f"{type(self).__name__}(alpha={self.alpha}, center={self.center}, "
            f"prefactor={self.prefactor})"
        )


class Slater(OrbitalFunction):
    """Slater function"""

    def __init__(self, alpha, center, *, prefactor=1, **kwargs):
        super().__init__(alpha, center, prefactor=prefactor, **kwargs)
        self.normalization_constant = np.sqrt(alpha ** 3 / np.pi)

    def _distance_func(self, x, center):
        return np.sqrt(np.sum((x - center) ** 2, axis=-1))


class Gaussian(OrbitalFunction):
    """Gaussian function"""

    def __init__(self, alpha, center, *, prefactor=1, normalized=True, **kwargs):
        super().__init__(alpha, center, prefactor=prefactor, **kwargs)
        self.normalization_constant = (
            (2 * alpha / np.pi) ** (3 / 4) if normalized else 1
        )

    def _distance_func(self, x, center):
        return np.sum((x - center) ** 2, axis=-1)

    def __mul__(self, other):
        if type(other) is Gaussian:
            beta = other.alpha
            new_alpha = self.alpha + beta
            prefactor = (
                2 * self.alpha * beta / ((self.alpha + beta) * np.pi)
            ) ** 0.75 * np.exp(
                -self.alpha
                * beta
                / (self.alpha + beta)
                * np.sum((self.center - other.center) ** 2, axis=-1)
            )
            new_center = (self.alpha * self.center + beta * other.center) / (
                self.alpha + beta
            )
            return type(self)(
                new_alpha, new_center, prefactor=self.prefactor * prefactor
            )
        else:
            return super().__mul__(other)


class STO_NG:
    """Slater Type Orbital
    Use N Gaussians to approximate Slater function."""

    def __init__(
        self, center, *, coefficients, exponents, slater_exponent=1.0, **kwargs
    ):
        """
        Parameters
        ----------
        center: float or array_like
            Center of the orbital
        coefficients: array_like
            The coefficients used for the linear combination of Gaussians
        exponents: array_like
            The exponents used for the Gaussians.
        slater_exponent:
            Exponent of the slater function to be fitted.
            If it does not equal one, the exponents will be rescaled
        """
        self.center = np.asfarray(center)
        self.coefficients = np.asfarray(coefficients)
        # Rescale the Gaussian exponents by the square of the slater exponent
        # (Szabo/Ostlund p.160)
        self.exponents = np.asfarray(exponents) * slater_exponent ** 2
        self.slater_exponent = np.asfarray(slater_exponent)
        self.gaussians = [
            Gaussian(expo, center, prefactor=coeff, normalized=True)
            for expo, coeff in zip(self.exponents, self.coefficients)
        ]
        self.normalization_constant = 1 / np.sqrt(overlap_integral(self, self))

    def __call__(self, x):
        result = self.gaussians[0](x)
        for gauss in self.gaussians[1:]:
            result += gauss(x)
        return result * self.normalization_constant

    def __repr__(self):
        center = f"Center: {self.center}"
        parameters = "\n".join(
            f"{exp:12.8f}   {coeff:12.8f}"
            for exp, coeff in zip(self.exponents, self.coefficients)
        )
        repr_ = f"{center}\n{parameters}"
        return repr_

    def __str__(self):
        return self.__repr__()

    def on_grid(self, *x):
        result = self.gaussians[0].on_grid(*x)
        for gauss in self.gaussians[1:]:
            result += gauss.on_grid(*x)
        return result * self.normalization_constant

    def find_coeffs_and_exponents(self, gridpoints=300, width=20):
        """Find parameters for coefficients and exponents by fitting to a Slater function"""
        x, y, z = [
            np.linspace(
                self.center[i] - width / 2, self.center[i] + width / 2, gridpoints
            )
            for i in range(3)
        ]
        dV = reduce(mul, (dim[1] - dim[0] for dim in (x, y, z)))
        slater = Slater(alpha=1.0, center=self.center)
        y_target = slater.on_grid(x, y, z)

        def difference(params):
            coefficients = params[: params.size // 2]
            exponents = params[params.size // 2 :]
            logger.debug(f"Coefficients: {coefficients}")
            logger.debug(f"Exponents: {exponents}")
            sto_ng = type(self)(
                self.center, coefficients=coefficients, exponents=exponents
            )
            diff = np.sum((sto_ng.on_grid(x, y, z) - y_target) ** 2) * dV
            logger.debug(f"Difference: {diff}")
            return diff

        if len(self.coefficients) == 1:
            coeff_bounds = [(1, 1)]
        else:
            coeff_bounds = [(0, None) for _ in self.coefficients]
        expo_bounds = [(0, None) for _ in self.exponents]
        res = minimize(
            difference,
            x0=np.asfarray(self.coefficients + self.exponents),
            bounds=coeff_bounds + expo_bounds,
            method="COBYLA",
            options={"maxiter": 10000},
        )
        logger.info(res)
        result = res.x
        coefficients = result[: len(result) // 2]
        exponents = result[len(result) // 2 :]
        return coefficients, exponents


class STO_1G(STO_NG):
    """Slater Type Orbital fitted with one Gaussian"""

    def __init__(self, center, *, slater_exponent=1.0, **kwargs):
        coefficients = [1]
        exponents = [0.270950]
        super().__init__(
            center=center,
            coefficients=coefficients,
            exponents=exponents,
            slater_exponent=slater_exponent,
            **kwargs,
        )


class STO_2G(STO_NG):
    """Slater Type Orbital fitted with two Gaussians"""

    def __init__(self, center, *, slater_exponent=1.0, **kwargs):
        coefficients = [0.678914, 0.430129]
        exponents = [0.151623, 0.851819]
        super().__init__(
            center=center,
            coefficients=coefficients,
            exponents=exponents,
            slater_exponent=slater_exponent,
            **kwargs,
        )


class STO_3G(STO_NG):
    """Slater Type Orbital fitted with three Gaussians"""

    def __init__(self, center, *, slater_exponent=1.0, **kwargs):
        coefficients = [0.444635, 0.535328, 0.154329]
        exponents = [0.109818, 0.405771, 2.22766]
        super().__init__(
            center=center,
            coefficients=coefficients,
            exponents=exponents,
            slater_exponent=slater_exponent,
            **kwargs,
        )


@dispatch(Gaussian, Gaussian)
def overlap_integral(g_a: Gaussian, g_b: Gaussian):
    alpha = g_a.alpha
    beta = g_b.alpha
    r_a = g_a.center
    r_b = g_b.center
    r_ab = np.linalg.norm(r_b - r_a)

    return (
        g_a.normalization_constant
        * g_b.normalization_constant
        * g_a.prefactor
        * g_b.prefactor
        * (np.pi / (alpha + beta)) ** 1.5
        * np.exp(-alpha * beta / (alpha + beta) * r_ab ** 2)
    )


@dispatch(STO_NG, STO_NG)
def overlap_integral(sto_a, sto_b):
    result = 0
    for g_a in sto_a.gaussians:
        for g_b in sto_b.gaussians:
            result += overlap_integral(g_a, g_b)
    return result


def f_0(t):
    return 0.5 * (np.pi / t) ** 0.5 * erf(t ** 0.5) if t != 0 else 1


@dispatch(Gaussian, Gaussian, Gaussian, Gaussian)
def two_electron_integral(g_a: Gaussian, g_b: Gaussian, g_c: Gaussian, g_d: Gaussian):
    alpha = g_a.alpha
    beta = g_b.alpha
    gamma = g_c.alpha
    delta = g_d.alpha
    r_a = g_a.center
    r_b = g_b.center
    r_c = g_c.center
    r_d = g_d.center
    r_ab = r_b - r_a
    r_cd = r_d - r_c
    r_ab_squared = np.dot(r_ab, r_ab)
    r_cd_squared = np.dot(r_cd, r_cd)
    r_p = (alpha * r_a + beta * r_b) / (alpha + beta)
    r_q = (gamma * r_c + delta * r_d) / (gamma + delta)
    r_pq = r_q - r_p
    r_pq_squared = np.dot(r_pq, r_pq)

    result = (
        2
        * np.pi ** 2.5
        / ((alpha + beta) * (gamma + delta) * (alpha + beta + gamma + delta) ** 0.5)
        * np.exp(
            -alpha * beta / (alpha + beta) * r_ab_squared
            - gamma * delta / (gamma + delta) * r_cd_squared
        )
        * f_0(
            (alpha + beta)
            * (gamma + delta)
            / (alpha + beta + gamma + delta)
            * r_pq_squared
        )
    )

    result *= (
        g_a.normalization_constant
        * g_b.normalization_constant
        * g_c.normalization_constant
        * g_d.normalization_constant
        * g_a.prefactor
        * g_b.prefactor
        * g_c.prefactor
        * g_d.prefactor
    )
    return result


@dispatch(STO_NG, STO_NG, STO_NG, STO_NG)
def two_electron_integral(sto_a, sto_b, sto_c, sto_d):
    result = 0
    for g_a in sto_a.gaussians:
        for g_b in sto_b.gaussians:
            for g_c in sto_c.gaussians:
                for g_d in sto_d.gaussians:
                    result += two_electron_integral(g_a, g_b, g_c, g_d)
    return result


@dispatch(Gaussian, Gaussian)
def kinetic_energy_integral(g_a: Gaussian, g_b: Gaussian):
    alpha = g_a.alpha
    beta = g_b.alpha
    r_a = g_a.center
    r_b = g_b.center
    r_ab = r_b - r_a
    r_squared = np.dot(r_ab, r_ab)

    result = (
        alpha
        * beta
        / (alpha + beta)
        * (3 - 2 * alpha * beta / (alpha + beta) * r_squared)
        * (np.pi / (alpha + beta)) ** 1.5
        * np.exp(-alpha * beta / (alpha + beta) * r_squared)
    )
    result *= (
        g_a.normalization_constant
        * g_b.normalization_constant
        * g_a.prefactor
        * g_b.prefactor
    )
    return result


@dispatch(STO_NG, STO_NG)
def kinetic_energy_integral(sto_a: STO_NG, sto_b: STO_NG):
    result = 0
    for g_a in sto_a.gaussians:
        for g_b in sto_b.gaussians:
            result += kinetic_energy_integral(g_a, g_b)
    return result


@dispatch(Gaussian, Gaussian, object, object)
def nuclear_attraction(g_a: Gaussian, g_b: Gaussian, r_nucleus, Z):
    g_c = g_a * g_b
    alpha, beta = g_a.alpha, g_b.alpha
    r_ab = g_b.center - g_a.center
    r_nuc_c = g_c.center - r_nucleus
    r_ab_squared = np.dot(r_ab, r_ab)
    r_nuc_c_squared = np.dot(r_nuc_c, r_nuc_c)
    result = (
        -2
        * np.pi
        / (alpha + beta)
        * Z
        * np.exp(-alpha * beta / (alpha + beta) * r_ab_squared)
        * f_0((alpha + beta) * r_nuc_c_squared)
    )
    result *= (
        g_a.prefactor
        * g_a.normalization_constant
        * g_b.prefactor
        * g_b.normalization_constant
    )
    return result


@dispatch(STO_NG, STO_NG, object, object)
def nuclear_attraction(sto_a: STO_NG, sto_b: STO_NG, r_nucleus, Z):
    result = 0

    for g_a in sto_a.gaussians:
        for g_b in sto_b.gaussians:
            result += nuclear_attraction(g_a, g_b, r_nucleus, Z)
    return result
