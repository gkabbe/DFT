#!/usr/bin/env python

import logging
from functools import reduce
from operator import mul
import math

import numpy as np
from scipy.optimize import minimize


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class OrbitalFunction:
    def __init__(self, alpha, center, *, prefactor=1, **kwargs):
        self.alpha = alpha
        self.center = center
        self.prefactor = prefactor
        self.normalization_constant = None

    def _distance_func(self, dims, center):
        raise NotImplementedError

    def __call__(self, *x):
        x = np.atleast_2d(x)
        dims = np.meshgrid(*x, sparse=True)
        center = np.broadcast_to(self.center, len(dims))
        return self.prefactor * self.normalization_constant * \
               np.exp(-self.alpha * self._distance_func(dims, center))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        return type(self)(self.alpha, self.center, prefactor=self.prefactor * other)

    def __repr__(self):
        return f"{type(self).__name__}(alpha={self.alpha}, center={self.center}, " \
               f"prefactor={self.prefactor})"


class Slater(OrbitalFunction):
    """Slater function"""
    def __init__(self, alpha, center, *, prefactor=1, **kwargs):
        super().__init__(alpha, center, prefactor=prefactor, **kwargs)
        self.normalization_constant = np.sqrt(alpha**3 / np.pi)

    def _distance_func(self, dims, center):
        return np.sqrt(np.sum([(d - c)**2 for d, c in zip(dims, center)], axis=0))


class Gaussian(OrbitalFunction):
    """Gaussian function"""
    def __init__(self, alpha, center, *, prefactor=1, **kwargs):
        super().__init__(alpha, center, prefactor=prefactor, **kwargs)
        self.normalization_constant = (2 * alpha / np.pi)**(3 / 4)

    def _distance_func(self, dims, center):
        return np.sum([(d - c)**2 for d, c in zip(dims, center)], axis=0)

    def __mul__(self, other):
        if type(other) is Gaussian:
            beta = other.alpha
            new_alpha = self.alpha + beta
            prefactor = (2 * self.alpha * beta / ((self.alpha + beta) * np.pi))**0.75 * \
                        np.exp(-self.alpha * beta / (self.alpha + beta) *
                               np.sum((self.center - other.center)**2, axis=-1))
            new_center = (self.alpha * self.center + beta * other.center) / (self.alpha + beta)
            return type(self)(new_alpha, new_center, prefactor=self.prefactor * prefactor)
        else:
            return super().__mul__(other)


class STO_NG:
    """Slater Type Orbital
    Use N Gaussians to approximate Slater function."""
    def __init__(self, center, *, coefficients, exponents, slater_exponent=1.0, **kwargs):
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
        self.exponents = np.asfarray(exponents) * slater_exponent**2
        self.slater_exponent = np.asfarray(slater_exponent)
        self.gaussians = [Gaussian(expo, center, prefactor=coeff)
                          for expo, coeff in zip(self.exponents, self.coefficients)]

    def __call__(self, *x):
        result = self.gaussians[0](*x)
        for gauss in self.gaussians[1:]:
            result += gauss(*x)
        return result

    def find_coeffs_and_exponents(self, gridpoints=300, width=20):
        """Find parameters for coefficients and exponents by fitting to a Slater function"""
        x, y, z = [np.linspace(self.center[i] - width / 2, self.center[i] + width / 2, gridpoints)
                   for i in range(3)]
        dV = reduce(mul, (dim[1] - dim[0] for dim in (x, y, z)))
        slater = Slater(alpha=1.0, center=self.center)
        y_target = slater(x, y, z)

        def difference(params):
            coefficients = params[: params.size // 2]
            exponents = params[params.size // 2:]
            logger.debug(f"Coefficients: {coefficients}")
            logger.debug(f"Exponents: {exponents}")
            sto_ng = type(self)(self.center, coefficients=coefficients, exponents=exponents)
            diff = np.sum((sto_ng(x, y, z) - y_target)**2) * dV
            logger.debug(f"Difference: {diff}")
            return diff

        if len(self.coefficients) == 1:
            coeff_bounds = [(1, 1)]
        else:
            coeff_bounds = [(0, None) for _ in self.coefficients]
        expo_bounds = [(0, None) for _ in self.exponents]
        res = minimize(difference, x0=np.asfarray(self.coefficients + self.exponents),
                       bounds=coeff_bounds + expo_bounds, method="COBYLA",
                       options={"maxiter": 10000})
        logger.info(res)
        result = res.x
        coefficients = result[:len(result) // 2]
        exponents = result[len(result) // 2:]
        return coefficients, exponents


class STO_1G(STO_NG):
    """Slater Type Orbital fitted with one Gaussian"""
    def __init__(self, center, **kwargs):
        coefficients = [1]
        exponents = [0.270950]
        super().__init__(center=center, coefficients=coefficients, exponents=exponents, **kwargs)


class STO_2G(STO_NG):
    """Slater Type Orbital fitted with two Gaussians"""
    def __init__(self, center, **kwargs):
        coefficients = [0.678914, 0.430129]
        exponents = [0.151623, 0.851819]
        super().__init__(center=center, coefficients=coefficients, exponents=exponents, **kwargs)


class STO_3G(STO_NG):
    """Slater Type Orbital fitted with three Gaussians"""
    def __init__(self, center, **kwargs):
        coefficients = [0.444635, 0.535328, 0.154329]
        exponents = [0.109818, 0.405771, 2.22766]
        super().__init__(center=center, coefficients=coefficients, exponents=exponents, **kwargs)
