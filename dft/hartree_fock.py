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
    def __init__(self, alpha, center, *, prefactor=1.0):
        self.alpha = alpha
        self.prefactor = prefactor
        self.center = np.atleast_2d(center)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"{type(self).__name__}(alpha={self.alpha}, center={self.center}, " \
               f"prefactor={self.prefactor})"


class Slater(OrbitalFunction):
    """Slater function"""
    def __call__(self, x):
        x = x.reshape((-1, self.center.shape[-1]))
        return self.prefactor * (self.alpha**3 / np.pi)**0.5 * \
               np.exp(-self.alpha * np.sqrt(np.sum((x - self.center)**2, axis=-1)))

    def __mul__(self, other):
        return type(self)(self.alpha, self.center, prefactor=self.prefactor * other)


class Gaussian(OrbitalFunction):
    """Gaussian function"""
    def __call__(self, x):
        x = x.reshape((-1, self.center.shape[-1]))
        return self.prefactor * (2 * self.alpha / np.pi)**0.75 * \
               np.exp(-self.alpha * np.sum((x - self.center)**2, axis=-1))

    def __mul__(self, other):
        if type(other) is Gaussian:
            beta = other.alpha
            new_alpha = self.alpha + beta
            prefactor = (2 * self.alpha * beta / ((self.alpha + beta) * np.pi))**0.75 * \
                        np.exp(-self.alpha * beta / (self.alpha + beta) *
                               np.sum((self.center - other.center)**2, axis=-1))
            new_center = (self.alpha * self.center + beta * other.center) / (self.alpha + beta)
            return type(self)(new_alpha, new_center, prefactor=prefactor)
        else:
            return type(self)(self.alpha, self.center, prefactor=self.prefactor * other)


class STO_NG:
    """Slater Type Orbital
    Use N Gaussians to approximate Slater function."""
    def __init__(self, center, *, coefficients, exponents):
        self.center = np.asarray(center, dtype=float)
        self.coefficients = coefficients
        self.exponents = exponents

        self.gaussians = [Gaussian(expo, center, prefactor=coeff)
                          for expo, coeff in zip(self.exponents, self.coefficients)]

    def __call__(self, x):
        result = np.zeros(x.shape[0])
        for gauss in self.gaussians:
            result += gauss(x)
        return result

    def find_coeffs_and_exponents(self):
        """Find parameters for coefficients and exponents by fitting to a Slater function"""
        width = 10
        x, y, z = [np.linspace(self.center[i] - width / 2, self.center[i] + width / 2, 100)
                   for i in range(3)]

        X, Y, Z = np.meshgrid(x, y, z)
        r = np.c_[X.flat, Y.flat, Z.flat]

        dV = reduce(mul, (dim[1] - dim[0] for dim in (x, y, z)))

        slater = Slater(alpha=1.0, center=self.center)
        y_target = slater(r)

        def difference(params):
            coefficients = params[: params.size // 2]
            exponents = params[params.size // 2:]
            sto_ng = type(self)(self.center, coefficients=coefficients, exponents=exponents)
            diff = np.sum((sto_ng(r) - y_target)**2) * dV
            logger.debug(f"Difference: {diff}")
            if math.isnan(diff):
                import ipdb; ipdb.set_trace()

            return diff

        res = minimize(difference, x0=self.coefficients + self.exponents)
        print(res)

