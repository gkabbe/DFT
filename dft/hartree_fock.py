#!/usr/bin/env python

import numpy as np


class OrbitalFunction:
    def __init__(self, alpha, center, *, prefactor=1.0):
        self.alpha = alpha
        self.prefactor = prefactor
        if type(center) is np.array:
            self.center = center[None, :]
        else:
            self.center = np.array([center])


class Slater(OrbitalFunction):
    """Slater function"""
    def __call__(self, x):
        x = x.reshape((-1, self.center.shape[-1]))
        return self.prefactor * (self.alpha**3 / np.pi)**0.5 * \
               np.exp(-self.alpha * np.sqrt(np.sum((x - self.center)**2)))

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
            return Gaussian(new_alpha, new_center, prefactor=prefactor)
        else:
            return Gaussian(self.alpha, self.center, other)

    def __rmul__(self, other):
        return self.__mul__(other)


class STO_NG:
    """Slater Type Orbital
    Use N Gaussians to approximate Slater function."""
    def __init__(self, coefficients, exponents):
        self.coefficients = coefficients
        self.exponents = exponents

        self.gaussians = [Gaussian(expo, 0, prefactor=coeff)
                          for expo, coeff in zip(self.exponents, self.coefficients)]

    def __call__(self, x):
        result = np.zeros(x.shape)
        for gauss in self.gaussians:
            result += gauss(x)
        return result

    def find_coeffs_and_exponents(self, number_of_gaussians):
        """Find parameters for coefficients and exponents by fitting to a Slater function"""
