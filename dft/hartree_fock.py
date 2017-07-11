#!/usr/bin/env python

import numpy as np


class Gaussian:
    """Gaussian function"""
    def __init__(self, alpha, center, prefactor=1):
        self.alpha = alpha
        self.prefactor = prefactor
        if type(center) is np.array:
            self.center = center[None, :]
        else:
            self.center = np.array([center])

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
            return Gaussian(new_alpha, new_center, prefactor)
        else:
            return Gaussian(self.alpha, self.center, other)

    def __rmul__(self, other):
        return self.__mul__(other)
