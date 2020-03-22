import fire
import numpy as np

from ..basis_sets.sto_ng import (
    STO_3G,
    overlap_integral,
    kinetic_energy_integral,
    nuclear_attraction,
)


def hydrogen():
    # units in a.u.
    h0 = np.asfarray([0, 0, 0])
    h1 = np.asfarray([1.4, 0, 0])

    phi1 = STO_3G(h0, slater_exponent=1.24)
    phi2 = STO_3G(h1, slater_exponent=1.24)

    overlap_matrix = np.zeros((2, 2))
    kinetic_matrix = np.zeros_like(overlap_matrix)
    potential_matrix_0 = np.zeros_like(overlap_matrix)
    potential_matrix_1 = np.zeros_like(overlap_matrix)

    for i, phi_i in enumerate((phi1, phi2)):
        for j, phi_j in enumerate((phi1, phi2)):
            overlap_matrix[i, j] = overlap_integral(phi_i, phi_j)
            kinetic_matrix[i, j] = kinetic_energy_integral(phi_i, phi_j)
            potential_matrix_0[i, j] = nuclear_attraction(phi_i, phi_j, h0, 1)
            potential_matrix_1[i, j] = nuclear_attraction(phi_i, phi_j, h1, 1)

    print("Overlap:\n", overlap_matrix)
    print("Kinetic:\n", kinetic_matrix)
    print("V0:\n", potential_matrix_0)
    print("V1:\n", potential_matrix_1)

    print("Hamiltonian:\n", kinetic_matrix + potential_matrix_0 + potential_matrix_1)


def cli():
    fire.Fire(hydrogen)
