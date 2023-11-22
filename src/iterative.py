import numpy as np
from tda import real_corr_se
from mf import setup_molecule, calculate_mean_field
from fock import fock_matrix_hf

# implement the iterative procedure to do g0w0

def g0w0(orbital_number, fock_matrix, real_corr_se):
    # Initial guess
    fock_element = fock_matrix[orbital_number, orbital_number]
    # the initial guess is the fog element
    initial_guess = fock_element
    # Initialize the qpe
    qpe = initial_guess
    iter = 0
    tol = 1e-4
    while True:
        # Update the self energy using the current guess as the frequency
        new_qpe = fock_element + real_corr_se(qpe)[orbital_number]

        # Check if the convergence criterion is met
        if np.abs(new_qpe - qpe) <= tol:
            break

        qpe = new_qpe
        iter += 1
        
    return qpe


my_fock = fock_matrix_hf(setup_molecule())
orbital_number = 6
print(g0w0(orbital_number, my_fock, real_corr_se))