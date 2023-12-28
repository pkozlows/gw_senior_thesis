import numpy as np
from tda import real_corr_se, my_drpa, my_dtda
from mf import setup_molecule, calculate_mean_field
from fock import simple_fock, fock_dft, pyscf_fock_dft
from pyscf import gw
from pyscf import tddft
import pyscf
from pyscf.dft import rks
from pyscf import scf


# implement the iterative procedure to do g0w0
def g0w0(orbital_number, fock_mo, real_corr_se, mf, tddft):
    '''Calculates the G0W0 correction for a given orbital number, fock matrix in the atomic orbital basis, and real part of the correlation self energy.'''
    # convert the fock matrix to the molecular orbital basis

    # Initial guess
    fock_element = fock_mo[orbital_number, orbital_number]
    # the initial guess is the fog element
    initial_guess = fock_element
    # Initialize the qpe
    qpe = initial_guess
    iter = 0
    tol = 1e-9
    while True:
        # Update the self energy using the current guess as the frequency
        new_qpe = fock_element + real_corr_se(qpe, tddft, mf)[orbital_number]

        # Check if the convergence criterion is met
        if np.abs(new_qpe - qpe) <= tol:
            break

        qpe = new_qpe
        iter += 1
        
    return qpe
