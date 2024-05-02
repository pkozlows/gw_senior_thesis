import numpy as np
from tda import real_corr_se, my_drpa, my_dtda
from mf import setup_molecule, calculate_mean_field
from fock import simple_fock, fock_dft, pyscf_fock_dft
from pyscf import gw
from pyscf import tddft
import pyscf
from pyscf.dft import rks
from pyscf import scf
from scipy.optimize import newton


# implement the iterative procedure to do g0w0
def g0w0(orbital_number, fock_mo, real_corr_se, corr_mf, tddft):
    '''Calculates the G0W0 correction for a given orbital number, fock matrix in the atomic orbital basis, and real part of the correlation self energy.'''
    def gw_correction(omega, fock_element, real_corr_se, tddft, corr_mf, orbital_number):
        sigma = real_corr_se(omega, tddft, corr_mf)[orbital_number]
        return omega - fock_element - sigma
    
    # Initial guess
    fock_element = fock_mo[orbital_number, orbital_number]

    # Use scipy.optimize.newton to find the root
    qpe = newton(gw_correction, fock_element, args=(fock_element, real_corr_se, tddft, corr_mf, orbital_number))

    return qpe

