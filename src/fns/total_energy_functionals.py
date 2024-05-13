import pyscf
import numpy as np
from mf import setup_molecule, calculate_mean_field
from dm import lin_gw_dm

def total_energy_g0w0_dm(density_matrix, mf):
    '''Calculates the total energy of a molecule using the expression given in Bruneval. Takes in a density matrix .'''
    def get_coefficients_from_density_matrix(density_matrix):
        '''Extracts the molecular orbital coefficients from a given density matrix by diagonalizing it.'''
        eigvals, eigvecs = np.linalg.eigh(density_matrix)
        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigvals)[::-1]
        sorted_eigvecs = eigvecs[:, sorted_indices]
        sorted_eigvals = eigvals[sorted_indices]

        return sorted_eigvals, sorted_eigvecs
    # get the orbital coefficients from the density matrix
    orbital_coefficients = get_coefficients_from_density_matrix(density_matrix)[1]
    # first the kinetic energy term. 
    Ts_ao = mf.intor_symmetric('int1e_kin')
    ts = np.einsum('ui,uv,vi->', orbital_coefficients, Ts_ao, orbital_coefficients)
    # now the nuclear attraction term
    V_ne_ao = mf.get_hcore() - Ts_ao 
    v_ne = np.einsum('ui,uv,vi->', orbital_coefficients, V_ne_ao, orbital_coefficients)

    # now the hartree term
    vh_ao = mf.get_j()
    vh = 0.5*np.einsum('ui,uvkl,vj->', orbital_coefficients, vh_ao, orbital_coefficients)
    # now the exchange term
    ve_ao = mf.get_k()
    ve = 0.5*np.einsum('ui,uvkl,vj->', orbital_coefficients, ve_ao, orbital_coefficients)

    v_nn = mf.energy_nuc()
    
    e_tot = ts + v_ne + vh + ve + v_nn
    return e_tot

