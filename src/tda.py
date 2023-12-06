import pyscf
from pyscf.dft import rks
from pyscf.tdscf.rks import dTDA
import numpy as np
from mf import setup_molecule, calculate_mean_field


def real_corr_se(freq):
    '''Calculates the real part of the correlation self energy for a given molecule and frequency. Returns a matrix of correlation energies for each orbital at the given frequency. Only the diagonal of this matrix is considered.'''
    
    molecule = setup_molecule()
    mf, n_orbitals, n_occupied, n_virtual, orbital_energies, eri = calculate_mean_field(molecule, 'hf')

    def dtda_excitations():
        '''Calculates the excitation energies and the R matrix for the molecule in the tda.'''
        # initialize the A matrix
        A = np.zeros((n_occupied, n_virtual, n_occupied, n_virtual))
        # loop over all of the relevant indices
        # Using einsum for the electron repulsion integrals part
        A = 2 * np.einsum('iajb->iajb', eri[:n_occupied, n_occupied:, :n_occupied, n_occupied:])
        reshaped_a = A.reshape(n_occupied*n_virtual, n_occupied*n_virtual)
        
        # initialize the E_ai matrix
        E_ai = np.zeros((n_virtual, n_occupied))
        virtual_energies = orbital_energies[n_occupied:]
        occupied_energies = orbital_energies[:n_occupied]
        E_ai = virtual_energies - occupied_energies[:, None]
        reshaped_E_ai = E_ai.reshape(n_occupied*n_virtual)
        reshaped_a += np.diag(reshaped_E_ai)
        return np.linalg.eigh(reshaped_a) 

    # from pyscf.tdscf.rks import dTDA
    # # now we want to calculate the excitation energies and the R matrix
    omega, R = dtda_excitations()
    # # perform a dtDA calculation to compare with what I have
    # comparison = dTDA(mf)
    # # comparison.xc = 'hf'
    # comparison.kernel()
    # # # assert that every energy in comparison is the same as omega
    # assert np.allclose(comparison.e, omega)
    
    # now that us compute the V matrix
    W_pqia = np.sqrt(2)*eri[:, :, :n_occupied, n_occupied:]
    # now that us reshape it into a form we want
    W_Ipq = W_pqia.reshape(n_orbitals, n_orbitals, n_occupied*n_virtual)
    V_npq = np.einsum('pqi,in->pqn', W_Ipq, R)
    excitations = n_occupied*n_virtual
    # initialize the correlation energies
    correlation_energies = np.zeros((n_orbitals, n_orbitals))
    # we start out by just getting the diagonal elements
    for p in range(n_orbitals):
        for u in range(excitations):
            for j in range(n_occupied):
                correlation_energies[p, p] += V_npq[p, j, u]*V_npq[p, j, u]/(freq - orbital_energies[j] + omega[u])
            for b in range(n_virtual):
                virtual_index = b + n_occupied
                correlation_energies[p, p] += V_npq[p, virtual_index, u]*V_npq[p, virtual_index , u]/(freq - orbital_energies[virtual_index] - omega[u])

    # returned the diagonal
    return np.diag(correlation_energies)    
    