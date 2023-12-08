import pyscf
from pyscf.dft import rks
from pyscf.tdscf.rks import dTDA, dRPA
import numpy as np
from mf import setup_molecule, calculate_mean_field


def real_corr_se(freq):
    '''Calculates the real part of the correlation self energy for a given molecule and frequency. Returns a matrix of correlation energies for each orbital at the given frequency. Only the diagonal of this matrix is considered.'''
    
    molecule = setup_molecule()
    mf, n_orbitals, n_occupied, n_virtual, orbital_energies = calculate_mean_field(molecule, 'hf')
    eri_mo = molecule.ao2mo(mf.mo_coeff, compact=False).reshape(n_orbitals, n_orbitals, n_orbitals, n_orbitals)

    def dtda_excitations():
        '''Calculates the excitation energies and the R matrix for the molecule in the tda.'''
        # initialize the A matrix
        A = np.zeros((n_occupied, n_virtual, n_occupied, n_virtual))
        # loop over all of the relevant indices
        # Using einsum for the electron repulsion integrals part
        A = 2 * np.einsum('iajb->iajb', eri_mo[:n_occupied, n_occupied:, :n_occupied, n_occupied:])
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
    def drpa():
        '''Calculates the excitation energies and the R matrix for the molecule in the rpa.'''
        # initialize the A matrix
        A = np.zeros((n_occupied, n_virtual, n_occupied, n_virtual))
        # initialize the B matrix
        B = np.zeros((n_occupied, n_virtual, n_occupied, n_virtual))
        # add the common part to boots
        A += 2 * np.einsum('iajb->iajb', eri_mo[:n_occupied, n_occupied:, :n_occupied, n_occupied:])
        B += 2 * np.einsum('iajb->iajb', eri_mo[:n_occupied, n_occupied:, :n_occupied, n_occupied:])
        # re shape both
        reshaped_a = A.reshape(n_occupied*n_virtual, n_occupied*n_virtual)
        reshaped_b = B.reshape(n_occupied*n_virtual, n_occupied*n_virtual)

        # add the unique part to A
        # initialize the E_ai matrix
        E_ai = np.zeros((n_virtual, n_occupied))
        virtual_energies = orbital_energies[n_occupied:]
        occupied_energies = orbital_energies[:n_occupied]
        E_ai = virtual_energies - occupied_energies[:, None]
        reshaped_E_ai = E_ai.reshape(n_occupied*n_virtual)
        reshaped_a += np.diag(reshaped_E_ai)

        # Stack the matrices to create the combined matrix
        combined_matrix = np.vstack((np.hstack((reshaped_a, reshaped_b)), np.hstack((-reshaped_b, -reshaped_a))))

        return np.linalg.eigh(combined_matrix)
    sigma, vwc = drpa()
    reversed_sigma = sigma[n_occupied*n_virtual:][::-1]
    result = np.abs(sigma[:n_occupied*n_virtual]) - reversed_sigma
    print(result)

    
    mf = rks.RKS(molecule)
    mf.xc = 'hf'
    mf.verbose = 0
    mf.kernel()

    comp2 = dRPA(mf)
    comp2.nstates = n_occupied*n_virtual
    comp2.xc = 'hf'
    comp2.kernel()
    print(sigma)

    
    # now that us compute the V matrix
    W_pqia = np.sqrt(2)*eri_mo[:, :, :n_occupied, n_occupied:]
    # now that us reshape it into a form we want
    W_Ipq = W_pqia.reshape(n_orbitals, n_orbitals, n_occupied*n_virtual)
    V_pqn = np.einsum('pqi,in->pqn', W_Ipq, R)
    excitations = n_occupied*n_virtual
    # initialize the correlation energies
    correlation_energies = np.zeros((n_orbitals, n_orbitals))

    # from 11-30

    # make the V in the first sum
    exc_occ_vector = np.zeros((n_orbitals, n_occupied, excitations))
    exc_occ_vector += np.square(V_pqn[:, :n_occupied, :])
    occupied_denominator = np.zeros((n_occupied, excitations))
    # make the denominator
    occupied_denominator += (freq - orbital_energies[:n_occupied, None] + omega[None, :])
    # contract them
    first_some = np.einsum('pqu,qu->p', exc_occ_vector, 1/occupied_denominator)

    # make the V in the second sum
    exc_vir_vector = np.zeros((n_orbitals, n_virtual, excitations))
    exc_vir_vector += np.square(V_pqn[:, n_occupied:, :])
    virtual_denominator = np.zeros((n_virtual, excitations))
    # make the denominator
    virtual_denominator += (freq - orbital_energies[n_occupied:, None] - omega[None, :])
    # contract them
    second_some = np.einsum('pqu,qu->p', exc_vir_vector, 1/virtual_denominator)

    # add the fondle results
    correlation_energies += first_some[:, None] + second_some[:, None]

    # returned the diagonal
    return np.diag(correlation_energies)    
real_corr_se(0.0)