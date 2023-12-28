import pyscf
from pyscf.dft import rks
from pyscf.tdscf.rks import dTDA, dRPA
from pyscf import tddft
import numpy as np
from mf import setup_molecule, calculate_mean_field
import numpy as np

def my_dtda(mf):  
    '''Calculates the excitation energies and the R matrix for the molecule in the direct tda.

    Args:
        mf (object): An object representing the molecule.

    Returns:
        tuple: A tuple containing the excitation energies (omega) and the V matrix (V_pqu).
    '''

    n_orbitals = mf.mol.nao_nr()
    n_occupied = mf.mol.nelectron//2
    n_virtual = n_orbitals - n_occupied
    # get the orbital energies
    orbital_energies = mf.mo_energy
    # get the MO integrals
    eri_mo = mf.mol.ao2mo(mf.mo_coeff, compact=False).reshape(n_orbitals, n_orbitals, n_orbitals, n_orbitals)
    
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

    omega, R = np.linalg.eigh(reshaped_a)

    # now that us compute the V matrix
    W_pqia = np.sqrt(2)*eri_mo[:, :, :n_occupied, n_occupied:]
    # now that us reshape it into a form we want
    W_pqu = W_pqia.reshape(n_orbitals, n_orbitals, n_occupied*n_virtual)
    V_pqu = np.einsum('pqi,in->pqn', W_pqu, R)
    return omega, V_pqu

def my_drpa(mf):
    '''Calculates the excitation energies and the R matrix for the molecule in the direct rpa.

    Args:
        mf (object): An object representing the molecule.

    Returns:
        tuple: A tuple containing the excitation energies and the R matrix.
    '''

    n_orbitals = mf.mol.nao_nr()
    n_occupied = mf.mol.nelectron//2
    n_virtual = n_orbitals - n_occupied
    # get the orbital energies
    orbital_energies = mf.mo_energy
    # get the MO integrals
    eri_mo = mf.mol.ao2mo(mf.mo_coeff, compact=False).reshape(n_orbitals, n_orbitals, n_orbitals, n_orbitals)
    
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
    omega, R = np.linalg.eigh(combined_matrix)



    # def compare_omega(omega):
    #     # Sort the omega array
    #     omega_sorted = np.sort(omega)

    #     # Split into positive and negative parts
    #     positives = omega_sorted[omega_sorted > 0]
    #     negatives = omega_sorted[omega_sorted < 0]

    #     # Find closest matches and calculate differences
    #     matches = []
    #     for pos in positives:
    #         closest_neg = min(negatives, key=lambda x: abs(x + pos))
    #         difference = abs(pos + closest_neg)
    #         matches.append((pos, closest_neg, difference))

    #     # Display the results
    #     print(f"{'Positive':>10} {'Negative':>10} {'Difference':>10}")
    #     for pos, neg, diff in matches:
    #         print(f"{pos:>10.5f} {neg:>10.5f} {diff:>10.5f}")

    #     # Summary statistics
    #     differences = [abs(p[0] + p[1]) for p in matches]
    #     print("\nSummary Statistics:")
    #     print(f"Average Difference: {np.mean(differences):.5f}")
    #     print(f"Maximum Difference: {np.max(differences):.5f}")
    #     return

    # compare_omega(omega)
    def keep_unique_positive(omega):
        # Keep only positive values
        positive_omega = omega[omega > 0]

        # Optionally, sort and ensure uniqueness
        unique_positive_omega = np.unique(np.sort(positive_omega))

        return unique_positive_omega



                



    return keep_unique_positive(omega), R



def real_corr_se(freq, tddft, mf):
    '''
    Calculates the real part of the correlation self energy for a given molecule and frequency.
    Returns a matrix of correlation energies for each orbital at the given frequency.
    Only the diagonal of this matrix is considered.

    Parameters:
    - freq: float, the frequency at which to calculate the correlation self energy
    - tddft: function, a function that calculates the excitation energies and the R matrix
    - mf: object, the molecule object containing basic information like the number of orbitals, occupied orbitals, and virtual orbitals

    Returns:
    - correlation_energies: numpy array, a matrix of correlation energies for each orbital at the given frequency (only the diagonal is considered)
    '''

    # I want to get basic information like the number of orbitals, occupied orbitals, and virtual orbitals
    n_orbitals = mf.mol.nao_nr()
    n_occupied = mf.mol.nelectron//2
    n_virtual = n_orbitals - n_occupied
    # get the orbital energies
    orbital_energies = mf.mo_energy
    

    # # now we want to calculate the excitation energies and the R matrix
    omega, V_pqu = tddft(mf)
    # omega, R = my_drpa(mf)

    excitations = n_occupied*n_virtual
    # initialize the correlation energies
    correlation_energies = np.zeros((n_orbitals, n_orbitals))

    # from 11-30

      # make the V in the first sum
    exc_occ_vector = np.zeros((n_orbitals, n_occupied, excitations))
    exc_occ_vector += np.square(V_pqu[:, :n_occupied, :])
    occupied_denominator = np.zeros((n_occupied, excitations))
    # make the denominator
    occupied_denominator += (freq - orbital_energies[:n_occupied, None] + omega[None, :])
    # contract them
    first_some = np.einsum('pqu,qu->p', exc_occ_vector, 1/occupied_denominator)

    # make the V in the second sum
    exc_vir_vector = np.zeros((n_orbitals, n_virtual, excitations))
    exc_vir_vector += np.square(V_pqu[:, n_occupied:, :])
    virtual_denominator = np.zeros((n_virtual, excitations))
    # make the denominator
    virtual_denominator += (freq - orbital_energies[n_occupied:, None] - omega[None, :])
    # contract them
    second_some = np.einsum('pqu,qu->p', exc_vir_vector, 1/virtual_denominator)

    # add the fondle results
    correlation_energies += first_some[:, None] + second_some[:, None]

    # returned the diagonal
    return np.diag(correlation_energies)
