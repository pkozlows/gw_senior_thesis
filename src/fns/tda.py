import pyscf
from pyscf.dft import rks
from pyscf.tdscf.rks import dTDA, dRPA
from pyscf import tddft
import numpy as np
from src.fns.mf import setup_molecule, calculate_mean_field
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
    # front out the normalization
    normalization = R.T @ R

    # print(np.diag(normalization))

    # now that us compute the V matrix transition densities
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
    
    # Direct assignment to A and B matrices
    A = 2 * eri_mo[:n_occupied, n_occupied:, :n_occupied, n_occupied:].reshape(n_occupied*n_virtual, n_occupied*n_virtual)
    B = A.copy()  # Assuming B is the same as A in your case

    # Efficiently create and reshape the E_ai matrix using broadcasting
    virtual_energies = orbital_energies[n_occupied:]
    occupied_energies = orbital_energies[:n_occupied]
    E_ai = (virtual_energies - occupied_energies[:, None]).reshape(n_occupied*n_virtual)


    A += np.diag(E_ai)
    

    # Stack the matrices to create the combined matrix
    combined_matrix = np.vstack((np.hstack((A, B)), np.hstack((-B, -A))))
    omega, R = np.linalg.eig(combined_matrix) # note this is for a non-hermitian matrix

    # Extract positive and negative eigenvalue indices
    positive_indices = np.where(omega > 0)[0]
    negative_indices = np.where(omega < 0)[0]

    # Extract the actual eigenvalues using the indices
    positive_eigenvalues = omega[positive_indices]

    # Select corresponding eigenvectors
    R_positive = R[:, positive_indices]

    # Extract the positive and negative parts
    X_u = R_positive[:n_occupied*n_virtual, :]
    Y_u = R_positive[n_occupied*n_virtual:, :]

    # print(np.max(np.abs(normalization - np.diag(np.diag(normalization)))))
    # front out the normalization
    normalization = (X_u + Y_u).T @ (X_u-Y_u)
    # print(np.diag(normalization))

    combined_representation = (X_u + Y_u).copy()
    combined_representation_m = (X_u - Y_u).copy()
    for icol, col in enumerate(combined_representation):
        combined_representation[:,icol] /= np.sqrt(normalization[icol, icol])
        combined_representation_m[:,icol] /= np.sqrt(normalization[icol, icol])

    normalization = combined_representation.T @ combined_representation_m
    # print('Normalization \n', np.diag(normalization))

    # now that us compute the V matrix
    W_pqia = np.sqrt(2)*eri_mo[:, :, :n_occupied, n_occupied:]
    # now that us reshape it into a form we want
    W_pqu = W_pqia.reshape(n_orbitals, n_orbitals, n_occupied*n_virtual)
    V_pqu = np.einsum('pqi,in->pqn', W_pqu, combined_representation)

    return positive_eigenvalues, V_pqu

def symm_drpa(mf):
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
    
    # Direct assignment to A and B matrices
    A = 2 * eri_mo[:n_occupied, n_occupied:, :n_occupied, n_occupied:].reshape(n_occupied*n_virtual, n_occupied*n_virtual)
    B = A.copy()  # Assuming B is the same as A in your case

    # Efficiently create and reshape the E_ai matrix using broadcasting
    virtual_energies = orbital_energies[n_occupied:]
    occupied_energies = orbital_energies[:n_occupied]

    E_ai = (virtual_energies - occupied_energies[:, None]).reshape(n_occupied*n_virtual)



    A += np.diag(E_ai)
    

    # Stack the matrices to create the combined matrix
    AplusB = A + B
    AminusB = A - B

    combined_matrix = ((AminusB**.5) @ AplusB).T @ (AminusB**.5)
    omega_squared, T = np.linalg.eig(combined_matrix) # note this is for a non-hermitian matrix

    # Take the real part of the eigenvalues and eigenvectors
    omega_squared = np.real(omega_squared)
    T = np.real(T)

    XplusY = AminusB @ T

    
    # prepare the inverse of AminusB
    just_diag = np.diag(AminusB)
    inv_AminusB = np.diag(1/just_diag)

    XminusY = (inv_AminusB @ XplusY) @ np.diag(np.sqrt(omega_squared))

    # check the normalization
    normalization = XplusY.T @ XminusY
    # print(np.diag(normalization))

    combined_representation = (XplusY).copy()
    combined_representation_m = (XminusY).copy()
    for icol, col in enumerate(combined_representation):
        combined_representation[:,icol] /= np.sqrt(normalization[icol, icol])
        combined_representation_m[:,icol] /= np.sqrt(normalization[icol, icol])

    normalization = combined_representation.T @ combined_representation_m
    # print('Normalization \n', np.diag(normalization))
    # # make a few assert statements
    # # Calculate the transformed matrix for AminusB
    # transformed_AminusB = XminusY.T @ AminusB @ XminusY
    # expected = np.diag(np.sqrt(omega_squared))
    # assert np.allclose(expected, transformed_AminusB, atol=1e-6), \
    #     f"Failed AminusB Transformation: Expected {expected}, got {transformed_AminusB}, diff {expected - transformed_AminusB}"

    # # Calculate the transformed matrix for AplusB
    # transformed_AplusB = XplusY.T @ AplusB @ XplusY
    # assert np.allclose(expected, transformed_AplusB, atol=1e-6), \
    #     f"Failed AplusB Transformation: Expected {expected}, got {transformed_AplusB}, diff {expected - transformed_AplusB}"


    # convert omega_squared and comb
    # now that us compute the V matrix
    W_pqia = np.sqrt(2)*eri_mo[:, :, :n_occupied, n_occupied:]
    # now that us reshape it into a form we want
    W_pqu = W_pqia.reshape(n_orbitals, n_orbitals, n_occupied*n_virtual)
    V_pqu = np.einsum('pqi,in->pqn', W_pqu, combined_representation)
    

    return np.sqrt(omega_squared), V_pqu
       

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