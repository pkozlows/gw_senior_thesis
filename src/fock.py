import numpy as np
import pyscf
from mf import setup_molecule, calculate_mean_field
def fock_matrix_hf(molecule):
    '''Calculates the fock matrix for a given set of molecular orbitals and occupations.'''

    
    mf, n_orbitals, n_occupied, n_virtual, orbital_energies, eri = calculate_mean_field(molecule, 'hf')
    
    # initialize the fock matrix
    fock = np.zeros((n_orbitals, n_orbitals))
    fock += np.diag(orbital_energies)
    return fock
  
def fock_dft(molecule):
    '''Calculates the HF fock matrix using the DFT electron density in AO basis for a given set of molecular orbitals and occupations.'''

    mf, n_orbitals, n_occupied, n_virtual, orbital_energies, eri = calculate_mean_field(molecule, 'dft')

    # initialize the fock matrix in the atomic orbital basis
    fock = np.zeros((n_orbitals, n_orbitals))
    h_core = mf.get_hcore()
    # in nationalize the codon matrix
    coulumb_matrix = np.zeros((n_orbitals, n_orbitals))
    coulumb_matrix += np.einsum('rs,pqrs->pq', mf.make_rdm1(), eri)
    # initialize the exchange term
    exchange_matrix = np.zeros((n_orbitals, n_orbitals))
    exchange_matrix += np.einsum('rs,prqs->pq', mf.make_rdm1(), eri)
    # add the terms
    fock += (h_core + coulumb_matrix - 0.5*exchange_matrix)
    return fock

molecule = pyscf.M(
    atom = 'O  0 0 0; H  0 0.758602 0.504284; H  0 0.758602 -0.504284',
    basis = 'ccpvdz',
    symmetry = True,
)
# check whether the fog matrix is diaconal and find its dimensions
fock = fock_matrix_hf(molecule)
assert(np.allclose(fock, fock.T))
print(fock.shape)