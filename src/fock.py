import numpy as np
import pyscf
from mf import setup_molecule, calculate_mean_field
def fock_matrix_hf(molecule):
    '''Calculates the fock matrix for a given set of molecular orbitals and occupations.'''

    
    molecule = setup_molecule()
    mf, n_orbitals, n_occupied, n_virtual, orbital_energies, eri = calculate_mean_field(molecule, 'hf')
    
    # first we want to find the charge density matrix
    # initialize the fock matrix
    fock = np.zeros((n_orbitals, n_orbitals))
    # loop over all orbitals twice
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            # check if we are dealing with a day cando element
            if p == q:
                fock[p,q] += orbital_energies[p]
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